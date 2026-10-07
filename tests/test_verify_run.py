"""Offline archives/backend; installed SDK smoke mocks auth/APIs and never uses VMs."""

import importlib.util
import ast
import json
import os
import subprocess
import signal
import sys
import time
import zipfile
import datetime
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("verify_run", Path(__file__).resolve().parents[1] / "tools/verify_run.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def dump(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def lines(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def fixture(tmp_path, count=1, condition="JFINAL_G4", latency=False, n_dev=20, n_synth=3,
            code_version="0.2.0", frozen_code_version=None, config_name="experiment_v2.json"):
    inputs = tmp_path / ("dev_inputs.jsonl" if latency else "pilot_inputs.jsonl")
    input_rows = [{"id": f"case-{i:03}", "problem": f"Problem {i}", "language": "en"}
                  for i in range(n_dev if latency else count)]
    inputs.write_text(lines(input_rows), encoding="utf-8")
    prompts = tmp_path / "prompts"; prompts.mkdir()
    for name in ("generador", "criterio_paso", "criterio_final"):
        (prompts / (name + ".txt")).write_text(name + "\n", encoding="utf-8")
    arms = ["G4_SINGLE", "B13", "B13_GREEDY", "J64_G4", "JSTEP_G4", "JFINAL_G4"]
    frozen = {"models": {role: {"repo": "fake-" + role, "revision": "fixed"} for role in ("G", "J", "B")},
              "implementation_profile": "1COPY-G4-VLLM-GRAPH", "jevk5_runtime": {"commit": "frozen"},
              "protocol_version": "v2", "conditions": arms}
    if frozen_code_version is not None:
        frozen["code_version"] = frozen_code_version
    config = tmp_path / config_name; config.write_text(json.dumps(frozen))
    cfg = {"params": {"CONDITION": condition, "RUN_TAG": "fake", "SPLIT": "dev" if latency else "pilot",
                      "SEEDS": "17", "KV_CACHE_GB_G": 3.0, "ENFORCE_EAGER": False,
                      "LAT_N_DEV": n_dev, "LAT_SYNTH_PER_BAND": n_synth},
           "models": frozen["models"], "profile": frozen["implementation_profile"],
           "jevk5_runtime": frozen["jevk5_runtime"], "protocol_version": "v2", "code_version": code_version,
           "experiment_config_sha256": verify.sha256(config.read_bytes()),
           "data_sha256": {inputs.name: verify.sha256(inputs.read_bytes())},
           "prompt_sha256": {name: verify.sha256(name.encode()) for name in ("generador", "criterio_paso", "criterio_final")}}
    digest = verify.sha256(dump(cfg).encode())
    run = f"{condition}_fake_{digest[:8]}"
    members = {}
    if latency:
        items = [{"item_id": r["id"], "kind": "dev", "problem": r["problem"]} for r in input_rows]
        items += [{"item_id": f"synth-{band}-{k}", "kind": "synthetic_" + band, "problem": "synthetic"}
                  for band in ("short", "medium", "long", "xlong") for k in range(n_synth)]
        items.reverse()  # The real latency plan shuffles; ordered comparisons are wrong.
        plan = {"items": items, "blocks": [[item["item_id"] for item in items]]}
        members["latency_plan.json"] = json.dumps(plan)
        rows = [{"item_id": item["item_id"], "condition": arm, "seed": 17, "kind": item["kind"],
                 "block": 0, "phase": "B" if arm.startswith("B13") else "H", "status": "eos_invalid",
                 "config_hash": digest, "t_start_utc": "start", "t_end_utc": "end"}
                for item in items for arm in arms]
        members["latency_metrics.jsonl"] = lines(rows)
        counts = {"latency_metrics": len(rows), "failures": 0}
    else:
        rows, candidates, decisions = [], [], []
        for index, item in enumerate(reversed(input_rows)):
            pid = item["id"]
            key = verify.sha256(f"{digest}|{pid}|17|{condition}|{cfg['profile']}".encode())
            common = {"problem_id": pid, "condition": condition, "seed": 17, "resume_key": key,
                      "config_hash": digest, "profile": cfg["profile"], "run_name": run}
            rows.append({**common, "status": ("final", "eos_invalid", "truncated")[index % 3],
                         "t_start_utc": "start", "t_end_utc": "end"})
            if condition == "JFINAL_G4":
                candidates.extend({**common, "round": 0, "branch": b, "chosen": b == 2} for b in range(4))
                decisions.append({**common, "round": 0, "status": "ok", "permutation": [3, 1, 2, 0],
                                  "winner_position": 2, "winner_branch": 2})
        members.update({"predictions.jsonl": lines(rows), "metrics.jsonl": lines(rows),
                        "candidates.jsonl": lines(candidates), "decisions.jsonl": lines(decisions)})
        counts = {"predictions": len(rows), "metrics": len(rows), "candidates": len(candidates),
                  "decisions": len(decisions), "failures": 0}
    manifest = {"condition": condition, "run_name": run, "config": cfg, "config_hash": digest,
                "finished_utc": "finished", "counts": counts, "stages": {"main": {"ok": True}},
                "preflight": {"check": {"blocking": True, "ok": True}},
                "model_lock": {role: {"ok": True} for role in ("G", "J", "B")},
                "environment": {"gpus": [{"name": "NVIDIA A100-SXM4-40GB", "memory.total": "40960"}]}}
    members["manifest.json"] = json.dumps(manifest)
    members["progress.json"] = json.dumps({"done": len(rows), "total": len(rows)})
    artifact = tmp_path / (run + "_final.zip")
    prefix = "08" if latency else ("06" if condition == "JFINAL_G4" else "01" if condition in ("G_SINGLE", "G4_SINGLE") else "02")
    notebook = tmp_path / (prefix + "_fake.out.timestamp.ipynb")
    nb = {"metadata": {"papermill": {"end_time": "end", "exception": False}},
          "cells": [{"cell_type": "code", "source": ["print('done')"], "outputs": [],
                     "metadata": {"papermill": {"status": "completed"}}}]}
    notebook.write_text(json.dumps(nb))
    options = dict(inputs=inputs, config=config, expected_count=count, split=cfg["params"]["SPLIT"],
                   run_tag="fake", prompts=prompts, latency_rows=len(rows), latency_dev_count=n_dev)
    return artifact, members, notebook, options


def archive(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in members.items():
            z.writestr("condition/run/" + name, text)


@pytest.mark.parametrize("count", [1, 40, 7])
def test_smoke_pilot_actual_n_exact_membership(tmp_path, count):
    path, members, nb, options = fixture(tmp_path, count=count)
    archive(path, members)
    report = verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)
    assert report["ok"], report
    assert report["runs"]["JFINAL_G4"]["records"] == count


@pytest.mark.parametrize("fault", ["missing_id", "duplicate_id", "wrong_id", "wrong_condition", "wrong_seed",
                                  "wrong_hash", "missing_candidates", "wrong_decision", "wrong_metrics",
                                  "exception", "no_final_marker", "stage_failed", "preflight_failed", "input_changed"])
def test_rejects_corrupt_or_incomplete_final(tmp_path, fault):
    path, members, nb, options = fixture(tmp_path, count=3)
    rows = verify.jsonl(members["predictions.jsonl"])
    if fault == "missing_id":
        rows.pop()
    elif fault == "duplicate_id":
        rows[-1] = rows[0]
    elif fault == "wrong_id":
        rows[0]["problem_id"] = "not-an-input"
    elif fault == "wrong_condition":
        rows[0]["condition"] = "B13"
    elif fault == "wrong_seed":
        rows[0]["seed"] = 29
    elif fault == "wrong_hash":
        rows[0]["config_hash"] = "bad"
    elif fault == "exception":
        rows[0]["status"] = "exception"
    members["predictions.jsonl"] = lines(rows)
    if fault == "missing_candidates":
        members["candidates.jsonl"] = lines(verify.jsonl(members["candidates.jsonl"])[1:])
    elif fault == "wrong_decision":
        ds = verify.jsonl(members["decisions.jsonl"]); ds[0]["winner_branch"] = 1
        members["decisions.jsonl"] = lines(ds)
    elif fault == "wrong_metrics":
        ms = verify.jsonl(members["metrics.jsonl"]); ms[0]["problem_id"] = "wrong"
        members["metrics.jsonl"] = lines(ms)
    elif fault in ("no_final_marker", "stage_failed", "preflight_failed"):
        manifest = json.loads(members["manifest.json"])
        if fault == "no_final_marker":
            del manifest["finished_utc"]
        elif fault == "stage_failed":
            manifest["stages"]["main"]["ok"] = False
        else:
            manifest["preflight"]["check"]["ok"] = False
        members["manifest.json"] = json.dumps(manifest)
    elif fault == "input_changed":
        options["inputs"].write_text(options["inputs"].read_text() + "\n")
    archive(path, members)
    assert not verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]


@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("exception", [None, False])
def test_actual_notebook_formats_and_interruption(tmp_path, literal, exception):
    path, members, nb, options = fixture(tmp_path)
    archive(path, members)
    notebook = json.loads(nb.read_text())
    notebook["metadata"]["papermill"]["exception"] = exception
    nb.write_text(repr(notebook) if literal else json.dumps(notebook))
    assert verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]
    notebook["metadata"]["papermill"]["end_time"] = None
    nb.write_text(repr(notebook) if literal else json.dumps(notebook))
    assert not verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]


@pytest.mark.parametrize('series', ['v1', 'v2', 'v2-reload'])
def test_operator_python_and_runtime_launch_template_compile(series):
    root = Path(__file__).resolve().parents[1]
    operator = (root / 'tools/colab/operator.sh').read_text(encoding='utf-8')
    worker = operator.split('exec python3 - ', 1)[1].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    compile(worker, 'operator-worker', 'exec')
    launcher = (root / 'tools/colab/launch_seq.sh').read_text(encoding='utf-8')
    remote = launcher.split('cat > "$TMP" <<PY\n', 1)[1].split('\nPY', 1)[0]
    remote = remote.replace('$LIST_PY', "['JFINAL:06_JFINAL']").replace('$ARGS_PY', "['-p', 'RUN_TAG', 'fake']")
    remote = remote.replace('$SERIES', series)
    tree = ast.parse(remote)
    compile(tree, 'remote-launch-template', 'exec')
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'driver')
    driver = eval(compile(ast.Expression(call.args[0]), 'driver-builder', 'eval'),
                  {'items': ['JFINAL:06_JFINAL'], 'args': ['-p', 'RUN_TAG', 'fake'], 'ts': 'timestamp'})
    compile(driver, 'generated-papermill-driver', 'exec')
    alias_branch = next(node for node in tree.body if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], ast.In))
    uses_v2_alias = eval(compile(ast.Expression(alias_branch.test), 'series-root-routing', 'eval'))
    assert uses_v2_alias is (series in ('v2', 'v2-reload'))


@pytest.mark.parametrize('series,config_file,data_subdir,run_tag', [
    ('v1', 'config/experiment.json', 'data', 'main'),
    ('v2', 'config/experiment_v2.json', 'data/v2', 'v2'),
    ('v2-reload', 'config/experiment_v2_reload.json', 'data/v2', 'v2-reload')])
def test_operator_state_defaults_resolve_reload_config_for_audit(series, config_file, data_subdir, run_tag):
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/operator.sh').read_text()
    tree = ast.parse(source.split('exec python3 - ', 1)[1].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    defaults = [node for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('series', 'v2_series')]
    update = next(node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id == 'state' and node.value.func.attr == 'update')
    ns = {'os': SimpleNamespace(environ={'JEV_SERIES': series}, getpid=lambda: 1), 'state': {},
          'deadline': 14400, 'work_deadline': 14040, 'inputs': '/fake/inputs.jsonl', 'count': 1,
          'mirror': '/fake/checkpoints', 'art': '/fake/artifacts', 'guard': '/fake/guard',
          'conditions': ['LATENCY'], 'pm_args': []}
    exec(compile(ast.Module(body=[*defaults, update], type_ignores=[]), 'operator-defaults', 'exec'), ns)
    assert ns['state']['series'] == series
    assert ns['state']['config_file'] == config_file
    assert ns['state']['data_subdir'] == data_subdir
    assert ns['state']['run_tag'] == run_tag
    assert ns['state']['split'] == 'dev'


@pytest.mark.parametrize("n_dev,n_synth,expected", [(20, 3, 192), (2, 1, 36)])
@pytest.mark.parametrize("reload", [False, True])
def test_latency_rows_and_shuffled_membership(tmp_path, n_dev, n_synth, expected, reload):
    path, members, nb, options = fixture(tmp_path, condition="LATENCY", latency=True, n_dev=n_dev, n_synth=n_synth,
                                        code_version="0.2.1" if reload else "0.2.0",
                                        frozen_code_version="0.2.1" if reload else None)
    archive(path, members)
    assert options["latency_rows"] == expected
    assert verify.verify_run(tmp_path, conditions=["LATENCY"], **options)["ok"]
    metrics = verify.jsonl(members["latency_metrics.jsonl"])
    metrics[-1] = metrics[0]
    members["latency_metrics.jsonl"] = lines(metrics)
    archive(path, members)
    assert not verify.verify_run(tmp_path, conditions=["LATENCY"], **options)["ok"]


@pytest.mark.parametrize('frozen_version,recorded_version,ok', [
    (None, '0.2.0', True), (None, '0.2.1', False), ('0.2.1', '0.2.1', True),
    ('0.2.1', '0.2.0', False), ('0.2.0', '0.2.0', True), ('0.2.0', '0.2.1', False)])
def test_frozen_code_version_preserves_old_smoke_and_pins_reload(tmp_path, frozen_version, recorded_version, ok):
    path, members, nb, options = fixture(tmp_path, condition='G_SINGLE', code_version=recorded_version,
                                        frozen_code_version=frozen_version,
                                        config_name='experiment_v2_reload.json' if frozen_version == '0.2.1' else 'experiment_v2.json')
    frozen_bytes = options['config'].read_bytes()
    archive(path, members)
    report = verify.verify_run(tmp_path, conditions=['G_SINGLE'], **options)
    assert report['ok'] is ok, report
    assert options['config'].read_bytes() == frozen_bytes
    if not ok:
        assert 'v2 code version mismatch' in report['runs']['G_SINGLE']['error']


def test_completed_g_only_audit_does_not_complete_failed_latency_combo(tmp_path, monkeypatch):
    path, members, nb, options = fixture(tmp_path, condition='G_SINGLE')
    archive(path, members)
    failed = tmp_path / '08_estudio_latencia.out.timestamp.ipynb'
    failed.write_text(json.dumps({'metadata': {'papermill': {'exception': True}}, 'cells': []}))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps({'status': 'failed', 'verified': False, 'released': True,
                                 'conditions': ['G_SINGLE', 'LATENCY']}))
    original_status = status.read_bytes()
    output = tmp_path / 'gsingle_verification.json'
    monkeypatch.setattr(sys, 'argv', ['verify_run', str(tmp_path), '--conditions', 'G_SINGLE',
                                    '--inputs', str(options['inputs']), '--config', str(options['config']),
                                    '--prompts', str(options['prompts']), '--expected-count', '1',
                                    '--split', 'pilot', '--run-tag', 'fake', '--output', str(output)])
    with pytest.raises(SystemExit) as result:
        verify.main()
    assert result.value.code == 0
    report = json.loads(output.read_text())
    assert report['ok'] and set(report['runs']) == {'G_SINGLE'}
    assert report['runs']['G_SINGLE']['records'] == 1
    assert status.read_bytes() == original_status
    assert not verify.verify_run(tmp_path, conditions=['G_SINGLE', 'LATENCY'], **options)['ok']


def test_zip_integrity_and_ambiguous_finals(tmp_path):
    path, members, nb, options = fixture(tmp_path)
    path.write_bytes(b"not a ZIP")
    assert not verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]
    archive(path, members)
    (tmp_path / "JFINAL_G4_fake_second_final.zip").write_bytes(path.read_bytes())
    assert not verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]


def test_smoke_selects_expected_ids_not_arbitrary_subset(tmp_path):
    path = tmp_path / "dev_inputs.jsonl"
    path.write_text(lines([{"id": "c"}, {"id": "a"}, {"id": "b"}]))
    assert verify.expected_ids(path, 1, "dev") == {"a"}
    with pytest.raises(ValueError):
        verify.expected_ids(path, 4, "dev")


def test_two_conditions_not_confused_by_b13_prefix(tmp_path):
    path, members, nb, options = fixture(tmp_path, condition="B13")
    archive(path, members)
    (tmp_path / "B13_GREEDY_fake_other_final.zip").write_bytes(path.read_bytes())
    assert verify.verify_run(tmp_path, conditions=["B13"], **options)["ok"]


def test_persisted_canonical_copy_is_not_a_retry(tmp_path):
    path, members, nb, options = fixture(tmp_path)
    archive(path, members)
    nb.with_name(nb.stem + '.canonical.ipynb').write_text(nb.read_text())
    assert verify.verify_run(tmp_path, conditions=["JFINAL_G4"], **options)["ok"]


@pytest.mark.parametrize("mode,verify_rc,release_ok,interrupted,guard_fault,condition", [
    ('launch', 0, True, False, None, 'JFINAL_G4'), ('launch', 1, True, False, None, 'JFINAL_G4'),
    ('launch', 0, False, False, None, 'JFINAL_G4'), ('launch', 0, True, True, None, 'JFINAL_G4'),
    ('monitor', 0, True, False, None, 'JFINAL_G4'),
    ('launch', 0, True, False, 'live_recovery', 'JFINAL_G4'),
    ('launch', 0, True, False, 'sustained', 'JFINAL_G4'),
    ('launch', 0, True, False, 'final_recovery', 'JFINAL_G4'),
    ('launch', 0, True, False, None, 'LATENCY')])
@pytest.mark.parametrize('series', ['v2', 'v2-reload'])
def test_operator_fake_backend_verifies_and_releases_only_owned_session(tmp_path, monkeypatch, mode, verify_rc, release_ok, interrupted, guard_fault, condition, series):
    root = tmp_path / 'project'; root.mkdir()
    path, members, nb, options = fixture(root, condition=condition, latency=condition == 'LATENCY')
    output = tmp_path / 'operator'; output.mkdir()
    config_path = output / 'sessions.json'
    start = time.time()
    (output / 'execution_handover.json').write_text(json.dumps({
        'session': 'owned-session', 'start_epoch': start, 'deadline_epoch': start + 14400}))
    monkeypatch.setenv('JEV_ROOT', str(root))
    monkeypatch.setenv('JEV_SERIES', series)
    monkeypatch.setenv('COLAB_SESSION_CONFIG', str(config_path))
    monkeypatch.setenv('JEV_CONFIG_FILE', options['config'].name)
    monkeypatch.setenv('JEV_SPLIT', 'pilot')
    monkeypatch.setenv('JEV_RUN_TAG', 'fake')
    monkeypatch.setattr(sys, 'argv', ['worker', mode, 'owned-session', condition, str(output), '1', str(options['inputs'])])
    clock = {'now': start}
    if guard_fault:
        monkeypatch.setattr(time, 'time', lambda: clock['now'])
        monkeypatch.setattr(time, 'sleep', lambda seconds: clock.update(now=clock['now'] + seconds))
    backend = {'present': mode == 'monitor', 'local': mode == 'monitor'}
    if mode == 'monitor':
        config_path.write_text(json.dumps({'owned-session': {'endpoint': 'owned-endpoint'}}))
    commands = []

    class FakeProcess:
        pid = 999999

        def __init__(self, command, stdout, **kwargs):
            commands.append(command)
            self.rc = 0
            if 'colab_identity' in ' '.join(command):
                stdout.write(json.dumps({'assignment_count': 1 + int(backend['present']),
                                         'local_exists': backend['local'], 'endpoint': 'owned-endpoint' if backend['local'] or command[-1] else None,
                                         'present': backend['present'], 'accelerator': 'A100' if backend['present'] else None}) + '\n')
            elif len(command) > 1 and command[1].endswith('launch_seq.sh'):
                backend.update(present=True, local=True)
                config_path.write_text(json.dumps({'owned-session': {'endpoint': 'owned-endpoint'}}))
            elif 'cexec' in ' '.join(command):
                if interrupted:
                    raise InterruptedError('simulated parent interruption')
                alive = guard_fault == 'sustained' or (guard_fault == 'live_recovery' and clock['now'] - start < 125)
                stdout.write('HEALTH_JSON::' + json.dumps({'known': True, 'alive': alive, 'conditions': {},
                              'sequence_status': {'_done': True, condition: {'rc': 0}}}) + '\n')
            elif len(command) > 1 and command[1].endswith('verify_run.py'):
                self.rc = verify_rc
            elif len(command) > 1 and command[1].endswith('stop.sh'):
                assert command[-1] == 'owned-session'
                if release_ok:
                    backend.update(present=False, local=False)
                    config_path.write_text('{}')
                else:
                    self.rc = 1
            stdout.flush()

        def wait(self, timeout=None):
            return self.rc

    monkeypatch.setattr(subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(os, 'killpg', lambda *args: None, raising=False)
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/operator.sh').read_text()
    worker = source.split('exec python3 - ', 1)[1].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    tree = ast.parse(worker)
    # Filesystem heartbeat parsing is tested separately; simulate guard health here.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'guard_running':
            node.body = [ast.Return(value=ast.Constant(value=True))]
        elif isinstance(node, ast.FunctionDef) and node.name == 'guard_healthy':
            node.body = [ast.Return(value=ast.Call(func=ast.Name(id='test_guard_healthy', ctx=ast.Load()), args=[], keywords=[]))]
    ast.fix_missing_locations(tree)

    def healthy():
        return not guard_fault or (guard_fault != 'sustained' and clock['now'] - start >= 120)

    with pytest.raises(SystemExit) as exit_info:
        exec(compile(tree, 'simulated-worker', 'exec'), {'__name__': '__main__', 'test_guard_healthy': healthy})
    result = json.loads((output / 'status.json').read_text())
    success = verify_rc == 0 and not interrupted and guard_fault != 'sustained'
    assert result['verified'] is success
    assert result['released'] is release_ok
    assert result['series'] == series
    assert result['status'] == ('completed' if success and release_ok else 'failed')
    assert exit_info.value.code == (0 if success and release_ok else 1)
    assert any(command[1].endswith('stop.sh') for command in commands if len(command) > 1)
    assert not any('all' in command for command in commands)
    assert sum(command[1].endswith('launch_seq.sh') for command in commands if len(command) > 1) == (0 if mode == 'monitor' else 1)
    if mode == 'monitor':
        assert not any(command[1].endswith('session_guard.sh') for command in commands if len(command) > 1)
    if guard_fault:
        assert not any(command[1].endswith('session_guard.sh') for command in commands if len(command) > 1)
        if guard_fault == 'sustained':
            assert clock['now'] - start == 180
            assert result['guard_degraded'] and 'sustained failure' in result['error']
        else:
            assert not result['guard_degraded'] and result['guard_recovery_started_epoch'] is None
    if condition == 'LATENCY':
        audit = next(command for command in commands if len(command) > 1 and command[1].endswith('verify_run.py'))
        assert result['split'] == audit[audit.index('--split') + 1] == 'dev'


def guard_harness(tmp_path, *, running=True, recovery_started=None):
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/operator.sh').read_text()
    tree = ast.parse(source.split('exec python3 - ', 1)[1].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    names = {'guard_healthy', 'maintain_guard', 'wait_for_guard'}
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    clock = {'now': 1000.0, 'running': running}
    config = tmp_path / 'sessions.json'
    config.write_text(json.dumps({'ours': {'endpoint': 'owned'}}))
    guard = tmp_path / 'guard'; guard.mkdir()
    (guard / 'session_guard_identity.json').write_text(json.dumps({'session': 'ours', 'endpoint': 'owned'}))
    (guard / 'session_lifecycle.jsonl').write_text('')
    state, calls, snapshots = {}, [], []

    def event(kind, when=None, **fields):
        when = clock['now'] if when is None else when
        record = {'event': kind, 'utc': datetime.datetime.fromtimestamp(when, datetime.timezone.utc).isoformat(), **fields}
        with (guard / 'session_lifecycle.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def write(status, **fields):
        state.update(status=status, **fields)
        snapshots.append(dict(state))

    def call(command, timeout, name):
        calls.append(command)
        clock['running'] = True
        event('guard_started')
        return 0

    namespace = {'json': json, 'pathlib': __import__('pathlib'), 'datetime': datetime,
                 'os': SimpleNamespace(environ={'COLAB_SESSION_CONFIG': str(config)}),
                 'time': SimpleNamespace(time=lambda: clock['now'], sleep=lambda seconds: clock.update(now=clock['now'] + seconds)),
                 'guard': guard, 'session': 'ours', 'owner': 'owned', 'root': tmp_path,
                 'state': state, 'write': write, 'call': call, 'work_deadline': 2000,
                 'guard_running': lambda: clock['running'], 'guard_recovery_started': recovery_started,
                 'guard_restart_attempt': None}
    exec(compile(functions, 'guard-functions', 'exec'), namespace)
    return SimpleNamespace(ns=namespace, clock=clock, event=event, state=state, calls=calls,
                           snapshots=snapshots, guard=guard, config=config)


@pytest.mark.parametrize('age,latest,healthy', [
    (10, 'health_unknown', True), (64, 'health_unknown', True), (65, 'health_unknown', False),
    (90, 'health_unknown', False), (-1, 'health_unknown', False),
    (10, 'guard_started', False), (10, 'assignment_absent', False), (10, 'guard_stopped', False)])
def test_guard_uses_recent_positive_evidence_not_last_event(tmp_path, age, latest, healthy):
    h = guard_harness(tmp_path)
    h.event('heartbeat', h.clock['now'] - age, assignment_present=True, socket_connected=True)
    h.event(latest)
    assert h.ns['guard_healthy']() is healthy
    if healthy:
        assert h.ns['maintain_guard']()
        assert not h.calls and not h.state['guard_degraded']


def test_unknown_transport_alone_is_not_healthy_and_live_guard_not_restarted(tmp_path):
    h = guard_harness(tmp_path)
    h.event('health_unknown')
    assert not h.ns['maintain_guard']()
    assert h.state['guard_degraded'] and h.state['guard_recovery_started_epoch'] == 1000
    h.clock['now'] += 179
    assert not h.ns['maintain_guard']()
    assert not h.calls
    h.clock['now'] += 1
    with pytest.raises(RuntimeError, match='180 seconds'):
        h.ns['maintain_guard']()
    assert h.state['guard_recovery_started_epoch'] == 1000


def test_recovery_clears_only_on_fresh_connected_heartbeat(tmp_path):
    h = guard_harness(tmp_path)
    assert not h.ns['maintain_guard']()
    h.clock['now'] += 100
    h.event('heartbeat', assignment_present=True, socket_connected=False)
    assert not h.ns['maintain_guard']()
    h.event('heartbeat', assignment_present=True, socket_connected=True)
    assert h.ns['maintain_guard']()
    assert h.state['guard_healthy'] and not h.state['guard_degraded']
    assert h.state['guard_recovery_started_epoch'] is None and not h.calls


def test_dead_owned_guard_restarts_immediately_without_crediting_old_heartbeat(tmp_path):
    h = guard_harness(tmp_path, running=False)
    h.event('heartbeat', 990, assignment_present=True, socket_connected=True)
    assert not h.ns['maintain_guard']()
    assert len(h.calls) == 1 and h.calls[0][2:4] == ['start', 'ours']
    assert not h.ns['guard_healthy']()  # The new guard must supply its own connection evidence.
    assert not h.ns['maintain_guard']() and len(h.calls) == 1
    h.event('heartbeat', assignment_present=True, socket_connected=True)
    assert h.ns['maintain_guard']()


def test_failed_dead_guard_restart_is_bounded_and_does_not_reset_recovery(tmp_path):
    h = guard_harness(tmp_path, running=False)

    def failed_start(command, timeout, name):
        h.calls.append(command)
        return 1

    h.ns['call'] = failed_start
    assert not h.ns['maintain_guard']() and h.state['guard_restart_rc'] == 1
    h.clock['now'] += 19
    assert not h.ns['maintain_guard']() and len(h.calls) == 1
    h.clock['now'] += 1
    assert not h.ns['maintain_guard']() and len(h.calls) == 2
    h.clock['now'] = 1180
    with pytest.raises(RuntimeError, match='180 seconds'):
        h.ns['maintain_guard']()
    assert h.state['guard_recovery_started_epoch'] == 1000


@pytest.mark.parametrize('mismatch', ['recorded_endpoint', 'local_endpoint', 'missing_identity'])
def test_dead_guard_with_unknown_identity_is_never_restarted(tmp_path, mismatch):
    h = guard_harness(tmp_path, running=False)
    if mismatch == 'recorded_endpoint':
        (h.guard / 'session_guard_identity.json').write_text(json.dumps({'session': 'ours', 'endpoint': 'other'}))
    elif mismatch == 'local_endpoint':
        h.config.write_text(json.dumps({'ours': {'endpoint': 'other'}}))
    else:
        (h.guard / 'session_guard_identity.json').unlink()
    assert not h.ns['maintain_guard']() and not h.calls


def test_final_guard_wait_has_180_seconds_not_expired_initial_grace(tmp_path):
    h = guard_harness(tmp_path)
    h.ns['guard_wait'] = 900  # The old initial grace expired before finalization.

    def sleep(seconds):
        h.clock['now'] += seconds
        if h.clock['now'] >= 1120:
            h.event('heartbeat', assignment_present=True, socket_connected=True)

    h.ns['time'].sleep = sleep
    h.ns['wait_for_guard']()
    assert h.clock['now'] == 1120 and h.state['guard_healthy']
    assert any(row.get('guard_degraded') for row in h.snapshots)
    assert not h.calls


def test_final_guard_wait_and_takeover_do_not_reset_recovery_budget(tmp_path):
    h = guard_harness(tmp_path, recovery_started=830)
    with pytest.raises(RuntimeError, match='180 seconds'):
        h.ns['wait_for_guard']()
    assert h.clock['now'] == 1010 and not h.calls


def test_guard_recovery_respects_four_hour_work_deadline(tmp_path):
    h = guard_harness(tmp_path)
    h.ns['work_deadline'] = 1030
    with pytest.raises(TimeoutError, match='Four-hour deadline'):
        h.ns['wait_for_guard']()
    assert h.clock['now'] == 1030


def bash_command():
    if os.name == 'nt':
        if not shutil.which('wsl.exe'):
            pytest.skip('WSL bash unavailable')
        return ['wsl.exe', '-e']
    if not shutil.which('bash'):
        pytest.skip('bash unavailable')
    return []


@pytest.mark.parametrize('series,overrides,expected', [
    ('v2', {}, ['config/experiment_v2.json', 'data/v2', 'v2', 'test']),
    ('v2-reload', {}, ['config/experiment_v2_reload.json', 'data/v2', 'v2-reload', 'test']),
    ('v1', {}, ['config/experiment.json', 'data', 'main', 'test']),
    ('v2', {'JEV_CONFIG_FILE': 'config/custom.json', 'JEV_DATA_SUBDIR': 'data/custom', 'JEV_RUN_TAG': 'pilot', 'JEV_SPLIT': 'pilot'},
     ['config/custom.json', 'data/custom', 'pilot', 'pilot']),
    ('v2', {'CONFIG_FILE': 'config/standard.json', 'DATA_SUBDIR': 'data/standard', 'RUN_TAG': 'smoke', 'SPLIT': 'dev'},
     ['config/standard.json', 'data/standard', 'smoke', 'dev']),
    ('v2-reload', {'JEV_CONFIG_FILE': 'config/custom_reload.json', 'JEV_RUN_TAG': 'pilot', 'JEV_SPLIT': 'pilot'},
     ['config/custom_reload.json', 'data/v2', 'pilot', 'pilot'])])
def test_real_bash_series_defaults_without_shadowed_v1_values(series, overrides, expected):
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/launch_seq.sh').read_text()
    fragment = '  local DEFAULT_CONFIG=' + source.split('  local DEFAULT_CONFIG=', 1)[1].split('  [[ "$CONFIG_FILE"', 1)[0]
    script = 'defaults() {\nlocal SERIES=$1; shift\n' + fragment + '\nprintf "%s\\n" "$CONFIG_FILE" "$DATA_SUBDIR" "$RUN_TAG" "$SPLIT"\nprintf "%s\\n" "$@"\n}\ndefaults "$@"\n'
    if series in ('v2', 'v2-reload'):
        canonical = 'canonical_conditions() {' + source.split('canonical_conditions() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        script += canonical + 'ROOT=/fake-root\npython3() { printf "%s\\n" "$2"; }\ncanonical_conditions JFINAL_G4\n'
    command = bash_command() + ['env']
    for name in ('JEV_CONFIG_FILE', 'JEV_DATA_SUBDIR', 'JEV_RUN_TAG', 'JEV_SPLIT', 'CONFIG_FILE', 'DATA_SUBDIR', 'RUN_TAG', 'SPLIT'):
        command += ['-u', name]
    command += ['JEV_SERIES=' + series, *[name + '=' + value for name, value in overrides.items()]]
    result = subprocess.run(command + ['bash', '-s', '--', series], input=script.encode(), capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    rows = result.stdout.decode().splitlines()
    assert rows[:4] == expected
    if series in ('v2', 'v2-reload'):
        assert rows[4:-1] == ['-p', 'CONFIG_FILE', expected[0], '-p', 'DATA_SUBDIR', expected[1],
                            '-p', 'SPLIT', expected[3], '-p', 'RUN_TAG', expected[2]]
        assert rows[-1] == '/fake-root/' + expected[0]


@pytest.mark.parametrize('series', ['v2', 'v2-reload'])
def test_synthetic_offline_preflight_routes_series_without_api_or_old_asset_edits(tmp_path, series):
    real_root = Path(__file__).resolve().parents[1]
    root = tmp_path / 'project'
    scripts = root / 'tools/colab'; scripts.mkdir(parents=True)
    for name in ('launch_seq.sh', '_cexec.sh'):
        (scripts / name).write_text((real_root / 'tools/colab' / name).read_text(), encoding='utf-8', newline='\n')
    for variant in ('v2', 'v2-reload'):
        config_name = 'config/experiment_v2_reload.json' if variant == 'v2-reload' else 'config/experiment_v2.json'
        config = root / config_name; config.parent.mkdir(exist_ok=True)
        config.write_text(json.dumps({'condition_aliases': {'JFINAL_G4': 'JFINAL'}, 'code_version': '0.2.1' if variant == 'v2-reload' else '0.2.0'}))
        notebooks = root / 'notebooks' / variant; notebooks.mkdir(parents=True)
        (notebooks / '06_JFINAL.ipynb').write_text(json.dumps({'synthetic_series': variant}))
        dist = root / 'dist' / variant; dist.mkdir(parents=True)
        with zipfile.ZipFile(dist / 'jev_llm_v2_bundle.zip', 'w') as archive:
            archive.writestr(config_name, config.read_bytes())
            archive.writestr('BUNDLE_MANIFEST.json', json.dumps({'contains_gold': False,
                              'files': {config_name: verify.sha256(config.read_bytes())}}))
    old_paths = [root / 'config/experiment_v2.json', root / 'dist/v2/jev_llm_v2_bundle.zip',
                 root / 'notebooks/v2/06_JFINAL.ipynb']
    old_bytes = {path: path.read_bytes() for path in old_paths}
    script = '''set -euo pipefail
root=$1
if command -v wslpath >/dev/null; then root=$(wslpath -u "$root"); fi
source "$root/tools/colab/launch_seq.sh"
prepare_colab() { echo FORBIDDEN_CLI_SETUP; return 79; }
colab_identity() { echo FORBIDDEN_API; return 79; }
colab() { echo FORBIDDEN_COLAB; return 79; }
launch_sequence synthetic-no-vm A100 JFINAL_G4
'''
    command = bash_command() + ['env']
    for name in ('JEV_CONFIG_FILE', 'JEV_DATA_SUBDIR', 'JEV_RUN_TAG', 'JEV_SPLIT', 'CONFIG_FILE', 'DATA_SUBDIR', 'RUN_TAG', 'SPLIT'):
        command += ['-u', name]
    command += ['JEV_SERIES=' + series, 'JEV_PREFLIGHT_ONLY=1', 'bash', '-s', '--', str(root)]
    result = subprocess.run(command, input=script.encode(), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    text = result.stdout.decode()
    assert 'Offline preflight passed:' in text and 'FORBIDDEN_' not in text
    assert f'/dist/{series}/jev_llm_v2_bundle.zip' in text
    assert ('config=config/experiment_v2_reload.json' if series == 'v2-reload' else 'config=config/experiment_v2.json') in text
    assert 'data=data/v2' in text
    assert all(path.read_bytes() == data for path, data in old_bytes.items())


def test_analysis_source_edits_invalidate_even_inference_bundle_preflight(tmp_path, monkeypatch):
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/launch_seq.sh').read_text()
    preflight = source.split('python3 - "$BUNDLE" "$ROOT" "$CONDS" "$CONFIG_FILE" "$DATA_SUBDIR" <<\'PY\'\n', 1)[1].split('\nPY', 1)[0]
    analysis = tmp_path / 'src/jevlab/analysis.py'; analysis.parent.mkdir(parents=True)
    analysis.write_text('# frozen analysis\n')
    config = tmp_path / 'config/experiment_v2.json'; config.parent.mkdir()
    config.write_text('{}')
    bundle = tmp_path / 'bundle.zip'
    members = {'src/jevlab/analysis.py': analysis.read_bytes(), 'config/experiment_v2.json': config.read_bytes()}
    manifest = {'contains_gold': False, 'files': {name: verify.sha256(data) for name, data in members.items()}}
    with zipfile.ZipFile(bundle, 'w') as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest))
    monkeypatch.setattr(sys, 'argv', ['preflight', str(bundle), str(tmp_path), 'JFINAL', 'config/experiment_v2.json', 'data/v2'])
    exec(compile(preflight, 'bundle-preflight', 'exec'), {})
    analysis.write_text('# post-freeze edit\n')
    with pytest.raises(SystemExit, match='Bundle/file hash mismatch: src/jevlab/analysis.py'):
        exec(compile(preflight, 'bundle-preflight', 'exec'), {})


def test_identity_auth_matches_installed_cli_defaults_in_same_mocked_process():
    source = (Path(__file__).resolve().parents[1] / 'tools/colab/launch_seq.sh').read_text()
    helper = source.split('"$COLAB_PYTHON" - "$COLAB_SESSION_CONFIG" "$1" "${2:-}" <<\'PY\'\n', 1)[1].split('\nPY', 1)[0]
    script = '''import importlib.util, inspect, os, sys
if importlib.util.find_spec('colab_cli') is None:
    sys.exit(77)
from unittest.mock import Mock, patch
from types import SimpleNamespace
from colab_cli import common
from colab_cli.cli import callback
defaults = inspect.signature(callback).parameters
state = common.State()
state.client_oauth_config = 'must-be-overwritten-before-client-access'
state._store = SimpleNamespace(list=lambda: {})
client = SimpleNamespace(list_assignments=Mock(return_value=[]), session=SimpleNamespace(request=Mock()))
original_request = client.session.request
sys.argv = ['identity', '/tmp/fake-isolated-config.json', 'ours', '']
with patch.object(common, 'state', state), patch.object(common, 'get_credentials', return_value=object()) as credentials, patch.object(common, 'Client', return_value=client):
    exec(compile(HELPER, 'identity-auth-probe', 'exec'), {})
    assert credentials.call_args.args[0] == defaults['client_oauth_config'].default == os.path.expanduser('~/.colab-cli-oauth-config.json')
    assert credentials.call_args.kwargs['provider'] == defaults['auth'].default
    assert state.config_path == '/tmp/fake-isolated-config.json'
    original_request.assert_not_called()
print('Default CLI OAuth configuration matches identity; all credentials/APIs mocked')
'''.replace('HELPER', repr(helper))
    python = '/root/.local/share/uv/tools/google-colab-cli/bin/python'
    result = subprocess.run(bash_command() + ['bash', '-c', 'test -x "$1" || exit 77; exec "$1" -', 'bash', python],
                            input=script, text=True, capture_output=True, timeout=30)
    if result.returncode == 77:
        pytest.skip('Installed Colab CLI environment unavailable')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'all credentials/APIs mocked' in result.stdout
