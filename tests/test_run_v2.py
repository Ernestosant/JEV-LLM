"""Offline coordinator sandbox: no real processes, credentials, or GPU API."""

import datetime
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def run_v2():
    path = Path(__file__).resolve().parents[1] / "tools/colab/run_v2.py"
    spec = importlib.util.spec_from_file_location("run_v2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch, run_v2):
    m = run_v2
    c = m.Coordinator(tmp_path)
    clock = {"now": 1000.0}
    running, launches, finished, api_calls = {}, [], [], []
    external = [{"endpoint": "external-t4", "accelerator": "T4"}]
    controls = dict(go=True, bad_go=False, backend_error=False, release=True, fail=None, lag=0,
                    interrupt=False, fail_verify=None, cost_calls=0)
    hashes = {"sealed": "hash"}
    real_frozen = m.frozen_files
    monkeypatch.setattr(m, "frozen_files", lambda root: hashes.copy())
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))

    def status(job, complete=False, initial=False):
        directory = c.out / job["key"]
        result = dict(status="starting" if initial else ("completed" if complete else "running"),
                      mode="cpu" if job["conditions"] == ["ANALYSIS"] else "launch", session=job["session"],
                      conditions=job["conditions"], output=str(directory), cli_config=str(directory / "sessions.json"),
                      cli_wrapper=str(directory / "operator_bin/colab"),
                      start_epoch=clock["now"] - 10, deadline_epoch=clock["now"] + 14400,
                      verified=complete, released=complete and controls["release"], completed_execution=complete)
        if initial:
            result.pop("completed_execution")
        if not initial:
            result.update(operator_pid=100 + list(c.jobs).index(job["key"]), owned_endpoint="own-" + job["key"],
                          expected_count=job["count"], split=job["split"], series=job.get("series", "v2"), config_file=job["config"],
                          data_subdir="data/v2", run_tag=job["tag"],
                          updated_utc=datetime.datetime.fromtimestamp(clock["now"], datetime.timezone.utc).isoformat())
        if job["key"] == controls["fail"] and complete:
            result.update(status="failed", verified=False)
        write_json(directory / "status.json", result)
        write_json(directory / "execution_handover.json", result)
        if not initial:
            write_json(directory / "ownership.json", dict(session=job["session"], endpoint=result["owned_endpoint"]))
            write_json(directory / "sessions.json", {} if complete else {job["session"]: {"endpoint": result["owned_endpoint"]}})
        return result

    def fresh(art, *, conditions, **kwargs):
        return dict(ok=not (controls["fail_verify"] and controls["fail_verify"] in conditions),
                    runs={condition: dict(ok=True, condition=condition, archive=str(art / (condition + "_final.zip")),
                                          archive_sha256=m.digest(art / (condition + "_final.zip")),
                                          notebook_sha256=m.digest(art / (condition + ".out.ipynb")),
                                          config_hash="config", records=kwargs["expected_count"])
                          for condition in conditions})

    def complete(job):
        art = c.out / job["key"] / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        if job["conditions"] == ["ANALYSIS"]:
            with zipfile.ZipFile(art / ("analysis_" + m.TAG + ".zip"), "w") as archive:
                archive.writestr("report.md", "reviewed=false")
                archive.writestr("analysis.json", json.dumps(dict(meta=dict(protocol_version="2", run_tag=m.TAG,
                                  n_gold=100, n_predictions=600, runs=list(m.NOTEBOOKS)[:6]))))
            write_json(art / "07_analisis.out.done.ipynb", {"notebook": True})
        else:
            for condition in job["conditions"]:
                with zipfile.ZipFile(art / (condition + "_final.zip"), "w") as archive:
                    archive.writestr("run/manifest.json", json.dumps(dict(stages={"inference": {"seconds": 5, "ok": True}},
                        config={"code_version": job.get("code_version", "0.2.0"),
                                "params": {"CONFIG_FILE": job["config"], "LATENCY_SWAP_MODE": controls.get("swap_mode", "reload")}})))
                write_json(art / (condition + ".out.ipynb"), {"notebook": True})
            write_json(art.parent / "verification.json", fresh(art, conditions=job["conditions"], expected_count=job["count"]))
        status(job, complete=True)
        finished.append(job["key"])

    def sleep(seconds):
        clock["now"] += seconds
        if controls["interrupt"]:
            controls["interrupt"] = False
            raise InterruptedError("mocked interrupt")
        for key in list(running):
            running[key] -= 1
            if running[key] <= 0:
                running.pop(key)
                complete(c.jobs[key])

    def backend():
        api_calls.append(clock["now"])
        if controls["backend_error"]:
            raise RuntimeError("uncertain")
        rows = list(external)
        for key in running:
            if key not in controls.get("hidden", set()):
                rows.append(dict(endpoint="own-" + key, accelerator="NONE" if key == "analysis" else "A100"))
        if not controls["release"]:
            rows += [dict(endpoint="own-" + key, accelerator="A100") for key in finished]
        return rows

    def process(command, **kwargs):
        if command[0] == "flock":
            return SimpleNamespace(returncode=0)
        if "pilot_decision.py" in command[1]:
            go = controls["go"]
            report = dict(go=go, decision="go" if go else "no-go", errors=[] if go else ["pilot no-go"],
                          conditions={k: {"verified": True, "graded": True} for k in ("JFINAL", "B13", "G_SINGLE")})
            if controls["bad_go"]:
                report["go"] = "true"
            write_json(Path(command[command.index("--output") + 1]), report)
            return SimpleNamespace(returncode=0 if go else 1)
        assert command[0] == "bash" and command[1].endswith("operator.sh")
        job = next(j for j in c.jobs.values() if j["session"] == command[4])
        # Submission intent is on disk before any process start.
        saved = m.read_json(c.path)
        assert saved["submissions"][job["key"]]["status"] == "intent"
        assert len(backend()) < 2
        launches.append((job["key"], list(command), kwargs["env"].copy(), list(finished)))
        status(job)
        running[job["key"]] = 2
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m, "time", SimpleNamespace(time=lambda: clock["now"], sleep=sleep))
    monkeypatch.setattr(c, "backend", backend)
    monkeypatch.setattr(c, "worker", lambda job, st: (st["operator_pid"], "birth") if st and job["key"] in running else None)
    monkeypatch.setattr(m, "verify_run", fresh)
    monkeypatch.setattr(m, "check_notebook", lambda path: None)
    monkeypatch.setattr(m.subprocess, "run", process)
    monkeypatch.setattr(m.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(m.os, "pidfd_open", lambda pid: m.os.open(m.os.devnull, m.os.O_RDONLY), raising=False)
    monkeypatch.setattr(m.signal, "pidfd_send_signal", lambda fd, sig: m.os.kill(100, sig), raising=False)
    hybrid = c.jobs["smoke/hybrid"]
    status(hybrid)
    running[hybrid["key"]] = 2
    status(c.jobs["smoke/single_lat"], initial=True)
    return SimpleNamespace(m=m, c=c, clock=clock, running=running, launches=launches, finished=finished,
                           api_calls=api_calls, external=external, controls=controls, status=status,
                           complete=complete, backend=backend, process=process, hashes=hashes, real_frozen=real_frozen)


def test_full_series_external_t4_sequential_pairs_barriers_and_explicit_parameters(sandbox):
    s = sandbox
    assert s.c.run() == 0
    keys = [item[0] for item in s.launches]
    assert keys == [j["key"] for batch in s.c.batches for j in batch if j["key"] != "smoke/hybrid"]
    for key, command, env, completed in s.launches:
        index = next(i for i, batch in enumerate(s.c.batches) if any(j["key"] == key for j in batch))
        assert all(j["key"] in completed for batch in s.c.batches[:index] for j in batch)
        job = s.c.jobs[key]
        assert env["JEV_SERIES"] == "v2" and env["JEV_SPLIT"] == job["split"] and env["JEV_RUN_TAG"] == job["tag"]
        assert env["JEV_CONFIG_FILE"] == job["config"] and env["JEV_DATA_SUBDIR"] == "data/v2"
        assert command[7] == str(job["count"])
        assert "setsid" not in command and "all" not in command
    smoke = next(item for item in s.launches if item[0] == "smoke/single_lat")
    assert smoke[2]["JEV_LATENCY_ROWS"] == "36" and smoke[2]["JEV_LATENCY_DEV_COUNT"] == "2"
    assert smoke[1][smoke[1].index("LAT_N_DEV") + 1] == "2"
    assert smoke[1][smoke[1].index("LAT_SYNTH_PER_BAND") + 1] == "1"
    diag = next(item for item in s.launches if item[0] == "smoke/diagnostic_08b")
    assert diag[2]["JEV_CONFIG_FILE"] == s.m.DIAGNOSTIC and diag[2]["JEV_RUN_TAG"] == "diagnostic_08b"
    assert diag[1][diag[1].index("KV_CACHE_GB_G") + 1] == "2.0"
    analysis = s.launches[-1]
    assert analysis[1][3] == "cpu" and analysis[2]["JEV_ANALYSIS_EXPECTED_COUNT"] == "100"
    assert analysis[2]["JEV_GOLD_FILE"] == str(s.c.root / "data/v2/test_gold.jsonl")
    assert len(list(Path(analysis[2]["JEV_FINALS_DIR"]).glob("*_final.zip"))) == 7
    assert s.c.state["assignment_observation"] == dict(global_assignments=1, own_active_assignments=0, external_assignments=1)
    assert s.c.state["reviewed"] is False and s.c.state["max_total_assignments"] == 2
    assert (s.c.out / "coordinator_report.md").exists()


def test_takeover_preserves_initial_records_and_existing_hybrid_ownership(sandbox):
    s = sandbox
    path = s.c.out / "smoke/single_lat/status.json"
    original = path.read_text()
    assert s.c.run() == 0
    takeover = s.m.read_json(path.parent / "coordinator_takeover.json")
    assert takeover["originals"]["status.json"]["text"] == original
    assert takeover["originals"]["status.json"]["sha256"] == s.m.hashlib.sha256(original.encode()).hexdigest()
    assert s.c.state["ownership"]["smoke/hybrid"] == "own-smoke/hybrid"
    events = [json.loads(line)["event"] for line in (s.c.out / "coordinator_events.jsonl").read_text().splitlines()]
    assert events.index("takeover_unstarted_single_lat") < events.index("submission_intent")


def test_no_go_stops_without_test_latency_or_cpu_allocation_and_leaves_external(sandbox):
    s = sandbox
    s.controls["go"] = False
    assert s.c.run() == 1
    assert s.c.state["outcome"] == "no-go"
    assert all(key.startswith(("smoke/", "pilot/")) for key, *_ in s.launches)
    assert s.c.state["assignment_observation"]["own_active_assignments"] == 0
    assert s.c.state["assignment_observation"]["global_assignments"] == 1
    assert s.m.read_json(s.c.out / "pilot/decision.json")["go"] is False
    cost = s.m.read_json(s.c.out / "cost_estimate.json")
    assert cost["compute_units"] is None and cost["monetary_cost"] is None
    assert cost["pilot_a100_operator_hours_sum"] > 0
    assert cost["confirmatory_extrapolated_a100_hours"] == cost["pilot_a100_operator_hours_sum"] * 2.5


@pytest.mark.parametrize("failure", ["smoke/hybrid", "smoke/single_lat", "smoke/diagnostic_08b"])
def test_failed_smoke_blocks_pilot_no_retry(sandbox, failure):
    s = sandbox
    s.controls["fail"] = failure
    assert s.c.run() == 1
    assert not any(key.startswith("pilot/") for key, *_ in s.launches)
    assert len([key for key, *_ in s.launches if key == failure]) <= 1
    assert s.c.state["outcome"] == "failed"


def test_unconfirmed_hybrid_release_blocks_all_new_work(sandbox):
    s = sandbox
    s.controls["release"] = False
    assert s.c.run() == 1
    assert not s.launches and s.c.state["outcome"] == "failed"
    assert s.c.state["assignment_observation"]["own_active_assignments"] == 1


@pytest.mark.parametrize("fault", ["capacity", "uncertain"])
def test_capacity_and_uncertainty_block_without_allocation_bounded(sandbox, fault):
    s = sandbox
    s.running.clear()
    s.complete(s.c.jobs["smoke/hybrid"])
    if fault == "capacity":
        s.external.append(dict(endpoint="unknown-other", accelerator="A100"))
    else:
        s.controls["backend_error"] = True
    assert s.c.run() == 1
    assert not s.launches and s.clock["now"] <= 2810
    events = (s.c.out / "coordinator_events.jsonl").read_text()
    assert "backend_blocked" in events or "capacity_blocked" in events
    assert s.external[0]["endpoint"] == "external-t4"


def test_pairs_run_concurrently_only_if_capacity_available(sandbox):
    s = sandbox
    s.external.clear()
    assert s.c.run() == 0
    jfinal = next(item for item in s.launches if item[0] == "pilot/JFINAL")
    b13 = next(item for item in s.launches if item[0] == "pilot/B13")
    assert jfinal[3] == b13[3]
    single = next(item for item in s.launches if item[0] == "pilot/G_SINGLE")
    assert {"pilot/JFINAL", "pilot/B13"} <= set(single[3])


def test_inflight_operator_reserves_slot_before_assignment_visible(sandbox, monkeypatch):
    s = sandbox
    s.controls["hidden"] = {"smoke/single_lat"}
    original = s.process
    def delayed(command, **kwargs):
        result = original(command, **kwargs)
        if command[0] == "bash" and command[4] == s.c.jobs["smoke/single_lat"]["session"]:
            s.running["smoke/single_lat"] = 4
            # Backend has not allocated and ownership is not established yet.
            for name in ("ownership.json", "sessions.json"):
                (s.c.out / "smoke/single_lat" / name).unlink()
            status = s.m.read_json(s.c.out / "smoke/single_lat/status.json")
            status.pop("owned_endpoint")
            write_json(s.c.out / "smoke/single_lat/status.json", status)
        return result
    monkeypatch.setattr(s.m.subprocess, "run", delayed)
    assert s.c.run() == 0
    diag = next(item for item in s.launches if item[0] == "smoke/diagnostic_08b")
    assert "smoke/single_lat" in diag[3]


@pytest.mark.parametrize("proof_failure", ["local", "lock", "expired", "pid", "handover", "ownership"])
def test_initial_takeover_requires_every_proof(sandbox, monkeypatch, proof_failure):
    s = sandbox
    job = s.c.jobs["smoke/single_lat"]
    directory = s.c.out / job["key"]
    if proof_failure == "local":
        write_json(directory / "sessions.json", {job["session"]: {"endpoint": "existing"}})
    elif proof_failure == "lock":
        monkeypatch.setattr(s.c, "no_worker", lambda job: False)
    elif proof_failure == "expired":
        st = s.m.read_json(directory / "status.json")
        st["deadline_epoch"] = 1001
        write_json(directory / "status.json", st)
        write_json(directory / "execution_handover.json", st)
    elif proof_failure == "pid":
        st = s.m.read_json(directory / "status.json")
        st["operator_pid"] = 789
        write_json(directory / "status.json", st)
    elif proof_failure == "handover":
        write_json(directory / "execution_handover.json", {"session": "other"})
    else:
        write_json(directory / "ownership.json", dict(session=job["session"], endpoint="existing"))
    assert s.c.run() == 1
    assert not any(key == job["key"] for key, *_ in s.launches)
    assert not (directory / "coordinator_takeover.json").exists()


def test_dead_hybrid_never_restarts(sandbox):
    s = sandbox
    s.running.clear()
    assert s.c.run() == 1
    assert not s.launches


def test_completed_jobs_resumed_are_freshly_verified_and_not_rerun(sandbox, monkeypatch):
    s = sandbox
    assert s.c.run() == 0
    before = len(s.launches)
    verified = []
    original = s.c.verify
    def record(job, status, rows):
        verified.append(job["key"])
        return original(job, status, rows)
    monkeypatch.setattr(s.c, "verify", record)
    assert s.c.run() == 0
    assert len(s.launches) == before and set(verified) == set(s.c.jobs)


def test_changed_completed_final_prevents_resume_no_rerun(sandbox):
    s = sandbox
    assert s.c.run() == 0
    path = s.c.out / "smoke/hybrid/artifacts/JFINAL_final.zip"
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("changed", "tamper")
    count = len(s.launches)
    assert s.c.run() == 1 and len(s.launches) == count
    assert s.c.state["outcome"] == "failed"


def test_strict_go_boolean_only_no_string_truthiness(sandbox):
    s = sandbox
    s.controls["bad_go"] = True
    assert s.c.run() == 1
    assert not any(key.startswith("confirmatory/") for key, *_ in s.launches)


def test_global_deadline_never_extended_on_resume(sandbox):
    s = sandbox
    s.c.hours = 0.001
    assert s.c.run() == 1
    deadline = s.c.state["deadline_epoch"]
    s.c.state["outcome"] = None
    s.m.atomic_json(s.c.path, s.c.state)
    s.c.hours = 12
    assert s.c.run() == 1
    assert s.c.state["deadline_epoch"] == deadline
    assert not s.launches


@pytest.mark.parametrize("hours", [0, -1, 12.1, float("nan"), float("inf")])
def test_invalid_global_deadline_rejected(run_v2, tmp_path, hours):
    with pytest.raises(ValueError, match="Deadline"):
        run_v2.Coordinator(tmp_path, hours)


def test_ambiguous_submission_never_retries_after_death(sandbox, monkeypatch):
    s = sandbox
    original = s.process
    def lost(command, **kwargs):
        if command[0] == "bash":
            raise OSError("start failed")
        return original(command, **kwargs)
    monkeypatch.setattr(s.m.subprocess, "run", lost)
    assert s.c.run() == 1
    assert s.c.state["submissions"]["smoke/single_lat"]["status"] == "intent"
    s.c.state["outcome"] = None
    s.m.atomic_json(s.c.path, s.c.state)
    s.clock["now"] += 121
    monkeypatch.setattr(s.m.subprocess, "run", original)
    assert s.c.run() == 1
    assert not s.launches


def test_frozen_mismatch_before_run_makes_no_calls_or_writes(sandbox, monkeypatch):
    s = sandbox
    def invalid(root):
        raise ValueError("Bundle/file hash mismatch")
    monkeypatch.setattr(s.m, "frozen_files", invalid)
    with pytest.raises(ValueError, match="hash mismatch"):
        s.c.run()
    assert not s.api_calls and not s.launches and not s.c.path.exists()


def test_changed_frozen_files_before_launch_blocks_allocation(sandbox):
    s = sandbox
    original = s.backend
    def changed():
        rows = original()
        if "smoke/hybrid" in s.finished:
            s.hashes["sealed"] = "changed"
        return rows
    s.c.backend = changed
    assert s.c.run() == 1
    assert not s.launches


def test_foreign_state_rejected_without_signalling_or_backend(sandbox, monkeypatch):
    s = sandbox
    write_json(s.c.path, {"version": 1, "root": "/foreign", "observed_jobs": ["unknown"]})
    killed = []
    monkeypatch.setattr(s.m.os, "kill", lambda *a: killed.append(a))
    with pytest.raises(ValueError, match="state/plan mismatch"):
        s.c.run()
    assert not killed and not s.api_calls


def test_collect_exactly_seven_finals_excludes_checkpoint_and_duplicate(sandbox):
    s = sandbox
    assert s.c.run() == 0
    collected = s.c.out / "collected_finals"
    (collected / "junk_checkpoint.zip").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="Unexpected collected"):
        s.c.collect()


def test_partial_collection_staging_resumes_without_duplicates(sandbox):
    s = sandbox
    assert s.c.run() == 0
    directory = s.c.out / "collected_finals"
    path = next(directory.glob("*_final.zip"))
    path.unlink()
    path.with_suffix(".tmp").write_bytes(b"partial")
    assert s.c.collect() == directory
    assert len(list(directory.glob("*_final.zip"))) == 7 and not list(directory.glob("*.tmp"))


def test_interrupt_signals_only_identified_workers_requests_independent_cleanup(sandbox, monkeypatch):
    s = sandbox
    killed = []
    def kill(pid, sig):
        killed.append((pid, sig))
        for key in list(s.running):
            if s.status(s.c.jobs[key])["operator_pid"] == pid:
                s.running.pop(key)
                s.complete(s.c.jobs[key])
    monkeypatch.setattr(s.m.os, "kill", kill)
    s.controls["interrupt"] = True
    assert s.c.run() == 1
    assert killed == [(100, signal.SIGTERM)]
    assert s.c.state["outcome"] == "interrupted"
    assert not s.launches and s.c.state["cleanup_unconfirmed"] == []


def test_cleanup_refuses_changed_endpoint_and_reused_pid(sandbox, monkeypatch):
    s = sandbox
    s.c.state = dict(observed_jobs=["smoke/hybrid"], ownership={})
    job = s.c.jobs["smoke/hybrid"]
    write_json(s.c.out / job["key"] / "sessions.json", {job["session"]: {"endpoint": "replacement"}})
    killed = []
    monkeypatch.setattr(s.m.os, "kill", lambda *a: killed.append(a))
    s.c.cleanup()
    assert not killed
    write_json(s.c.out / job["key"] / "sessions.json", {job["session"]: {"endpoint": "own-smoke/hybrid"}})
    identities = iter([(100, "first"), (100, "reused")])
    monkeypatch.setattr(s.c, "worker", lambda *a: next(identities))
    s.c.cleanup()
    assert not killed


def test_publish_no_go_requires_own_absence_not_zero_global(sandbox):
    s = sandbox
    s.c.state = dict(ownership={"smoke/hybrid": "own-smoke/hybrid"}, observed_jobs=[])
    assert s.c.publish("no-go") == 1
    assert s.c.state["outcome"] == "blocked_release_unconfirmed"
    assert s.c.state["assignment_observation"]["global_assignments"] == 2


def test_backend_subprocess_bounded_and_malformed_output_secret_suppressed(run_v2, tmp_path, monkeypatch):
    c = run_v2.Coordinator(tmp_path)
    calls = []
    def process(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="secret malformed credential", stderr="secret")
    monkeypatch.setattr(run_v2.subprocess, "run", process)
    with pytest.raises(ValueError) as error:
        c.backend()
    assert "secret" not in str(error.value)
    assert calls[0][1]["timeout"] == 60 and calls[0][1]["capture_output"]
    assert calls[0][0][-1] == str(c.out / "coordinator_sessions.json")


def test_backend_code_uses_cli_auth_and_bounded_socket_requests_with_mock_modules(run_v2, monkeypatch, capsys):
    class State:
        @property
        def client(self):
            assert self.config_path == "/isolated/coordinator.json"
            assert self.client_oauth_config.endswith(".colab-cli-oauth-config.json")
            return client
    calls = []
    session = SimpleNamespace(request=lambda *args, **kw: calls.append(kw))
    def assignments():
        session.request("GET", "fake", timeout=None)
        return [SimpleNamespace(endpoint="external", accelerator=SimpleNamespace(value="T4"))]
    client = SimpleNamespace(session=session, list_assignments=assignments)
    monkeypatch.setitem(sys.modules, "colab_cli", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "colab_cli.common", SimpleNamespace(state=State()))
    monkeypatch.setattr(sys, "argv", ["backend", "/isolated/coordinator.json"])
    import logging
    previous = logging.root.manager.disable
    try:
        exec(run_v2.BACKEND_CODE, {})
    finally:
        logging.disable(previous)
    assert calls == [{"timeout": (5, 15)}]
    assert json.loads(capsys.readouterr().out) == [{"endpoint": "external", "accelerator": "T4"}]


@pytest.mark.parametrize("mismatch", [None, "session", "output", "env", "zombie", "absent"])
def test_real_worker_identity_without_real_process(run_v2, tmp_path, monkeypatch, mismatch):
    c = run_v2.Coordinator(tmp_path)
    job = c.jobs["smoke/hybrid"]
    directory = c.out / job["key"]
    st = dict(operator_pid=789, mode="launch")
    args = ["python3", "-", "launch", job["session"], " ".join(job["conditions"]), str(directory), "1",
            str(tmp_path / "data/v2/dev_inputs.jsonl")]
    env = dict(JEV_ROOT=str(tmp_path), JEV_SERIES="v2", JEV_OPERATOR_OUT=str(directory),
               COLAB_SESSION_CONFIG=str(directory / "sessions.json"))
    if mismatch == "session":
        args[3] = "foreign"
    elif mismatch == "output":
        args[5] = "/foreign"
    elif mismatch == "env":
        env["JEV_ROOT"] = "/foreign"
    state = "Z" if mismatch == "zombie" else "S"
    virtual = {"/proc/789/stat": "789 (python3) " + " ".join([state, *(["0"] * 18), "birth", "0"]),
               "/proc/789/cmdline": "\0".join(args) + "\0",
               "/proc/789/environ": "\0".join(k + "=" + v for k, v in env.items()) + "\0"}
    original_text, original_bytes = Path.read_text, Path.read_bytes
    def text(path, *a, **kw):
        key = path.as_posix()
        if key in virtual:
            if mismatch == "absent":
                raise FileNotFoundError()
            return virtual[key]
        return original_text(path, *a, **kw)
    def raw(path):
        return virtual[path.as_posix()].encode() if path.as_posix() in virtual else original_bytes(path)
    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "read_bytes", raw)
    if mismatch in ("session", "output", "env"):
        with pytest.raises(ValueError, match="PID identity mismatch"):
            c.worker(job, st)
    else:
        assert c.worker(job, st) == (None if mismatch else (789, "birth"))


@pytest.fixture
def frozen_sandbox(tmp_path, run_v2):
    m = run_v2
    root = tmp_path
    common = {m.CONFIG: b'{"runtime_defaults": {}}', m.DIAGNOSTIC: b'{}', "src/jevlab/core.py": b"# frozen\n",
              "src/jevlab/__init__.py": b'__version__ = "0.2.0"\n',
              "data/v2/schedule.json": b"{}"}
    common.update({"data/v2/" + s + "_inputs.jsonl": b'{}\n' for s in ("dev", "pilot", "test")})
    common.update({"prompts/" + n + ".txt": b"prompt" for n in ("generador", "criterio_paso", "criterio_final")})
    gold = {"data/v2/" + s + "_gold.jsonl": b'{}\n' for s in ("dev", "pilot", "test")}
    for name, value in {**common, **gold}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    seals = "\n".join(m.hashlib.sha256(value).hexdigest() + "  " + Path(name).name
                      for name, value in {**common, **gold}.items() if "data/v2/" in name)
    (root / "data/v2/SHA256SUMS").write_text(seals)
    for yes_gold, name in ((False, "jev_llm_v2_bundle.zip"), (True, "jev_llm_v2_analysis_bundle.zip")):
        path = root / "dist/v2" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        members = {**common, **(gold if yes_gold else {})}
        with zipfile.ZipFile(path, "w") as archive:
            for n, value in members.items():
                archive.writestr(n, value)
            archive.writestr("BUNDLE_MANIFEST.json", json.dumps(dict(series="v2", contains_gold=yes_gold,
                              files={n: m.hashlib.sha256(value).hexdigest() for n, value in members.items()})))
    for name in ("operator.sh", "launch_seq.sh", "session_guard.sh", "session_guard.py",
                 "stop.sh", "pull.sh", "status.sh", "_cexec.sh", "run_v2.py"):
        path = root / "tools/colab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# script")
    for name in ("verify_run.py", "pilot_decision.py"):
        (root / "tools" / name).write_text("# script")
    for name in m.NOTEBOOKS.values():
        write_json(root / "notebooks/v2" / (name + ".ipynb"), {"cells": [dict(metadata={"tags": ["parameters"]},
                   source=['ROOT = "/content/jev_llm_v2"\n', 'CONFIG_FILE = "config/experiment_v2.json"\n',
                           'DATA_SUBDIR = "data/v2"\n'])]})
    return root


def test_both_real_frozen_bundle_verification_and_post_freeze_source_rejection(run_v2, frozen_sandbox):
    root = frozen_sandbox
    assert "config/diagnostic_08b_a100.json" in run_v2.frozen_files(root)
    (root / "src/jevlab/core.py").write_text("# modified after freeze")
    with pytest.raises(ValueError, match="Bundle/file hash mismatch"):
        run_v2.frozen_files(root)


@pytest.mark.parametrize("fault", ["gold", "config", "extra", "path", "seal", "notebook"])
def test_frozen_bundle_safety_sentinels_fail_closed(run_v2, frozen_sandbox, fault):
    root = frozen_sandbox
    bundle = root / "dist/v2/jev_llm_v2_bundle.zip"
    if fault in ("gold", "extra", "path"):
        name = {"gold": "data/v2/test_gold.jsonl", "extra": "unsealed", "path": "../outside"}[fault]
        with zipfile.ZipFile(bundle, "a") as archive:
            archive.writestr(name, "unsafe")
    elif fault == "config":
        (root / run_v2.DIAGNOSTIC).write_text("changed")
    elif fault == "seal":
        (root / "data/v2/pilot_gold.jsonl").write_text("changed")
    else:
        path = root / "notebooks/v2/01_G_SINGLE.ipynb"
        path.write_text(path.read_text().replace("/content/jev_llm_v2", "/wrong/root"))
    with pytest.raises(ValueError):
        run_v2.frozen_files(root)


def test_plan_is_offline_and_no_execution_without_explicit_run(run_v2, frozen_sandbox, monkeypatch, capsys):
    monkeypatch.setattr(run_v2, "ROOT", frozen_sandbox)
    def forbidden(*a, **kw):
        pytest.fail("Plan must not spawn a process or contact backend")
    monkeypatch.setattr(run_v2.subprocess, "run", forbidden)
    assert run_v2.main(["--plan"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["batches"]) == 9 and report["max_total_assignments"] == 2
    assert not (frozen_sandbox / "results").exists()
    with pytest.raises(SystemExit):
        run_v2.main([])


def test_existing_failed_second_member_blocks_unsubmitted_first_member(sandbox):
    s = sandbox
    s.controls["fail"] = "smoke/diagnostic_08b"
    s.complete(s.c.jobs["smoke/diagnostic_08b"])
    assert s.c.run() == 1
    assert not s.launches
    assert not (s.c.out / "smoke/single_lat/coordinator_takeover.json").exists()


@pytest.mark.parametrize("phase", ["failed", "completed", "preserving_and_releasing", "verified_pending_release"])
def test_cleanup_never_interrupts_operator_release_phase(sandbox, monkeypatch, phase):
    s = sandbox
    job = s.c.jobs["smoke/hybrid"]
    status = s.m.read_json(s.c.out / job["key"] / "status.json")
    status["status"] = phase
    write_json(s.c.out / job["key"] / "status.json", status)
    s.c.state = dict(observed_jobs=[job["key"]], ownership={})
    killed = []
    monkeypatch.setattr(s.m.os, "kill", lambda *a: killed.append(a))
    s.c.cleanup()
    assert not killed and s.c.state["cleanup_unconfirmed"] == []


def test_cleanup_signal_intent_is_idempotent(sandbox, monkeypatch):
    s = sandbox
    s.c.state = dict(observed_jobs=["smoke/hybrid"], ownership={},
                     cleanup_requests={"smoke/hybrid": {"pid": 100, "starttime": "birth", "epoch": 999}})
    killed = []
    monkeypatch.setattr(s.m.os, "kill", lambda *a: killed.append(a))
    s.c.cleanup()
    assert not killed and s.c.state["cleanup_unconfirmed"] == []


def test_deadline_crossed_during_prelaunch_verification_never_submits(sandbox, monkeypatch):
    s = sandbox
    def slow(root):
        if "smoke/hybrid" in s.finished:
            s.clock["now"] = s.c.state["deadline_epoch"] + 1
        return s.hashes.copy()
    monkeypatch.setattr(s.m, "frozen_files", slow)
    assert s.c.run() == 1
    assert not s.launches and not s.c.state["submissions"]


def test_operator_deadline_is_capped_by_remaining_global_budget(sandbox):
    s = sandbox
    s.c.hours = 1
    assert s.c.run() == 0
    assert all(0 < int(env["JEV_DEADLINE_SECONDS"]) < 3600 for _, _, env, _ in s.launches)


@pytest.mark.parametrize("evidence", ["operator.log", "checkpoint_staging", "guard", "extra_record_field"])
def test_takeover_rejects_any_noninitial_evidence(sandbox, evidence):
    s = sandbox
    job = s.c.jobs["smoke/single_lat"]
    directory = s.c.out / job["key"]
    if evidence == "guard":
        (s.c.root / "results/lifecycle" / job["session"]).mkdir(parents=True)
    elif evidence == "extra_record_field":
        st = s.m.read_json(directory / "status.json")
        st["health"] = {"known": True}
        write_json(directory / "status.json", st)
        write_json(directory / "execution_handover.json", st)
    else:
        (directory / evidence).touch()
    assert s.c.run() == 1
    assert not s.launches and not (directory / "coordinator_takeover.json").exists()


def test_duplicate_extra_final_blocks_fresh_verification(sandbox):
    s = sandbox
    s.running.clear()
    s.complete(s.c.jobs["smoke/hybrid"])
    (s.c.out / "smoke/hybrid/artifacts/duplicate_final.zip").write_bytes(b"duplicate")
    assert s.c.run() == 1
    assert not s.launches


def test_fresh_verifier_failure_blocks_all_followup(sandbox):
    s = sandbox
    s.controls["fail_verify"] = "JFINAL"
    assert s.c.run() == 1
    assert not s.launches


def test_guard_or_coordinator_edit_changes_frozen_snapshot(run_v2, frozen_sandbox):
    before = run_v2.frozen_files(frozen_sandbox)
    (frozen_sandbox / "tools/colab/session_guard.py").write_text("# changed guard")
    (frozen_sandbox / "tools/colab/run_v2.py").write_text("# changed coordinator")
    after = run_v2.frozen_files(frozen_sandbox)
    assert after["tools/colab/session_guard.py"] != before["tools/colab/session_guard.py"]
    assert after["tools/colab/run_v2.py"] != before["tools/colab/run_v2.py"]


def test_invalid_plan_bundle_never_starts_backend_or_process(run_v2, frozen_sandbox, monkeypatch):
    monkeypatch.setattr(run_v2, "ROOT", frozen_sandbox)
    (frozen_sandbox / run_v2.DIAGNOSTIC).write_text("post-freeze edit")
    monkeypatch.setattr(run_v2.subprocess, "run", lambda *a, **kw: pytest.fail("No process on invalid plan"))
    with pytest.raises(ValueError, match="hash mismatch"):
        run_v2.main(["--plan"])
    assert not (frozen_sandbox / "results").exists()


def test_inplace_ownership_publication_retries_bounded_without_cleanup(sandbox, monkeypatch):
    s = sandbox
    job = s.c.jobs["smoke/hybrid"]
    path = s.c.out / job["key"] / "ownership.json"
    original = s.m.read_json
    attempts = []
    def partial(p):
        if Path(p) == path:
            attempts.append(1)
            if len(attempts) < 3:
                raise json.JSONDecodeError("partial write", "", 0)
        return original(p)
    monkeypatch.setattr(s.m, "read_json", partial)
    assert s.c.run() == 0
    assert len(attempts) >= 3 and s.c.state["outcome"] == "completed"


def test_persistent_invalid_ownership_is_bounded_and_fail_closed(sandbox):
    s = sandbox
    s.running.clear()
    job = s.c.jobs["smoke/hybrid"]
    (s.c.out / job["key"] / "ownership.json").write_text("partial forever")
    with pytest.raises(ValueError, match="publication remains uncertain"):
        s.c.endpoint(job, s.c.status(job))
    assert 1002 <= s.clock["now"] < 1003 and not s.launches


def test_takeover_snapshot_before_submission_crash_can_resume_without_overwrite(sandbox, monkeypatch):
    s = sandbox
    original = s.m.frozen_files
    crashed = []
    def crash_after_takeover(root):
        path = s.c.out / "smoke/single_lat/coordinator_takeover.json"
        if path.exists() and not crashed:
            crashed.append(True)
            raise SystemExit("simulate hard death before submission intent")
        return original(root)
    monkeypatch.setattr(s.m, "frozen_files", crash_after_takeover)
    assert s.c.run() == 1
    path = s.c.out / "smoke/single_lat/coordinator_takeover.json"
    preserved = path.read_bytes()
    assert not s.launches and not s.c.state["submissions"]
    # A hard death leaves outcome unset, unlike the caught simulation above.
    s.c.state["outcome"] = None
    s.m.atomic_json(s.c.path, s.c.state)
    monkeypatch.setattr(s.m, "frozen_files", original)
    assert s.c.run() == 0
    assert path.read_bytes() == preserved
    assert "takeover_proof_revalidated" in (s.c.out / "coordinator_events.jsonl").read_text()


def test_takeover_resume_rejects_changed_snapshot(sandbox, monkeypatch):
    s = sandbox
    st = s.c.status(s.c.jobs["smoke/single_lat"])
    write_json(s.c.out / "smoke/single_lat/coordinator_takeover.json", {"originals": {"changed": True}})
    assert s.c.run() == 1
    assert not s.launches


def test_backend_process_budget_cannot_overrun_global_deadline(run_v2, tmp_path, monkeypatch):
    c = run_v2.Coordinator(tmp_path)
    c.state = {"deadline_epoch": 1030}
    clock = {"now": 1000}
    monkeypatch.setattr(run_v2, "time", SimpleNamespace(time=lambda: clock["now"]))
    budgets = []
    def process(*a, **kw):
        budgets.append(kw["timeout"])
        return SimpleNamespace(returncode=0, stdout="[]")
    monkeypatch.setattr(run_v2.subprocess, "run", process)
    assert c.backend() == [] and budgets == [30]
    clock["now"] = 1030
    with pytest.raises(ValueError, match="deadline expired"):
        c.backend()
    assert budgets == [30]


def test_global_budget_reserves_cleanup_before_work_deadline(sandbox):
    s = sandbox
    assert s.c.run() == 0
    assert s.c.state["work_deadline_epoch"] == s.c.state["deadline_epoch"] - 480


def test_missing_hybrid_records_are_never_permission_to_start_again(sandbox):
    s = sandbox
    s.running.clear()
    s.m.shutil.rmtree(s.c.out / "smoke/hybrid")
    assert s.c.run() == 1
    assert not s.launches and not s.c.state["submissions"]


def make_reload_files(root, m):
    """Build a miniature amended freeze, leaving both old ZIPs/notebooks intact."""
    config = m.read_json(root / m.CONFIG)
    config.update(code_version="0.2.1", amendment={"approved_by": "user"})
    config["runtime_defaults"]["LATENCY_SWAP_MODE"] = "reload"
    write_json(root / m.RELOAD_CONFIG, config)
    (root / "src/jevlab/__init__.py").write_text('__version__ = "0.2.1"\n')
    for gold, name in ((False, "jev_llm_v2_bundle.zip"), (True, "jev_llm_v2_analysis_bundle.zip")):
        with zipfile.ZipFile(root / "dist/v2" / name) as original:
            old = json.loads(original.read("BUNDLE_MANIFEST.json"))
            members = {n: original.read(n) for n in old["files"]}
        members["src/jevlab/__init__.py"] = (root / "src/jevlab/__init__.py").read_bytes()
        members[m.RELOAD_CONFIG] = (root / m.RELOAD_CONFIG).read_bytes()
        path = root / "dist/v2-reload" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            for n, value in members.items():
                archive.writestr(n, value)
            archive.writestr("BUNDLE_MANIFEST.json", json.dumps(dict(series="v2-reload", code_version="0.2.1",
                              contains_gold=gold, files={n: m.hashlib.sha256(v).hexdigest() for n, v in members.items()})))
    for name in m.NOTEBOOKS.values():
        nb = m.read_json(root / "notebooks/v2" / (name + ".ipynb"))
        nb["cells"][0]["source"] = [s.replace(m.CONFIG, m.RELOAD_CONFIG) for s in nb["cells"][0]["source"]]
        write_json(root / "notebooks/v2-reload" / (name + ".ipynb"), nb)


@pytest.fixture
def reload_sandbox(sandbox, frozen_sandbox, monkeypatch):
    s = sandbox
    m, c = s.m, s.c
    s.running.clear()
    old_hashes = s.real_frozen(c.root)
    for key in ("smoke/hybrid", "smoke/diagnostic_08b", "smoke/single_lat"):
        s.complete(c.jobs[key])
    combo = c.out / "smoke/single_lat"
    st = m.read_json(combo / "status.json")
    st.update(status="failed", verified=False, completed_execution=False,
              health={"known": True, "alive": False, "sequence_status": {
                  "G_SINGLE": {"rc": 0}, "LATENCY": {"rc": 1}, "_done": True}})
    write_json(combo / "status.json", st)
    write_json(combo / "execution_handover.json", st)
    (combo / "artifacts/LATENCY_final.zip").unlink()
    (combo / "verification.json").unlink()
    (combo / "artifacts/papermill_LATENCY.unit.log").write_text("torch.OutOfMemoryError: CUDA out of memory")
    fresh = m.verify_run(combo / "artifacts", conditions=["G_SINGLE"], expected_count=1)
    write_json(combo / "gsingle_verification.json", fresh)
    previous = dict(version=1, outcome="failed", plan=m.plan(), frozen_files=old_hashes, max_total_assignments=2,
                    ownership={k: "own-" + k for k in ("smoke/hybrid", "smoke/diagnostic_08b", "smoke/single_lat")},
                    observed_jobs=["smoke/hybrid", "smoke/diagnostic_08b", "smoke/single_lat"], submissions={},
                    verified_jobs={key: m.read_json(c.out / key / "verification.json") for key in ("smoke/hybrid", "smoke/diagnostic_08b")})
    write_json(c.path, previous)
    c.events.write_text('{"event": "old_failed"}\n')
    c.summary.write_text("# Original coordinator failed; do not rewrite\n")
    preserved = {p.relative_to(c.root).as_posix(): m.digest(p) for p in [c.path, c.events, c.summary,
                 *[p for p in (c.out / "smoke").rglob("*") if p.is_file()]]}
    make_reload_files(c.root, m)
    amended = m.Coordinator(c.root, amend_latency_reload=True)
    # Reuse the sandbox process closures, which reference the original object.
    c.__dict__.update(amended.__dict__)
    monkeypatch.setattr(c, "backend", s.backend)
    monkeypatch.setattr(c, "worker", lambda job, st: (st["operator_pid"], "birth") if st and job["key"] in s.running else None)
    monkeypatch.setattr(m, "frozen_files", s.real_frozen)
    s.preserved = preserved
    return s


def test_reload_amendment_partial_g_full_plan_and_original_records_unchanged(reload_sandbox):
    s = reload_sandbox
    assert s.c.run() == 0
    keys = [key for key, *_ in s.launches]
    assert keys == [job["key"] for batch in s.c.batches for job in batch]
    assert keys[:4] == ["smoke/latency_reload", "pilot_reload/JFINAL", "pilot_reload/B13", "pilot_reload/G_SINGLE"]
    assert keys[4:10] == ["confirmatory_reload/" + c for c in ("B13", "B13_GREEDY", "G_SINGLE", "J64", "JSTEP", "JFINAL")]
    assert all(env["JEV_SERIES"] == "v2-reload" and env["JEV_CONFIG_FILE"] == s.m.RELOAD_CONFIG for _, _, env, _ in s.launches)
    assert s.launches[0][2]["JEV_LATENCY_ROWS"] == "36" and s.launches[0][2]["JEV_LATENCY_DEV_COUNT"] == "2"
    assert s.launches[-1][2]["JEV_ANALYSIS_EXPECTED_COUNT"] == "100"
    assert len(list((s.c.out / "collected_finals_reload").glob("*_final.zip"))) == 7
    assert s.c.path.name == "coordinator_reload_state.json" and s.c.events.name == "coordinator_reload_events.jsonl"
    assert s.c.summary.name == "summary_reload.md" and (s.c.out / "pilot_reload/decision.json").is_file()
    assert not {"smoke/hybrid", "smoke/single_lat", "smoke/diagnostic_08b"} & set(s.c.state["verified_jobs"])
    partial = s.c.state["legacy_evidence"]["evidence"]["smoke/single_lat"]
    assert partial["operator_status"] == "failed" and set(partial["verification"]["runs"]) == {"G_SINGLE"}
    assert "ONLY" in partial["scope"]
    assert s.preserved == {n: s.m.digest(s.c.root / n) for n in s.preserved}


def test_reload_plan_offline_accepts_partial_g_and_old_code_divergence(reload_sandbox, monkeypatch, capsys):
    s = reload_sandbox
    monkeypatch.setattr(s.m, "ROOT", s.c.root)
    monkeypatch.setattr(s.m.subprocess, "run", lambda *a, **kw: pytest.fail("No process/backend for --plan"))
    assert s.m.main(["--plan", "--amend-latency-reload"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["batches"][0][0]["conditions"] == ["LATENCY"]
    assert printed["legacy_evidence"]["old_bundles_sha256"]
    assert not s.c.path.exists() and not s.api_calls
    with pytest.raises(ValueError, match="hash mismatch"):
        s.real_frozen(s.c.root)


@pytest.mark.parametrize("missing", ["gsingle_verification.json", "ownership.json", "artifacts/G_SINGLE_final.zip",
                                     "artifacts/G_SINGLE.out.ipynb", "artifacts/papermill_LATENCY.unit.log"])
def test_reload_missing_partial_g_proof_blocks_without_allocation(reload_sandbox, missing):
    s = reload_sandbox
    (s.c.out / "smoke/single_lat" / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        s.c.run()
    assert not s.launches and not s.api_calls and not s.c.path.exists()


@pytest.mark.parametrize("fault", ["g_rc", "lat_rc", "not_done", "no_oom", "unreleased", "whole_success", "changed_hash", "old_bundle"])
def test_reload_only_known_released_latency_oom_exception_is_allowed(reload_sandbox, fault):
    s = reload_sandbox
    directory = s.c.out / "smoke/single_lat"
    status = s.m.read_json(directory / "status.json")
    if fault == "g_rc":
        status["health"]["sequence_status"]["G_SINGLE"]["rc"] = 1
    elif fault == "lat_rc":
        status["health"]["sequence_status"]["LATENCY"]["rc"] = 0
    elif fault == "not_done":
        status["health"]["sequence_status"]["_done"] = False
    elif fault == "unreleased":
        status["released"] = False
    elif fault == "whole_success":
        status.update(status="completed", verified=True, completed_execution=True)
    elif fault == "no_oom":
        (directory / "artifacts/papermill_LATENCY.unit.log").write_text("other failed stage, not known OOM")
    elif fault == "changed_hash":
        evidence = s.m.read_json(directory / "gsingle_verification.json")
        evidence["runs"]["G_SINGLE"]["notebook_sha256"] = "tampered"
        write_json(directory / "gsingle_verification.json", evidence)
    else:
        with zipfile.ZipFile(s.c.root / "dist/v2/jev_llm_v2_bundle.zip", "a") as archive:
            archive.writestr("changed", "old bundle must stay immutable")
    write_json(directory / "status.json", status)
    write_json(directory / "execution_handover.json", status)
    with pytest.raises(ValueError):
        s.c.run()
    assert not s.launches and not s.api_calls


def test_reload_requires_backend_absence_of_old_combo_endpoint(reload_sandbox):
    s = reload_sandbox
    s.external.append(dict(endpoint="own-smoke/single_lat", accelerator="A100"))
    assert s.c.run() == 1
    assert not s.launches
    assert "Legacy endpoint still present" in s.c.state["error_message"]
    assert s.preserved == {n: s.m.digest(s.c.root / n) for n in s.preserved}


@pytest.mark.parametrize("kind", ["old_hybrid_failed", "old_extra_failed", "old_pilot_started", "new_failed"])
def test_reload_existing_other_failed_or_attempted_work_never_retries(reload_sandbox, kind):
    s = reload_sandbox
    if kind == "old_hybrid_failed":
        path = s.c.out / "smoke/hybrid/status.json"
        status = s.m.read_json(path)
        status.update(status="failed", verified=False)
        write_json(path, status)
        write_json(path.parent / "execution_handover.json", status)
    elif kind == "old_extra_failed":
        write_json(s.c.out / "smoke/another/status.json", {"status": "failed"})
    elif kind == "old_pilot_started":
        write_json(s.c.out / "pilot/JFINAL/status.json", {"status": "failed"})
    else:
        job = s.c.jobs["pilot_reload/B13"]
        status = s.status(job)
        status.update(status="failed", released=True, verified=False)
        write_json(s.c.out / job["key"] / "status.json", status)
    with pytest.raises(ValueError):
        s.c.run()
    assert not s.launches and not s.api_calls


def test_reload_smoke_failure_stops_before_pilot_without_retry(reload_sandbox):
    s = reload_sandbox
    s.controls["fail"] = "smoke/latency_reload"
    assert s.c.run() == 1
    assert [key for key, *_ in s.launches] == ["smoke/latency_reload"]
    with pytest.raises(ValueError, match="no retry"):
        s.c.run()
    assert len(s.launches) == 1


def test_reload_no_go_never_allocates_confirmatory_latency_or_cpu(reload_sandbox):
    s = reload_sandbox
    s.controls["go"] = False
    assert s.c.run() == 1 and s.c.state["outcome"] == "no-go"
    assert [key for key, *_ in s.launches] == ["smoke/latency_reload", "pilot_reload/JFINAL", "pilot_reload/B13", "pilot_reload/G_SINGLE"]
    assert s.preserved == {n: s.m.digest(s.c.root / n) for n in s.preserved}


def test_reload_completed_jobs_resume_without_rerunning_quality(reload_sandbox):
    s = reload_sandbox
    assert s.c.run() == 0
    count = len(s.launches)
    assert s.c.run() == 0 and len(s.launches) == count
    assert s.preserved == {n: s.m.digest(s.c.root / n) for n in s.preserved}


def test_reload_fresh_old_verification_uses_correct_configs_versions_and_scope(reload_sandbox, monkeypatch):
    s = reload_sandbox
    calls = []
    original = s.m.verify_run
    def record(art, **kwargs):
        calls.append((art.parent.name, kwargs))
        return original(art, **kwargs)
    monkeypatch.setattr(s.m, "verify_run", record)
    evidence = s.c.legacy()
    assert calls[0][1]["config"] == s.c.root / s.m.CONFIG
    assert calls[1][1]["config"] == s.c.root / s.m.DIAGNOSTIC
    assert calls[2][1]["conditions"] == ["G_SINGLE"] and calls[2][1]["config"] == s.c.root / s.m.CONFIG
    assert all(kw["code_version"] == "0.2.0" and kw["expected_count"] == 1 and kw["split"] == "dev" for _, kw in calls)
    assert evidence["evidence"]["smoke/single_lat"]["operator_status"] == "failed"


def test_reload_executed_latency_must_bind_reload_not_sleep(reload_sandbox):
    s = reload_sandbox
    s.controls["swap_mode"] = "sleep"
    assert s.c.run() == 1
    assert [key for key, *_ in s.launches] == ["smoke/latency_reload"]
    assert "approved reload mode" in s.c.state["error_message"]


def test_sanitized_diagnostics_include_local_reason_not_raw_api_or_credentials(run_v2):
    safe = run_v2.diagnosis(run_v2.CoordinatorError("Missing stage proof; token=private-secret https://api.example/?password=abc Bearer token-value"))
    assert "Missing stage proof" in safe["error_message"]
    assert not any(secret in safe["error_message"] for secret in ("private-secret", "api.example", "abc", "token-value"))
    unsafe = run_v2.diagnosis(RuntimeError("API raw payload: password=very-secret"))
    assert "very-secret" not in str(unsafe) and "API raw" not in str(unsafe)


def test_reload_capacity_appearing_during_reaudit_blocks_before_submission(reload_sandbox, monkeypatch):
    s = reload_sandbox
    original = s.c.legacy
    injected = []
    def appeared(assignments=None):
        result = original(assignments)
        if assignments is not None and not injected:
            injected.append(True)
            s.external.append(dict(endpoint="new-external", accelerator="A100"))
        return result
    monkeypatch.setattr(s.c, "legacy", appeared)
    assert s.c.run() == 1
    assert not s.launches and not s.c.state["submissions"]
    assert "capacity_blocked_before_submission" in s.c.events.read_text()


def test_reload_fresh_cap_reserves_hidden_first_pair_when_external_arrives(reload_sandbox, monkeypatch):
    s = reload_sandbox
    s.external.clear()
    original = s.c.legacy
    injected = []
    def appeared(assignments=None):
        result = original(assignments)
        if assignments is not None and "pilot_reload/JFINAL" in s.running and not injected:
            injected.append(True)
            s.external.append(dict(endpoint="new-external", accelerator="T4"))
            s.controls["hidden"] = {"pilot_reload/JFINAL"}
            directory = s.c.out / "pilot_reload/JFINAL"
            for name in ("ownership.json", "sessions.json"):
                (directory / name).unlink()
            status = s.m.read_json(directory / "status.json")
            status.pop("owned_endpoint")
            write_json(directory / "status.json", status)
        return result
    monkeypatch.setattr(s.c, "legacy", appeared)
    assert s.c.run() == 0
    second = next(item for item in s.launches if item[0] == "pilot_reload/B13")
    assert "pilot_reload/JFINAL" in second[3]
    assert "capacity_blocked_before_submission" in s.c.events.read_text()


@pytest.mark.parametrize("fault", ["version", "mode", "quality"])
def test_reload_config_must_have_only_approved_phase_amendment(reload_sandbox, fault):
    s = reload_sandbox
    path = s.c.root / s.m.RELOAD_CONFIG
    config = s.m.read_json(path)
    if fault == "version":
        config["code_version"] = "0.2.0"
    elif fault == "mode":
        config["runtime_defaults"]["LATENCY_SWAP_MODE"] = "sleep"
    else:
        config["models"] = {"G": "unauthorized model replacement"}
    write_json(path, config)
    with pytest.raises(ValueError):
        s.c.run()
    assert not s.api_calls and not s.launches


def test_confirmed_global_cap_exceeded_aborts_immediately_not_capacity_wait(sandbox):
    s = sandbox
    s.external += [dict(endpoint="unknown-2", accelerator="T4"), dict(endpoint="unknown-3", accelerator="A100")]
    assert s.c.run() == 1
    assert not s.launches and s.clock["now"] < 1100
    assert "Confirmed global assignment cap exceeded" in s.c.state["error_message"]
    assert len(s.external) == 3


def test_old_master_completed_smoke_must_have_positive_verification(reload_sandbox):
    s = reload_sandbox
    path = s.c.out / "coordinator_state.json"
    old = s.m.read_json(path)
    old["verified_jobs"]["smoke/hybrid"]["ok"] = False
    write_json(path, old)
    with pytest.raises(ValueError, match="did not verify"):
        s.c.run()
    assert not s.api_calls and not s.launches


def test_backend_invalid_response_diagnosis_has_safe_specific_reason(run_v2, tmp_path, monkeypatch):
    c = run_v2.Coordinator(tmp_path)
    monkeypatch.setattr(run_v2.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="Bearer private-api-payload"))
    with pytest.raises(ValueError) as error:
        c.backend()
    message = run_v2.diagnosis(error.value)["error_message"]
    assert "Backend uncertain" in message and "private-api-payload" not in message
