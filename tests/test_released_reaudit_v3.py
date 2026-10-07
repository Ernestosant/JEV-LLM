"""Released CPU re-audits; historical evidence stays read-only, never rent GPUs."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_colab_v3 import ROOT, m, write_json, bound_consent
from test_postpilot_operator_v3 import auditor_job, amended


def released_state(m, root, approval, frozen):
    item = m.job("G_SINGLE", "test")
    directory = root / "results/v3" / item["key"]
    return directory, dict(
        series="v3", root=str(root), output=str(directory), cli_config=str(directory / "sessions.json"),
        job=item, options=dict(approval=str(approval), pilot_report="pinned-pilot.json"),
        evidence=dict(frozen_files=frozen), status="completed", verified=True, released=True,
        completed_execution=True, allocation_attempted=True, owned_endpoint="historical-owned",
        final_collection_attempted=True, deadline_epoch=1234)


def test_terminal_collection_flag_and_changed_control_are_read_only(m, auditor_job, monkeypatch):
    root, approval, frozen, runtime = auditor_job
    frozen["tools/colab/run_v3.py"] = "obsolete-control-sha"
    directory, state = released_state(m, root, approval, frozen)
    write_json(directory / "status.json", state)
    before = (directory / "status.json").read_bytes()
    calls = []

    def cli(args, **kwargs):
        calls.append((args, kwargs))
        write_json(directory / "verification.json", dict(ok=True, runs={"G_SINGLE": dict(records=1500)}))
        return 0, ""

    monkeypatch.setattr(m, "bounded_call", cli)
    monkeypatch.setattr(m.Operator, "validate_frozen", lambda self: pytest.fail("Historical control revalidated"))
    remaining = m.remaining_confirmatory(root)
    assert state["job"] not in remaining and len(remaining) == 6
    assert len(calls) == len(runtime) == 1
    assert 590 < calls[0][1]["timeout"] <= 600
    assert Path(calls[0][0][1]).name == "audit_native_container_v3.py"
    assert (directory / "status.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [("released", False), ("status", "verifying"),
                                          ("verified", False), ("completed_execution", False)])
def test_nonterminal_jobs_keep_strict_control_binding(m, auditor_job, monkeypatch, field, value):
    root, approval, frozen, _ = auditor_job
    control = root / "tools/colab/run_v3.py"
    control.parent.mkdir()
    control.write_text("repaired coordinator", encoding="utf-8")
    frozen["tools/colab/run_v3.py"] = "obsolete-control-sha"
    directory, state = released_state(m, root, approval, frozen)
    state[field] = value
    operator = m.Operator(root, directory, state)
    monkeypatch.setattr(m, "bounded_call", lambda *a, **k: pytest.fail("Unbound checker ran"))
    with pytest.raises(m.Blocked, match="Frozen source/config/data/evidence changed"):
        operator.verify()


@pytest.mark.parametrize("released,owner,gpu,allowance,expected", [
    (False, "owned", "A100", 300, 120),
    (False, "owned", "A100", 120, None),
    (False, None, "A100", 120, 120),
    (False, "owned", "CPU", 120, 120),
    (True, "owned", "A100", 120, 120),
    (True, "owned", "A100", 900, 600),
])
def test_release_reserve_only_for_live_owned_gpu(m, auditor_job, monkeypatch,
                                                 released, owner, gpu, allowance, expected):
    root, approval, frozen, _ = auditor_job
    directory, state = released_state(m, root, approval, frozen)
    state.update(released=released, owned_endpoint=owner)
    state["job"]["gpu"] = gpu
    operator = m.Operator(root, directory, state)
    monkeypatch.setattr(operator, "remaining", lambda *a: allowance)
    calls = []

    def cli(args, **kwargs):
        calls.append(kwargs["timeout"])
        write_json(directory / "verification.json", dict(ok=True, runs={"G_SINGLE": dict(records=1500)}))
        return 0, ""

    monkeypatch.setattr(m, "bounded_call", cli)
    if expected is None:
        with pytest.raises(m.Blocked, match="preserve owned release reserve"):
            operator.verify()
        assert not calls
    else:
        operator.verify()
        assert calls == [expected]


@pytest.mark.parametrize("tamper", ["approval", "verifier", "helper", "inherited", "unfrozen", "science"])
def test_released_scientific_and_exact_file_guards(m, auditor_job, monkeypatch, tamper):
    root, approval, frozen, _ = auditor_job
    directory, state = released_state(m, root, approval, frozen)
    if tamper == "approval":
        approved = m.read_json(approval)
        approved["approved_by"] = "changed"
        write_json(approval, approved)
    elif tamper == "unfrozen":
        frozen.pop("tools/audit_inherited_v3.py")
    elif tamper == "science":
        def reject(*a):
            raise m.Blocked("blocked_budget_approval", "Runtime consent/evidence binding failed")
        monkeypatch.setattr(m, "runtime_approval_gate", reject)
    else:
        name = {"verifier": "verify_run_v3.py", "helper": "audit_native_container_v3.py",
                "inherited": "audit_inherited_v3.py"}[tamper]
        (root / "tools" / name).write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(m, "bounded_call", lambda *a, **k: pytest.fail("Unbound checker ran"))
    with pytest.raises(m.Blocked):
        m.Operator(root, directory, state).verify()


@pytest.mark.parametrize("name", ["src/jevlab/v3/runner.py", "src/jevlab/v3/extra.py",
                                  "config/experiment_v3.json", "data/v3/test_inputs.jsonl",
                                  "prompts/generador.txt"])
def test_released_actual_runtime_scientific_tamper(m, amended, monkeypatch, name):
    fixture, _, _, _ = amended
    frozen = {str(fixture.approval): m.digest(fixture.approval)}
    directory, state = released_state(m, fixture.root, fixture.approval, frozen)
    state["options"]["pilot_report"] = str(fixture.report)
    # Dispatch pins are independent of the scientific validator exercised here.
    monkeypatch.setattr(m, "private_auditor_files", lambda *a: {})
    monkeypatch.setattr(m, "bounded_call", lambda *a, **k: pytest.fail("Tampered science reached CLI"))
    path = fixture.root / name
    if path.suffix == ".json":
        config = m.read_json(path)
        config["models"]["G"]["revision"] = "f" * 40
        write_json(path, config)
    else:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n# scientific tamper\n")
    with pytest.raises(m.Blocked):
        m.Operator(fixture.root, directory, state).verify()


@pytest.mark.parametrize("rc,ok,records", [(1, True, 1500), (0, False, 1500), (0, True, 1499)])
def test_released_requires_fresh_success_and_full_coverage(m, auditor_job, monkeypatch, rc, ok, records):
    root, approval, frozen, _ = auditor_job
    directory, state = released_state(m, root, approval, frozen)
    operator = m.Operator(root, directory, state)
    monkeypatch.setattr(operator, "remaining", lambda *a: 600)
    write_json(directory / "verification.json", dict(ok=True, runs={"G_SINGLE": dict(records=1500)}))

    def cli(*a, **k):
        write_json(directory / "verification.json", dict(ok=ok, runs={"G_SINGLE": dict(records=records)}))
        return rc, ""

    monkeypatch.setattr(m, "bounded_call", cli)
    with pytest.raises(m.Blocked):
        operator.verify()


def test_actual_released_g_single_read_only():
    if not (ROOT / "results/v3/test/G_SINGLE/status.json").is_file():
        pytest.skip("Actual released G_SINGLE unavailable")
    root = ROOT.as_posix()
    command = [sys.executable]
    if os.name == "nt":
        root = "/mnt/" + root[0].lower() + root[2:]
        command = ["wsl.exe", "--exec", "python3"]
    code = r'''
import copy, pathlib, sys, tempfile, time
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "tools/colab"))
import run_v3 as r
directory = root / "results/v3/test/G_SINGLE"
state = r.status_gate(root, directory, r.job("G_SINGLE", "test"))
assert state["status"] == "completed" and state["released"] is True and state["verified"] is True
assert state["final_collection_attempted"] is True
archive, = (directory / "artifacts").glob("*_final.zip")
expected = "49d18f168b4008a4aab0fef407724145d09f2a73a54a7f1c99f155bdb7a4b0f8"
assert r.digest(archive) == expected
protected = [directory / "status.json", * (directory / "artifacts").iterdir(),
             *directory.glob("*receipt*.json"), directory / "verification.json"]
before = {p: r.digest(p) for p in protected if p.is_file()}
def forbidden(*a, **k):
    raise AssertionError("CPU re-audit attempted backend or historical control access")
r.Backend.identity = r.Backend.call = forbidden
r.Operator.validate_frozen = forbidden
original = r.bounded_call
with tempfile.TemporaryDirectory(prefix="jev-released-reaudit-") as temp:
    output = pathlib.Path(temp) / "verification.json"
    def read_only_cli(args, **kwargs):
        args = list(args)
        args[args.index("--output") + 1] = str(output)
        kwargs["log"] = pathlib.Path(temp) / "verification.log"
        assert pathlib.Path(args[1]).name == "audit_native_container_v3.py"
        assert 0 < kwargs["timeout"] <= 600
        return original(args, **kwargs)
    r.bounded_call = read_only_cli
    read_json = r.read_json
    r.read_json = lambda p: read_json(output if pathlib.Path(p) == directory / "verification.json" else p)
    operator = r.Operator(root, directory, copy.deepcopy(state))
    operator.deadline = operator.work_deadline = time.time() + 600
    operator.monotonic_deadline = operator.monotonic_work_deadline = time.monotonic() + 600
    operator.verify()
    result = read_json(output)
    assert result["ok"] and result["runs"]["G_SINGLE"]["records"] == 1500
    assert operator.state == state
assert {p: r.digest(p) for p in before} == before
print("Fresh released G_SINGLE audit PASS: 1500 rows, exact archive, historical evidence unchanged")
'''
    result = subprocess.run([*command, "-B", "-c", code, root], capture_output=True, text=True, timeout=660)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1500 rows, exact archive" in result.stdout
