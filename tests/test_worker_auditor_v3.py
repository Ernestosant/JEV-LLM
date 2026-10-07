"""Detached-interpreter auditor regression checks; no allocation or evidence writes."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def r():
    spec = importlib.util.spec_from_file_location("worker_auditor_test", ROOT / "tools/colab/run_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("preimport", [False, True])
def test_actual_historical_pilot_in_fresh_interpreter(preimport):
    reports = list((ROOT / "results/v3/pilot").glob("decision_*.json"))
    if not reports or not (ROOT / "results/v3/pilot_verified").is_dir():
        pytest.skip("Read-only historical pilot unavailable")
    # Historical reports bind WSL absolute archive paths. Run there on Windows.
    root = ROOT.as_posix()
    command = [sys.executable]
    if os.name == "nt":
        root = "/mnt/" + root[0].lower() + root[2:]
        command = ["wsl.exe", "--exec", "python3"]
    code = r'''
import pathlib, sys
from types import SimpleNamespace
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "tools/colab"))
import run_v3 as r
def forbidden(*a, **k):
    raise AssertionError("Offline auditor attempted subprocess/backend access")
r.subprocess.run = r.subprocess.Popen = forbidden
report = root / "results/v3/pilot" / sys.argv[2]
approval = root / "results/v3/operational_amendment/approval.json"
# Do not call the canonical view builder: the actual existing view is read-only.
r.canonical_pilot = lambda root: pathlib.Path(root) / "results/v3/pilot_verified"
import pilot_decision_v3 as pilot
original = pilot.verify_run
old = pilot.decide(r.canonical_pilot(root), root / "data/v3/pilot_inputs.jsonl",
                   config=root / r.CONFIG, prompts=root / "prompts", run_tag="pilot_v3")
assert not old["go"]
assert any("source revision" in e.lower() or "provenance" in e.lower() for e in old["errors"])
if sys.argv[3] == "False":
    del sys.modules["pilot_decision_v3"]
sys.modules["audit_inherited_v3"] = SimpleNamespace(AUDITOR_SHA256="stale")
approved, fresh = r.approval_gate(root, "test", approval, report, True)
assert fresh["go"] and fresh["aggregate"]["observed_cases"] == 600
auditor = sys.modules["verify_run_v3"]
assert auditor.AUDITOR_VERSION == "2.0.0"
assert auditor.VERIFIER_SHA256 == approved["new_verifier_sha256"]
assert auditor.AUDITOR_SHA256 == r.PRIVATE_AUDITOR_SHA256
assert auditor.INHERITED_AUDITOR_SHA256 == r.INHERITED_AUDITOR_SHA256
assert sys.modules["pilot_decision_v3"].verify_run is auditor.verify_run
assert auditor.verify_run is not original
assert r.install_private_auditor(root, approved) is auditor
delta = {k for k in set(approved["operational_amendment"]["old_code_sha256"]) |
         set(approved["operational_amendment"]["new_code_sha256"])
         if approved["operational_amendment"]["old_code_sha256"].get(k) !=
            approved["operational_amendment"]["new_code_sha256"].get(k)}
assert delta == {"v3/runner.py", "v3/latency_study.py", "v3/operational.py"}
print("original rejects inherited provenance; approval-bound V2 freshly verifies 600 rows; exact three-file delta")
'''
    result = subprocess.run([*command, "-B", "-c", code, root, reports[0].name, str(preimport)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "freshly verifies 600 rows" in result.stdout


@pytest.mark.parametrize("amended,fresh", [(False, True), (True, False), (True, True)])
def test_approval_read_and_validated_before_auditor_and_pilot(r, monkeypatch, amended, fresh):
    calls = []
    approved = dict(approved=True, scope="test", config_sha256="config", pilot_report_sha256="report",
                    approved_by="user", approved_utc="2020-01-01T00:00:00+00:00", max_gpu_hours=25)
    if amended:
        approved["operational_amendment"] = {}
    monkeypatch.setattr(r, "read_json", lambda p: calls.append("approval") or approved)
    monkeypatch.setattr(r, "digest", lambda p: "config" if Path(p).as_posix().endswith(r.CONFIG) else "report")
    monkeypatch.setattr(r, "install_private_auditor", lambda *a: calls.append("install"))
    monkeypatch.setattr(r, "pilot_gate", lambda *a, **k: calls.append("pilot") or {})
    monkeypatch.setattr(r, "runtime_approval_gate", lambda *a: {"operational_amendment": {}})
    r.approval_gate(ROOT, "test", "approval", "report", True, fresh=fresh)
    assert calls == ["approval", *(["install"] if amended and fresh else []), "pilot"]
    approved["scope"] = "invalid"
    calls.clear()
    with pytest.raises(r.Blocked, match="Approval must bind"):
        r.approval_gate(ROOT, "test", "approval", "report", True, fresh=fresh)
    assert calls == ["approval"]


def test_bad_checker_sha_blocks_before_import(r, monkeypatch):
    approved = dict(approved=True, operational_amendment={}, new_verifier_sha256="0" * 64)
    monkeypatch.setattr(r.importlib.util, "spec_from_file_location",
                        lambda *a: pytest.fail("Unapproved code imported"))
    with pytest.raises(r.Blocked, match="SHA mismatch"):
        r.install_private_auditor(ROOT, approved)


@pytest.mark.parametrize("field", ["AUDITOR_ID", "AUDITOR_VERSION", "AUDITOR_SHA256",
                                   "VERIFIER_SHA256", "INHERITED_AUDITOR_SHA256"])
def test_loaded_metadata_mismatch_is_rejected(r, monkeypatch, field):
    approved = r.read_json(ROOT / "results/v3/operational_amendment/approval.json")
    original_exec = r.importlib.util.spec_from_file_location
    def spec(name, path):
        result = original_exec(name, path)
        execute = result.loader.exec_module
        def load(module):
            execute(module)
            if name == "audit_native_container_v3":
                original_load = module.load_verifier
                def forged():
                    auditor = original_load()
                    setattr(auditor, field, "0" * 64)
                    return auditor
                module.load_verifier = forged
        result.loader.exec_module = load
        return result
    monkeypatch.setattr(r.importlib.util, "spec_from_file_location", spec)
    monkeypatch.delitem(sys.modules, "verify_run_v3", raising=False)
    # Restore helper bindings after testing this process-local failure.
    for name in ("audit_inherited_v3", "audit_native_container_v3"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    with pytest.raises(r.Blocked, match="Loaded auditor identity mismatch"):
        r.install_private_auditor(ROOT, approved)


@pytest.mark.parametrize("blocked", [True, False])
def test_worker_startup_records_safe_diagnosis_without_allocation(r, monkeypatch, tmp_path, blocked):
    out = tmp_path.resolve()
    item = r.job("G_SINGLE", "test")
    lock = "/tmp/jev-v3-operator-" + item["session"] + ".lock"
    state = dict(root=str(ROOT), output=str(out), series="v3", status="submission_intent",
                 startup_ack=False, job=item, operator_lock=lock, operator_lock_fd=123,
                 allocation_attempted=False, options={}, evidence={})
    saved = []
    monkeypatch.setattr(r, "read_json", lambda p: state)
    monkeypatch.setattr(r, "status_gate", lambda *a: state)
    monkeypatch.setattr(r.os, "fstat", lambda fd: SimpleNamespace(st_dev=1, st_ino=2))
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda p, *a, **k: SimpleNamespace(st_dev=1, st_ino=2)
                        if str(p).replace("\\", "/") == lock else original_stat(p, *a, **k))
    monkeypatch.setattr(r, "Operator", lambda *a: SimpleNamespace(
        save=lambda status, **fields: saved.append(dict(status=status, **fields)),
        run=lambda: pytest.fail("Startup failure reached allocation")))
    def fail(*a, **k):
        message = "Pilot binding failed token=private_value Bearer private_bearer https://private.invalid\n"
        raise r.Blocked("blocked_pilot_go", message) if blocked else ValueError(message)
    monkeypatch.setattr(r, "preflight", fail)
    assert r.main(["--root", str(ROOT), "--_worker", str(out)]) == 1
    final = saved[-1]
    assert final["status"] == "failed" and final["released"] is True
    assert final["completed_execution"] is False and final["verified"] is False
    assert final["release_evidence"] == "durable_no_allocation"
    assert final["error_type"] == ("Blocked" if blocked else "ValueError")
    assert ("Pilot binding failed" in final["error_message"]) == blocked
    assert "private" not in final["error_message"] and "\n" not in final["error_message"]
    assert state["allocation_attempted"] is False
