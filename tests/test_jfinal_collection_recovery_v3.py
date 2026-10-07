"""Recovery checks use temporary evidence only; never allocate or run real recovery."""

import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab"))
m = importlib.import_module("recover_jfinal_collection_v3")
p = importlib.import_module("postpilot_v3")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def packed(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo()
            info.filename = name
            archive.writestr(info, content)
    return stream.getvalue()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    directory = root / "results/v3/test/JFINAL"
    write(root / m.r.CONFIG, {})
    marker = dict(series="v3", session="jev-v3-test-jfinal", endpoint="released-endpoint",
                  run_id=m.RUN_ID, output=str(directory), config_sha256=m.r.digest(root / m.r.CONFIG))
    state = dict(series="v3", root=str(root), output=str(directory), job=m.r.job("JFINAL", "test"),
                 cli_config=str(directory / "sessions.json"), run_id=m.RUN_ID, status="failed", verified=False,
                 startup_ack=True, remote_startup_ack=True, completed_execution=True, released=True,
                 health=dict(execution=dict(run_id=m.RUN_ID, done=True, rc=0)), remote_marker=marker,
                 owned_endpoint=marker["endpoint"], deadline_epoch=20, deadline_monotonic=30,
                 observed_gpu_seconds_upper_bound=12.5, evidence={"source": "unchanged"},
                 collection_operation="local_integrity", options={})
    write(directory / "status.json", state)
    ledger = dict(experiment_id=m.r.EXPERIMENT_ID, max_gpu_seconds=m.r.MAX_GPU_SECONDS,
                  jobs={m.RUN_ID: dict(key="test/JFINAL", session=marker["session"], output=str(directory),
                        endpoint=marker["endpoint"], released=True, release_verified=True,
                        release_evidence="backend_absent", actual_seconds=12.5, deadline_epoch=19.99)})
    write(root / m.r.BUDGET_LEDGER, ledger)
    loose = {"results/v3/JFINAL/run/predictions.jsonl": b"synthetic public predictions\n"}
    final = packed({n.removeprefix("results/"): value for n, value in loose.items()})
    # Deliberately not a ZIP: recovery must never parse checkpoint bytes.
    files = {**loose, m.FINAL_MEMBER: final, m.NOTEBOOK_MEMBER: b'{"completed": true}',
             "results/v3/JFINAL_checkpoint.zip": b"opaque checkpoint, do not expand",
             "job_status_v3.json": json.dumps(state["health"]["execution"]).encode()}
    manifest = dict(marker=marker, files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()})
    outer = packed({**files, "COLLECTION_MANIFEST.json": json.dumps(manifest).encode()})
    (directory / "collection.pending.zip").write_bytes(outer)
    monkeypatch.setattr(m, "SNAPSHOT_SHA", hashlib.sha256(outer).hexdigest())
    monkeypatch.setattr(m, "SNAPSHOT_BYTES", len(outer))
    monkeypatch.setattr(m, "PINS", {n: (len(files[n]), hashlib.sha256(files[n]).hexdigest()) for n in m.PINS})
    metadata = dict(AUDITOR_ID="jev-v3-native-container-provenance-audit", AUDITOR_VERSION="2.0.0",
                    AUDITOR_SHA256=m.r.PRIVATE_AUDITOR_SHA256, VERIFIER_SHA256=m.VERIFIER_SHA,
                    INHERITED_AUDITOR_SHA256=m.r.INHERITED_AUDITOR_SHA256)
    report = dict(ok=True, runs={"JFINAL": dict(ok=True, records=1500,
                  archive_sha256=m.PINS[m.FINAL_MEMBER][1], notebook_sha256=m.PINS[m.NOTEBOOK_MEMBER][1])})
    monkeypatch.setattr(m.r.Backend, "identity", lambda *a: pytest.fail("No backend/API access"))
    monkeypatch.setattr(m.r.Backend, "call", lambda *a, **kw: pytest.fail("No backend/API access"))
    return root, directory, state, files, report, metadata


def test_recover_preserves_identity_and_no_remote_hash_claim(evidence, monkeypatch):
    root, directory, original, files, report, metadata = evidence
    original_bytes = (directory / "status.json").read_bytes()
    ledger_bytes = (root / m.r.BUDGET_LEDGER).read_bytes()
    def audit(*args):
        assert (directory / m.HISTORY / "original_status.json").read_bytes() == original_bytes
        assert (directory / m.HISTORY / "original_verification_absent.json").is_file()
        assert m.r.read_json(directory / "status.json")["status"] == "failed"
        return report, metadata
    monkeypatch.setattr(m, "audit", audit)
    receipt = m.recover(root)
    state = m.r.read_json(directory / "status.json")
    assert state["status"] == "completed" and state["verified"] is True
    assert {k: v for k, v in state.items() if k not in m.CHANGED_FIELDS} == {k: v for k, v in original.items() if k not in m.CHANGED_FIELDS}
    assert (root / m.r.BUDGET_LEDGER).read_bytes() == ledger_bytes
    assert receipt["remote_snapshot_sha256"] is None and receipt["remote_snapshot_hash_persisted"] is False
    assert len(list((directory / "artifacts").iterdir())) == 2
    for name in m.PINS:
        assert (directory / "artifacts" / Path(name).name).read_bytes() == files[name]
    assert m.validate_receipt(root, directory, state) == receipt
    with pytest.raises(m.r.Blocked, match="Only unrecovered"):
        m.recover(root)


def test_audit_failure_never_marks_completed(evidence, monkeypatch):
    root, directory, *_ = evidence
    original = (directory / "status.json").read_bytes()
    def fail(*args):
        raise m.r.Blocked("blocked_integrity", "Synthetic audit failure")
    monkeypatch.setattr(m, "audit", fail)
    with pytest.raises(m.r.Blocked):
        m.recover(root)
    assert (directory / "status.json").read_bytes() == original
    assert not (directory / "verification.json").exists()
    assert (directory / m.HISTORY / "original_status.json").read_bytes() == original


@pytest.mark.parametrize("field,value", [("run_id", "wrong"), ("completed_execution", False),
    ("remote_startup_ack", False), ("released", False), ("root", "wrong")])
def test_execution_identity_fail_closed(evidence, field, value):
    root, directory, state, *_ = evidence
    state[field] = value
    write(directory / "status.json", state)
    with pytest.raises(m.r.Blocked):
        m.recover(root)
    assert not (directory / m.HISTORY).exists()


def test_snapshot_corruption_and_marker_fail_closed(evidence):
    _, directory, state, *_ = evidence
    with pytest.raises(m.r.Blocked, match="marker"):
        m.validate_snapshot(directory / "collection.pending.zip", {})
    path = directory / "collection.pending.zip"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(m.r.Blocked, match="SHA/size"):
        m.validate_snapshot(path, state["remote_marker"])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "C:evil", "a//b"])
def test_unsafe_archive_paths(name):
    with zipfile.ZipFile(io.BytesIO(packed({name: b"x"}))) as archive:
        with pytest.raises(m.r.Blocked, match="Unsafe"):
            m.inventory(archive)


def test_separate_expansion_cap_and_opaque_checkpoint(evidence, monkeypatch):
    _, directory, state, *_ = evidence
    path = directory / "collection.pending.zip"
    with zipfile.ZipFile(path) as outer:
        expanded = sum(i.file_size for i in outer.infolist())
    monkeypatch.setattr(m.r, "MAX_COLLECTION_BYTES", expanded)
    _, commitments = m.validate_snapshot(path, state["remote_marker"])
    assert commitments["outer_expanded_bytes"] + commitments["final_expanded_bytes"] > expanded
    monkeypatch.setattr(m.r, "MAX_COLLECTION_BYTES", expanded - 1)
    with pytest.raises(m.r.Blocked, match="expansion"):
        m.validate_snapshot(path, state["remote_marker"])


def test_selected_inner_expansion_cap(monkeypatch):
    content = packed({"public.jsonl": b"x" * 4096})
    monkeypatch.setattr(m.r, "MAX_COLLECTION_BYTES", 4095)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        with pytest.raises(m.r.Blocked, match="expansion"):
            m.inventory(archive)


def test_cli_requires_explicit_recovery_authorization(tmp_path):
    with pytest.raises(SystemExit) as error:
        m.main(["--root", str(tmp_path)])
    assert error.value.code == 2


def test_postpilot_receipt_still_requires_fresh_audit(evidence, monkeypatch):
    root, directory, _, _, report, metadata = evidence
    monkeypatch.setattr(m, "audit", lambda *a: (report, metadata))
    m.recover(root)
    coordinator = object.__new__(p.PostPilotCoordinator)
    coordinator.root = root
    coordinator.options = dict(approval="synthetic", pilot_report="synthetic")
    coordinator.state = dict(verified_jobs={})
    monkeypatch.setattr(coordinator, "job_options", lambda *a: None)
    monkeypatch.setattr(coordinator, "job_directory", lambda *a: directory)
    monkeypatch.setattr(coordinator, "revalidate_operator", lambda *a: pytest.fail("No provenance-only recovery"))
    calls = []
    class Auditor:
        def verify_run(self, *a, **kw):
            calls.append(kw)
            return copy.deepcopy(report)
    for name, value in metadata.items():
        setattr(Auditor, name, value)
    monkeypatch.setattr(coordinator, "install_private_auditor", lambda: Auditor())
    monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a: None)
    monkeypatch.setattr(coordinator, "save", lambda *a: None)
    assert coordinator.wait_operator(m.r.job("JFINAL", "test")) == report
    assert len(calls) == 1 and calls[0]["expected_count"] == 500
    report["ok"] = False
    with pytest.raises(m.r.Blocked, match="Fresh"):
        coordinator.wait_operator(m.r.job("JFINAL", "test"))


def test_missing_verification_reports_collection_diagnosis(evidence, monkeypatch):
    root, directory, *_ = evidence
    coordinator = object.__new__(p.PostPilotCoordinator)
    coordinator.root = root
    monkeypatch.setattr(coordinator, "job_options", lambda *a: None)
    monkeypatch.setattr(coordinator, "job_directory", lambda *a: directory)
    with pytest.raises(m.r.Blocked, match="no verification.json; collection failed at local_integrity"):
        coordinator.wait_operator(m.r.job("JFINAL", "test"))


def test_real_pinned_v2_loader_and_full_audit_arguments(evidence, monkeypatch):
    root, directory, state, _, report, metadata = evidence
    source = Path(__file__).resolve().parents[1]
    (root / "tools").mkdir()
    for name in ("verify_run_v3.py", "audit_native_container_v3.py", "audit_inherited_v3.py"):
        shutil.copyfile(source / "tools" / name, root / "tools" / name)
    approval, pilot = root / "approval.json", root / "pilot.json"
    write(approval, dict(approved=True, operational_amendment={}, new_verifier_sha256=m.VERIFIER_SHA))
    write(pilot, {})
    state["options"] = dict(approval=str(approval), pilot_report=str(pilot))
    previous = {n: sys.modules.get(n) for n in ("verify_run_v3", "audit_inherited_v3", "audit_native_container_v3")}
    try:
        module = m.r.install_private_auditor(root, m.r.read_json(approval))
        assert {k: getattr(module, k) for k in m.AUDITOR_FIELDS} == metadata
        calls = []
        def verify(*args, **kwargs):
            calls.append((args, kwargs))
            return copy.deepcopy(report)
        monkeypatch.setattr(module, "verify_run", verify)
        monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a: None)
        assert m.audit(root, directory, state) == (report, metadata)
        args, kwargs = calls[0]
        assert args == (directory / "artifacts",)
        assert kwargs == dict(conditions=["JFINAL"], inputs=root / "data/v3/test_inputs.jsonl",
            config=root / m.r.CONFIG, prompts=root / "prompts", expected_count=500, split="test",
            run_tag="confirmatory_v3", approval=approval, pilot_report=pilot)
        report["runs"]["JFINAL"]["records"] = 1499
        with pytest.raises(m.r.Blocked, match="Fresh full V2"):
            m.audit(root, directory, state)
        (root / "tools/verify_run_v3.py").write_bytes(b"unapproved checker")
        with pytest.raises(m.r.Blocked, match="SHA mismatch"):
            m.audit(root, directory, state)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@pytest.mark.parametrize("evidence_name", ["original_status.json", "original_collection.pending.zip",
                                          "original_verification_absent.json"])
def test_receipt_evidence_tamper_rejected(evidence, monkeypatch, evidence_name):
    root, directory, _, _, report, metadata = evidence
    monkeypatch.setattr(m, "audit", lambda *a: (report, metadata))
    m.recover(root)
    path = directory / m.HISTORY / evidence_name
    path.chmod(0o644)
    path.write_bytes(b"changed")
    with pytest.raises((m.r.Blocked, ValueError)):
        m.validate_receipt(root, directory, m.r.read_json(directory / "status.json"))
