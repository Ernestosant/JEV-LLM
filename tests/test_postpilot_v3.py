"""Synthetic-only post-pilot identity, history, and fail-closed barrier checks."""

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def m(monkeypatch):
    spec = importlib.util.spec_from_file_location("postpilot_test", ROOT / "tools/colab/postpilot_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("Real dataset, private proof, subprocess, or API access forbidden")

    monkeypatch.setattr(module.r.Backend, "identity", forbidden)
    monkeypatch.setattr(module.r.subprocess, "run", forbidden)
    monkeypatch.setattr(module.r.subprocess, "Popen", forbidden)
    monkeypatch.setattr(module.r, "dataset_gate", forbidden)
    monkeypatch.setattr(module.r, "flock", lambda *a, **k: nullcontext())
    return module


def options():
    return dict(authorize_initial=True, authorize_confirmatory=True, approval="synthetic-approval",
                pilot_report="synthetic-report", initial_budget="synthetic-initial", host_ack="synthetic-host",
                dev_bundle=False)


@pytest.mark.parametrize("phase,count,rows", [("test", 6, 6000), ("latency", 1, 1800)])
def test_phase_identity_and_original_methods(m, tmp_path, phase, count, rows):
    c = m.PostPilotCoordinator(tmp_path, phase, attempt_prefix="infra03", **options())
    assert c.jobs == m.r.plan(phase)
    assert len(c.jobs) == count and sum(j["expected_rows"] for j in c.jobs) == rows
    assert all(j["deadline_seconds"] == 21600 and j["phase"] == phase for j in c.jobs)
    assert c.path.name == "coordinator_postpilot_" + phase + "_state.json"
    assert c.attempt_prefix is None and "attempt_prefix" not in c.options
    assert c.run.__func__ is m.r.Coordinator.run
    assert c.execute.__func__ is m.r.Coordinator.execute
    assert all(c.job_directory(j) == c.out / j["key"] for j in c.jobs)
    assert all(j["deadline_seconds"] == 14400 for j in m.r.plan("prepared"))


@pytest.mark.parametrize("phase,prefix", [("full", None), ("smoke", None), ("pilot", None),
                                         ("analysis", None), ("test", "infra04"), ("test", "../infra03")])
def test_wrong_phase_or_prefix_rejected(m, tmp_path, phase, prefix):
    with pytest.raises(m.r.Blocked):
        m.PostPilotCoordinator(tmp_path, phase, attempt_prefix=prefix, **options())


@pytest.fixture
def history(m, tmp_path, monkeypatch):
    r = m.r
    write(tmp_path / r.CONFIG, {"synthetic": True})
    initial, report, approval = [tmp_path / name for name in ("initial.json", "report.json", "approval.json")]
    write(initial, {"synthetic": True})
    write(report, {"synthetic": True})
    opts = options() | dict(initial_budget=str(initial), pilot_report=str(report), approval=str(approval))
    c = m.PostPilotCoordinator(tmp_path, "test", **opts)
    old_bytes = {"v3/runner.py": b"old runner", "v3/latency_study.py": b"old latency"}
    old = {k: r.hashlib.sha256(v).hexdigest() for k, v in old_bytes.items()}
    for name, data in {"v3/runner.py": "new runner", "v3/latency_study.py": "new latency",
                       "v3/operational.py": "new policy"}.items():
        path = tmp_path / "src/jevlab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    new = {p.relative_to(tmp_path / "src/jevlab").as_posix(): r.digest(p)
           for p in (tmp_path / "src/jevlab/v3").glob("*.py")}
    from jevlab.v3.operational import POLICY_ID
    policy = dict(id=POLICY_ID, approved=True, approved_by="user", approved_utc="2020-01-01T00:00:00+00:00",
                  config_sha256=r.digest(tmp_path / r.CONFIG), pilot_report_sha256=r.digest(report),
                  old_code_sha256=old, new_code_sha256=new, max_job_seconds=21600, max_gpu_seconds=90000,
                  expected_test_rows=6000, expected_latency_rows=1800, seeds=[17, 29, 43], repetitions=3,
                  load_counts={"B": 2, "H": 1}, authorization_source="synthetic only",
                  latency_source_plan_sha256="a" * 64, latency_schedule_sha256="b" * 64,
                  calendar_sha256="c" * 64, cohort_a=["a" + str(i) for i in range(50)],
                  cohort_c=["c" + str(i) for i in range(50)])
    parent = c.out / "coordinator_full_infra03_state.json"
    write(parent, {"synthetic": True})
    receipt = dict(experiment_id=r.EXPERIMENT_ID, root=str(tmp_path), parent=str(parent), parent_sha256=r.digest(parent),
                   config_sha256=r.digest(tmp_path / r.CONFIG), initial_consent_sha256=r.digest(initial),
                   source_sha256=r.hashlib.sha256(b"old controller").hexdigest())
    write(c.receipt_path, receipt)
    c.assets_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(c.assets_path, "w") as z:
        z.writestr("tools/colab/run_v3.py", b"old controller")
        for name, data in old_bytes.items():
            z.writestr("src/jevlab/" + name, data)
    saved = dict(series="v3", root=str(tmp_path), phase="full", max_total_assignments=2, technical_go=True,
                 pilot_report=str(report), pilot_report_sha256=r.digest(report), verified_jobs={}, frozen_evidence={})
    ledger, archives = {"jobs": {}}, {}
    for index, j in enumerate(r.plan("prepared")):
        item = r.job(j["condition"], "smoke", j["session"] + "-infra03") if j["phase"] == "smoke" else j
        directory = (c.out / "recovery/infra03" / j["key"] if j["key"] == "smoke/B13_GREEDY" else
                     c.out / "attempts/infra03" / j["key"] if j["phase"] == "smoke" else c.out / j["key"])
        artifact = directory / "artifacts/synthetic_final.zip"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact, "w") as z:
            z.writestr("synthetic_run/manifest.json", json.dumps({"config": {"code_sha256": old}}))
        notebook = directory / "artifacts" / (r.NOTEBOOKS[j["condition"]] + ".out.synthetic.ipynb")
        write(notebook, {"synthetic": True})
        state = dict(series="v3", root=str(tmp_path), output=str(directory), cli_config=str(directory / "sessions.json"),
                     job=item, run_id=str(index), status="completed", startup_ack=True,
                     completed_execution=True, verified=True, released=True, deadline_epoch=100)
        write(directory / "status.json", state)
        verification = dict(ok=True, runs={j["condition"]: dict(ok=True, records=j["expected_rows"], archive=str(artifact),
                              archive_sha256=r.digest(artifact), notebook_sha256=r.digest(notebook))})
        write(directory / "verification.json", verification)
        saved["verified_jobs"][j["key"]] = verification
        saved["frozen_evidence"][j["key"]] = dict(initial_budget=str(initial),
            frozen_files={"tools/colab/run_v3.py": receipt["source_sha256"], str(initial): r.digest(initial)},
            bundle_manifest=dict(config_sha256=receipt["config_sha256"],
                                 files={"src/jevlab/" + k: v for k, v in old.items()}))
        ledger["jobs"][str(index)] = dict(key=j["key"], session=item["session"], output=str(directory),
            released=True, release_verified=True, release_evidence="backend_absent")
        if j["phase"] == "pilot":
            archives[j["condition"]] = artifact.resolve()
    write(c.recovery_path, saved)
    approved = dict(max_gpu_hours=25, no_additional_allowance_after_pilot=True, operational_amendment=policy,
                    preserved_assets_sha256=r.digest(c.assets_path), recovery_state_sha256=r.digest(c.recovery_path),
                    historical_recovery_receipt_sha256=r.digest(c.receipt_path))
    write(approval, approved)
    monkeypatch.setattr(r, "budget_ledger", lambda root: ledger)
    monkeypatch.setattr(r, "pilot_inputs", lambda *a: archives)
    monkeypatch.setattr(r, "operational_calendar", lambda *a: [])
    return c, approved, saved, ledger


def test_history_read_only_and_full_inventory(history, m):
    c, approved, _, _ = history
    before = {p: p.read_bytes() for p in (c.recovery_path, c.receipt_path, c.assets_path)}
    assert c.validate_history() == approved["operational_amendment"]
    assert all(p.read_bytes() == data for p, data in before.items())
    approved["operational_amendment"]["new_code_sha256"].pop("v3/runner.py")
    write(Path(c.options["approval"]), approved)
    with pytest.raises(ValueError, match="bindings/limits"):
        c.validate_history()


def test_history_accepts_hash_identical_go_staging_copies(history, m, monkeypatch):
    c, _, saved, _ = history
    copies = {}
    for condition in m.r.CONDITIONS:
        original = Path(saved["verified_jobs"]["pilot/" + condition]["runs"][condition]["archive"])
        copy = c.out / "pilot_verified" / condition / original.name
        copy.parent.mkdir(parents=True)
        copy.write_bytes(original.read_bytes())
        copies[condition] = copy
    monkeypatch.setattr(m.r, "pilot_inputs", lambda *args: copies)
    assert c.validate_history()
    copies["JFINAL"].write_bytes(b"changed staging copy")
    with pytest.raises(ValueError, match="Original six pilot archives not linked"):
        c.validate_history()


@pytest.mark.parametrize("field", ["preserved_assets_sha256", "recovery_state_sha256", "historical_recovery_receipt_sha256"])
def test_amendment_sha_mismatch(history, m, field):
    c, approved, _, _ = history
    approved[field] = "f" * 64
    write(Path(c.options["approval"]), approved)
    with pytest.raises(m.r.Blocked, match="SHA mismatch"):
        c.validate_history()


def test_historical_source_proof_not_rebased(history, m):
    c, approved, saved, _ = history
    saved["frozen_evidence"]["pilot/G_SINGLE"]["frozen_files"]["tools/colab/run_v3.py"] = "f" * 64
    write(c.recovery_path, saved)
    approved["recovery_state_sha256"] = m.r.digest(c.recovery_path)
    write(Path(c.options["approval"]), approved)
    with pytest.raises(m.r.Blocked, match="frozen source"):
        c.validate_history()


@pytest.mark.parametrize("gate,state", [("preflight", "blocked_dataset_review"), ("approval_gate", "blocked_budget_approval"),
                                       ("estimate_jobs", "blocked_budget_or_eta"), ("host_gate", "blocked_host_identity")])
def test_original_run_gates_preserved(m, tmp_path, monkeypatch, gate, state):
    c = m.PostPilotCoordinator(tmp_path, "test", **options())
    monkeypatch.setattr(c, "validate_history", lambda: {})
    monkeypatch.setattr(c, "install_private_auditor", lambda *a, **k: None)
    monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a, **k: {})
    monkeypatch.setattr(m.r, "preflight", lambda *a, **k: {})
    monkeypatch.setattr(m.r, "approval_gate", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(m.r, "estimate_jobs", lambda *a, **k: {})
    monkeypatch.setattr(m.r, "remaining_confirmatory", lambda *a, **k: c.jobs)
    monkeypatch.setattr(m.r, "host_gate", lambda *a, **k: {})

    def blocked(*a, **k):
        raise m.r.Blocked(state, "synthetic original gate")

    monkeypatch.setattr(m.r, gate, blocked)
    assert c.run() == 1 and c.state["outcome"] == state
    assert c.state["submissions"] == {}


@pytest.mark.parametrize("phase", ["test", "latency"])
def test_core_source_order_batches_only_new_phase(m, tmp_path, monkeypatch, phase):
    c = m.PostPilotCoordinator(tmp_path, phase, **options())
    monkeypatch.setattr(c, "validate_history", lambda: {})
    monkeypatch.setattr(c, "install_private_auditor", lambda *a, **k: None)
    monkeypatch.setattr(m.r.Backend, "identity", lambda *a, **k: {"local_endpoint": None, "assignments": []})
    monkeypatch.setattr(m.r, "preflight", lambda *a, **k: {})
    submitted = []

    def start(root, directory, item, opts):
        assert item["phase"] == phase and opts["attempt_prefix"] is None
        submitted.append(copy.deepcopy(item))
        state = dict(series="v3", root=str(root), output=str(directory), cli_config=str(directory / "sessions.json"),
                     job=item, startup_ack=True, status="completed", released=True)
        write(directory / "status.json", state)
        return state

    monkeypatch.setattr(m.r, "start_operator", start)
    monkeypatch.setattr(c, "wait_operator", lambda item: c.state["verified_jobs"].update({item["key"]: {"ok": True}}))
    assert c.execute({j["key"]: {} for j in c.jobs}, {}, {}) == 0
    assert submitted == c.jobs and c.state["outcome"] == "completed"
    assert c.state["batch_plan"] == [[j["key"] for j in c.jobs[i:i + 2]] for i in range(0, len(c.jobs), 2)]


@pytest.mark.parametrize("error,should_revalidate", [("ValueError: Missing/nonpublic pinned source provenance", True),
                                                     ("AIME", False), ("ValueError: token mismatch", False)])
def test_only_exact_terminal_failure_delegated(m, tmp_path, monkeypatch, error, should_revalidate):
    c = m.PostPilotCoordinator(tmp_path, "latency", **options())
    item = c.jobs[0]
    monkeypatch.setattr(c, "job_options", lambda item: {})
    monkeypatch.setattr(m.r, "status_gate", lambda *a: dict(status="failed", completed_execution=True, deadline_epoch=0))
    monkeypatch.setattr(m.r, "read_json", lambda *a: {"runs": {"LATENCY": {"error": error}}})
    calls = []
    monkeypatch.setattr(c, "revalidate_operator", lambda item: calls.append(item) or {"ok": True})
    if should_revalidate:
        assert c.wait_operator(item) == {"ok": True}
        assert calls == [item]
    else:
        with pytest.raises(m.r.Blocked):
            c.wait_operator(item)
        assert calls == []


def test_auditor_current_sha_mismatch_rejected(m, tmp_path):
    c = m.PostPilotCoordinator(tmp_path, "test", **options())
    approval = tmp_path / "approval.json"
    write(approval, {"operational_amendment": {"new_verifier_sha256": "a" * 64}})
    c.options["approval"] = str(approval)
    for name in ("audit_native_container_v3.py", "verify_run_v3.py", "audit_inherited_v3.py"):
        path = tmp_path / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic, never executed")
    with pytest.raises(m.r.Blocked, match="auditor SHA mismatch"):
        c.install_private_auditor()


def test_private_v2_uses_new_verifier_not_historical_sha(m, tmp_path, monkeypatch):
    c = m.PostPilotCoordinator(tmp_path, "test", **options())
    helper = tmp_path / "tools/audit_native_container_v3.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# synthetic auditor loader placeholder")
    verifier = helper.with_name("verify_run_v3.py")
    verifier.write_text("# synthetic new checker")
    inherited = helper.with_name("audit_inherited_v3.py")
    inherited.write_text("# synthetic inherited checker")
    sha = m.r.digest(helper)
    monkeypatch.setattr(m, "AUDITOR_SHA256", sha)
    monkeypatch.setattr(m.r, "PRIVATE_AUDITOR_SHA256", sha)
    monkeypatch.setattr(m.r, "INHERITED_AUDITOR_SHA256", m.r.digest(inherited))
    auditor = SimpleNamespace(AUDITOR_ID="synthetic V2", AUDITOR_VERSION="2.0.0", AUDITOR_SHA256=sha,
                             VERIFIER_SHA256=m.r.digest(verifier), INHERITED_AUDITOR_SHA256=m.r.digest(inherited))
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(m.importlib.util, "spec_from_file_location", lambda *a: spec)
    monkeypatch.setattr(m.importlib.util, "module_from_spec", lambda *a: SimpleNamespace(load_verifier=lambda: auditor))
    monkeypatch.setitem(m.sys.modules, "verify_run_v3", SimpleNamespace())
    monkeypatch.delitem(m.sys.modules, "pilot_decision_v3", raising=False)
    approval = tmp_path / "approval.json"
    write(approval, {"approved": True, "operational_amendment": {}, "new_verifier_sha256": m.r.digest(verifier)})
    c.options["approval"] = str(approval)
    assert c.install_private_auditor() is auditor
    assert m.sys.modules["verify_run_v3"] is auditor
    assert c.state["private_auditor"] == {k: getattr(auditor, k) for k in m.AUDITOR_FIELDS}
    assert c.install_private_auditor() is auditor
    c.state["private_auditor"]["AUDITOR_VERSION"] = "changed"
    with pytest.raises(m.r.Blocked, match="auditor changed"):
        c.install_private_auditor()


def test_parent_installs_before_fresh_gates_and_checks_runtime(m, tmp_path, monkeypatch):
    c = m.PostPilotCoordinator(tmp_path, "test", **options())
    calls = []
    monkeypatch.setattr(c, "validate_history", lambda: calls.append("history"))
    monkeypatch.setattr(c, "install_private_auditor", lambda: calls.append("install"))
    monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a: calls.append("runtime"))
    monkeypatch.setattr(c, "job_options", lambda item: {})

    def preflight(*a, **k):
        assert calls == ["history", "install", "runtime"]
        raise m.r.Blocked("blocked_pilot_go", "Stop at first fresh gate")

    monkeypatch.setattr(m.r, "preflight", preflight)
    assert c.run() == 1
    assert c.state["outcome"] == "blocked_pilot_go"
    assert c.state["submissions"] == {}


@pytest.mark.parametrize("change", [{"released": False}, {"completed_execution": False},
                                    {"health": {"known": True, "alive": False, "run_id": "same",
                                                "execution": {"done": True, "rc": 1}}}])
def test_delegated_revalidation_retains_execution_release_guards(m, tmp_path, monkeypatch, change):
    c = m.PostPilotCoordinator(tmp_path, "latency", **options())
    item = c.jobs[0]
    state = dict(status="failed", verified=False, released=True, completed_execution=True, run_id="same",
                 health=dict(known=True, alive=False, run_id="same", execution=dict(done=True, rc=0)))
    state.update(change)
    monkeypatch.setattr(c, "job_options", lambda item: {})
    monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a: {})
    monkeypatch.setattr(m.r, "status_gate", lambda *a: state)
    monkeypatch.setattr(m.RecoveryCoordinator, "__init__", lambda *a, **k: pytest.fail("Recovery constructor called"))
    with pytest.raises(m.r.Blocked, match="Execution/release proof"):
        c.revalidate_operator(item)
    assert not hasattr(c, "local_analysis")


def test_cli_output_compact_and_phase_scoped(m, tmp_path, monkeypatch, capsys):
    def run(c):
        c.state["outcome"] = "completed"
        c.state["frozen_evidence"] = {"synthetic": "x" * 100000}
        return 0

    monkeypatch.setattr(m.PostPilotCoordinator, "run", run)
    args = ["--run", "--root", str(tmp_path), "--phase", "latency", "--attempt-prefix", "infra03",
            "--authorize-initial", "--authorize-confirmatory"]
    for name in ("initial-budget", "host-ack", "approval", "pilot-report"):
        args.extend(["--" + name, str(tmp_path / (name + ".json"))])
    assert m.main(args) == 0
    output = capsys.readouterr().out
    assert len(output) < 1000
    result = json.loads(output)
    assert result["outcome"] == "completed" and result["phase"] == "latency" and result["rc"] == 0
    assert Path(result["state"]).name == "coordinator_postpilot_latency_state.json"


def test_full_revalidation_preserves_failed_bytes_identity_and_charges(m, tmp_path, monkeypatch):
    c = m.PostPilotCoordinator(tmp_path, "latency", **options())
    item = c.jobs[0]
    directory = c.job_directory(item)
    archive, notebook = directory / "artifacts/synthetic_final.zip", directory / "artifacts/synthetic.ipynb"
    write(archive, {"synthetic": True})
    write(notebook, {"synthetic": True})
    write(tmp_path / m.r.CONFIG, {"synthetic": True})
    write(tmp_path / "data/v3/test_inputs.jsonl", {"synthetic": True})
    state = dict(series="v3", root=str(tmp_path), output=str(directory), cli_config=str(directory / "sessions.json"),
        job=item, options=c.options, status="failed", verified=False, completed_execution=True, released=True,
        run_id="same", owned_endpoint="gone", deadline_epoch=1234, deadline_monotonic=5678,
        actual_seconds=999, health=dict(known=True, alive=False, run_id="same", execution=dict(done=True, rc=0)))
    failure = dict(ok=False, runs={"LATENCY": {"ok": False, "error": m.PROVENANCE_ERROR}})
    write(directory / "status.json", state)
    write(directory / "verification.json", failure)
    before_status, before_failure = (directory / "status.json").read_bytes(), (directory / "verification.json").read_bytes()
    result = dict(ok=True, condition="LATENCY", records=1800, certificate_schema="v3-offline-audit-1",
                  archive=str(archive), archive_sha256=m.r.digest(archive), notebook_sha256=m.r.digest(notebook))
    report = dict(ok=True, runs={"LATENCY": result})
    calls = []

    def audit(*a, **k):
        calls.append(k)
        return report

    auditor = SimpleNamespace(verify_run=audit, condition_notebooks=lambda *a: [notebook])
    c.state["private_auditor"] = {"synthetic": "V2"}
    monkeypatch.setattr(c, "job_options", lambda item: {})
    monkeypatch.setattr(c, "install_private_auditor", lambda *a, **k: auditor)
    monkeypatch.setattr(m.r, "runtime_approval_gate", lambda *a: {})
    monkeypatch.setattr(m.r.Backend, "identity", lambda *a, **k: {"assignments": []})
    assert c.revalidate_operator(item) == report
    repaired = m.r.read_json(directory / "status.json")
    assert repaired["status"] == "completed" and repaired["verified"] is True
    for key in ("run_id", "deadline_epoch", "deadline_monotonic", "actual_seconds", "health"):
        assert repaired[key] == state[key]
    assert (directory / "revalidation_history/original_status.json").read_bytes() == before_status
    assert (directory / "revalidation_history/original_verification.json").read_bytes() == before_failure
    assert calls[0]["expected_count"] == 500 and calls[0]["split"] == "test"
    assert c.revalidate_operator(item) == report
    assert len(calls) == 2 and c.state["submissions"] == {}
