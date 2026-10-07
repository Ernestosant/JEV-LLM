"""Synthetic-only post-pilot ordering/consent checks; no GPU or bundle builds."""

from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from test_colab_v3 import ROOT, m, bound_consent, write_json
from jevlab.common import config_hash
from jevlab.v3.operational import POLICY_ID, MAX_NEW_JOB_SECONDS, TOTAL_GPU_SECONDS


@pytest.fixture
def amended(m, bound_consent):
    fixture = bound_consent
    package = fixture.root / "src/jevlab"
    shutil.copytree(ROOT / "src/jevlab", package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    new = {p.relative_to(package).as_posix(): m.digest(p) for p in
           sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
    old = {k: v for k, v in new.items() if k != "v3/operational.py"}
    old.update({"v3/runner.py": "a" * 64, "v3/latency_study.py": "b" * 64})
    report = m.read_json(fixture.report)
    for condition, path in fixture.archives.items():
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        manifest["config"]["code_sha256"] = old
        manifest["config_hash"] = config_hash(manifest["config"])
        manifest["stages"] = {n: {"ok": True, "seconds": 400} for n in
                              ("environment", "snapshots", "preflight_pre_engines", "engines",
                               "preflight_post_engines", "warmup")}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("progress.json", "{}")
        entry = report["conditions"][condition]
        entry["verification"].update(archive_sha256=m.digest(path), config_hash=manifest["config_hash"])
        entry["estimate"] = {"estimated_test_walltime_s": 7600 if condition == "B13" else 100}
    write_json(fixture.report, report)
    approved = m.read_json(fixture.approval)
    data = fixture.root / "data/v3"
    from jevlab.v3.operational import build_calendar
    items = [json.loads(line) for line in (data / "latency_inputs.jsonl").read_text().splitlines()]
    order = m.read_json(data / "schedule.json")["order"]
    metadata = {r["item_id"]: r for r in m.read_json(data / "latency_plan.json")["items"]}
    calendar = build_calendar(items, order, metadata)
    policy = dict(id=POLICY_ID, approved=True, approved_by="user", approved_utc="2020-01-01T00:00:00+00:00",
                  config_sha256=m.digest(fixture.root / m.CONFIG), pilot_report_sha256=m.digest(fixture.report),
                  old_code_sha256=old, new_code_sha256=new, max_job_seconds=MAX_NEW_JOB_SECONDS,
                  max_gpu_seconds=TOTAL_GPU_SECONDS, expected_test_rows=6000, expected_latency_rows=1800,
                  seeds=[17, 29, 43], repetitions=3, load_counts={"B": 2, "H": 1},
                  latency_source_plan_sha256=m.digest(data / "latency_plan.json"),
                  latency_schedule_sha256=m.digest(data / "schedule.json"), calendar_sha256=config_hash(calendar),
                  authorization_source="Synthetic unit fixture only, not actual funding")
    policy.update({"cohort_" + c.lower(): list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == c))
                   for c in ("A", "C")})
    approved.update(max_gpu_hours=25, pilot_report_sha256=m.digest(fixture.report), operational_amendment=policy)
    write_json(fixture.approval, approved)
    return fixture, approved, report, calendar


def test_phase_deadlines_and_paths(m):
    assert m.JOB_SECONDS == MAX_NEW_JOB_SECONDS == 21600
    assert m.MAX_GPU_SECONDS == TOTAL_GPU_SECONDS == 90000
    for phase in ("smoke", "pilot"):
        assert all(j["deadline_seconds"] == 14400 for j in m.plan(phase))
    for phase in ("test", "latency", "analysis"):
        assert all(j["deadline_seconds"] == 21600 for j in m.plan(phase))
        item = m.plan(phase)[0]
        amended_path = m.phase_bundle_path(ROOT, item, {"operational_amendment": {}})
        assert amended_path.parent == ROOT / "dist/v3/postpilot"
        assert m.phase_bundle_path(ROOT, item).parent == ROOT / "dist/v3"
    assert m.phase_bundle_path(ROOT, m.job("JFINAL", "pilot"), {"operational_amendment": {}}).parent == ROOT / "dist/v3"
    assert sum(j["expected_rows"] for j in m.plan("test")) == 6000
    assert m.plan("latency")[0]["expected_rows"] == 1800


def test_bound_amendment_shared_calendar_and_eta(m, amended):
    fixture, approved, report, calendar = amended
    actual, _ = m.approval_gate(fixture.root, "test", fixture.approval, fixture.report, True, fresh=False)
    assert actual == approved
    authorization = m.runtime_approval_gate(fixture.root, m.job("LATENCY", "latency"), fixture.approval, fixture.report)
    assert authorization["operational_policy_sha256"] == config_hash(approved["operational_amendment"])
    assert m.operational_calendar(fixture.root, approved) == calendar
    phases = [r["residency"] for r in calendar]
    assert list(dict.fromkeys(phases)) == [0, 1, 2]
    assert Counter(phases) == {0: 450, 1: 900, 2: 450}
    eta = m.estimate_jobs(fixture.root, m.plan("latency"), approved, report)["latency/LATENCY"]
    # Fixed setup once (800 + 800), three loads (3 * 1200), hot arms (200).
    assert eta == 1.5 * (1600 + 3600 + 200) + 900 + m.CLEANUP_SECONDS
    assert 14400 < m.estimate_jobs(fixture.root, [m.job("B13", "test")], approved, report)["test/B13"] < 21600
    legacy = {k: v for k, v in approved.items() if k != "operational_amendment"}
    with pytest.raises(m.Blocked):
        m.estimate_jobs(fixture.root, m.plan("latency"), legacy, report)
    with pytest.raises(m.Blocked):
        m.estimate_jobs(fixture.root, [m.job("B13", "test")], legacy, report)
    write_json(fixture.approval, legacy)
    with pytest.raises(m.Blocked, match="Runtime consent"):
        m.runtime_approval_gate(fixture.root, m.job("JFINAL", "test"), fixture.approval, fixture.report)


def test_bootstrap_preserves_phase_work_allowance(m, amended):
    fixture, _, _, _ = amended
    bundle = fixture.root / "synthetic-bundle.zip"
    bundle.write_bytes(b"Synthetic transport placeholder, not a built bundle")
    notebook = fixture.root / "synthetic.ipynb"
    notebook.write_text("{}")
    evidence = dict(bundle=str(bundle), notebook=str(notebook),
                    frozen_files={str(notebook): m.digest(notebook)})
    source = m.bootstrap_code(m.job("JFINAL", "pilot"), evidence, {"run_id": "synthetic"})
    assert "'work_seconds': 13800" in source
    evidence.update(authorization=m.runtime_approval_gate(fixture.root, m.job("JFINAL", "test"),
                                                         fixture.approval, fixture.report),
                    pilot_archives={c: str(p) for c, p in fixture.archives.items()})
    source = m.bootstrap_code(m.job("JFINAL", "test"), evidence, {"run_id": "synthetic"}, "synthetic-approval.json")
    assert "'work_seconds': 21000" in source
    assert "APPROVAL_FILE" in source and "budget_approval.json" in source
    with pytest.raises(m.Blocked, match="work allowance"):
        m.bootstrap_code(m.job("JFINAL", "pilot"), evidence, {}, work_seconds=21000)


@pytest.mark.parametrize("key,value", [
    ("id", "unknown"), ("max_job_seconds", 21601), ("max_gpu_seconds", 90001),
    ("load_counts", {"B": 17, "H": 16}), ("new_code_sha256", {}), ("old_code_sha256", {}),
    ("pilot_report_sha256", "f" * 64), ("config_sha256", "f" * 64),
    ("calendar_sha256", "f" * 64), ("latency_source_plan_sha256", "f" * 64),
    ("latency_schedule_sha256", "f" * 64), ("cohort_a", []), ("seeds", [17]),
    ("repetitions", 4), ("expected_test_rows", 5999), ("approved", False),
])
def test_amendment_tamper_rejected(m, amended, key, value):
    fixture, approved, _, _ = amended
    changed = copy.deepcopy(approved)
    changed["operational_amendment"][key] = value
    write_json(fixture.approval, changed)
    with pytest.raises(m.Blocked) as error:
        m.approval_gate(fixture.root, "latency", fixture.approval, fixture.report, True, fresh=False)
    assert error.value.state == "blocked_budget_approval"


def test_historical_failed_consumption_and_clock_not_extended(m, tmp_path):
    root = tmp_path.resolve()
    directory = root / "results/v3/pilot/B13"
    item = m.job("B13", "pilot")
    state = dict(series="v3", root=str(root), output=str(directory), run_id="old-failed",
                 job=item, allocation_attempted=True, allocation_epoch=10, deadline_epoch=14410,
                 status="failed")
    write_json(directory / "status.json", state)
    ledger = m.budget_ledger(root)
    entry = ledger["jobs"]["old-failed"]
    assert entry["reserved_seconds"] == 14400 and entry["deadline_epoch"] == 14410
    assert m.read_json(directory / "status.json") == state
    entry.update(released=True, release_verified=True, actual_seconds=17789.297)
    assert m.budget_spent(ledger) == 17789.297
    assert ledger["max_gpu_seconds"] - m.budget_spent(ledger) == pytest.approx(72210.703)


def test_bundle_builder_policy_source_binding_without_build(m):
    builder = m.bundle_tools()
    payload, manifest = builder._payload(ROOT, False, True)
    module = "src/jevlab/v3/operational.py"
    assert manifest["operational_policy_id"] == POLICY_ID
    assert manifest["operational_module_sha256"] == builder.sha(payload[module])
    assert manifest["source_mapping"][module] == module
    assert manifest["source_sha256"][module] == manifest["files"][module]
    assert manifest["source_sha256"]["tools/colab/run_v3.py"] == m.digest(ROOT / "tools/colab/run_v3.py")
    assert not any("gold" in name.lower() for name in payload)


@pytest.fixture
def auditor_job(m, tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("verify_run_v3.py", "audit_native_container_v3.py", "audit_inherited_v3.py"):
        shutil.copyfile(ROOT / "tools" / name, tools / name)
    approval = tmp_path / "approval.json"
    approved = dict(approved=True, operational_amendment={}, new_verifier_sha256=m.digest(tools / "verify_run_v3.py"))
    write_json(approval, approved)
    frozen = m.private_auditor_files(tmp_path, approved)
    frozen[str(approval)] = m.digest(approval)
    runtime = []
    monkeypatch.setattr(m, "runtime_approval_gate", lambda *a: runtime.append(a) or {})
    monkeypatch.setattr(m.Backend, "identity", lambda *a, **k: pytest.fail("No backend access"))
    monkeypatch.setattr(m.Backend, "call", lambda *a, **k: pytest.fail("No backend access"))
    return tmp_path, approval, frozen, runtime


@pytest.mark.parametrize("phase,condition", [("test", "JFINAL"), ("latency", "LATENCY"), ("pilot", "JFINAL")])
def test_shared_worker_verifier_dispatch(m, auditor_job, monkeypatch, phase, condition):
    root, approval, frozen, runtime = auditor_job
    item = m.job(condition, phase)
    directory = root / "results/v3" / item["key"]
    directory.mkdir(parents=True)
    state = dict(job=item, deadline_epoch=m.time.time() + 300, options=dict(approval=str(approval), pilot_report="pinned-pilot.json"),
                 evidence=dict(frozen_files=frozen))
    operator = m.Operator(root, directory, state)
    monkeypatch.setattr(operator, "remaining", lambda *a: 300)
    calls = []

    def cli(args, **kwargs):
        calls.append(args)
        write_json(directory / "verification.json", dict(ok=True, runs={condition: dict(records=item["expected_rows"])}))
        return 0, ""

    monkeypatch.setattr(m, "bounded_call", cli)
    operator.verify()
    amended = phase != "pilot"
    assert Path(calls[0][1]).name == ("audit_native_container_v3.py" if amended else "verify_run_v3.py")
    assert ("--pilot-report" in calls[0]) is amended
    assert len(runtime) == int(amended)
    assert calls[0][calls[0].index("--expected-count") + 1] == str(item["count"])


@pytest.mark.parametrize("tamper", ["nested_only", "verifier", "v2", "inherited", "unfrozen"])
def test_private_dispatch_fails_closed(m, auditor_job, monkeypatch, tamper):
    root, approval, frozen, _ = auditor_job
    approved = m.read_json(approval)
    if tamper == "nested_only":
        approved["operational_amendment"]["new_verifier_sha256"] = approved.pop("new_verifier_sha256")
        write_json(approval, approved)
    elif tamper == "unfrozen":
        frozen.pop("tools/audit_inherited_v3.py")
    else:
        name = {"verifier": "verify_run_v3.py", "v2": "audit_native_container_v3.py", "inherited": "audit_inherited_v3.py"}[tamper]
        (root / "tools" / name).write_text("tampered")
    item = m.job("JFINAL", "test")
    operator = m.Operator(root, root / "results/v3" / item["key"],
        dict(job=item, deadline_epoch=m.time.time() + 300, options=dict(approval=str(approval), pilot_report="pinned-pilot.json"),
             evidence=dict(frozen_files=frozen)))
    monkeypatch.setattr(m, "bounded_call", lambda *a, **k: pytest.fail("Unbound checker executed"))
    with pytest.raises(m.Blocked):
        operator.verify()


def test_remaining_confirmatory_reuses_actual_private_dispatch(m, auditor_job, monkeypatch):
    root, approval, frozen, runtime = auditor_job
    item = m.job("JFINAL", "test")
    directory = root / "results/v3" / item["key"]
    state = dict(series="v3", root=str(root), output=str(directory), cli_config=str(directory / "sessions.json"),
                 job=item, options=dict(approval=str(approval), pilot_report="pinned-pilot.json"),
                 evidence=dict(frozen_files=frozen), status="completed", verified=True, released=True,
                 completed_execution=True, deadline_epoch=1234)
    write_json(directory / "status.json", state)
    before = (directory / "status.json").read_bytes()
    calls = []

    def cli(args, **kwargs):
        calls.append(args)
        write_json(directory / "verification.json", dict(ok=True, runs={"JFINAL": dict(records=1500)}))
        return 0, ""

    monkeypatch.setattr(m, "bounded_call", cli)
    remaining = m.remaining_confirmatory(root)
    assert item not in remaining and len(remaining) == 6
    assert Path(calls[0][1]).name == "audit_native_container_v3.py" and len(runtime) == 1
    assert (directory / "status.json").read_bytes() == before
