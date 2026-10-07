"""Synthetic, offline recovery lineage and unchanged-coordinator checks."""

from contextlib import nullcontext
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
def setup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("resume_v3_test", ROOT / "tools/colab/resume_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    core = m.run_v3

    def forbidden(*args, **kwargs):
        pytest.fail("Real backend, dataset, gold, or subprocess access forbidden")

    monkeypatch.setattr(core.Backend, "identity", forbidden)
    monkeypatch.setattr(core.Backend, "call", forbidden)
    monkeypatch.setattr(core, "dataset_gate", forbidden)
    monkeypatch.setattr(core.subprocess, "run", forbidden)
    monkeypatch.setattr(core.subprocess, "Popen", forbidden)
    monkeypatch.setattr(core, "flock", lambda *a, **k: nullcontext())
    write(tmp_path / core.CONFIG, {"synthetic": True})
    source = tmp_path / "tools/colab/run_v3.py"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic frozen source", encoding="utf-8")
    initial = tmp_path / "results/v3/initial_budget.json"
    write(initial, {"synthetic_consent": True})
    options = dict(attempt_prefix="infra03", authorize_initial=True, initial_budget=str(initial), host_ack="synthetic")
    parent = core.Coordinator(tmp_path, "full", **options)
    evidence = dict(bundle_manifest={"config_sha256": core.digest(tmp_path / core.CONFIG)},
                    frozen_files={"tools/colab/run_v3.py": core.digest(source), str(initial): core.digest(initial)},
                    initial_budget=str(initial))
    parent.state["frozen_evidence"] = {j["key"]: evidence for j in parent.jobs if j["phase"] in {"smoke", "pilot"}}
    write(parent.path, parent.state)
    ledger = dict(experiment_id=core.EXPERIMENT_ID, max_gpu_seconds=core.MAX_GPU_SECONDS, jobs={})
    for index, condition in enumerate((*m.PRIOR, "B13_GREEDY")):
        item = next(j for j in parent.jobs if j["key"] == "smoke/" + condition)
        directory = parent.job_directory(item)
        failed = item["key"] == m.TARGET
        run_id = m.FAILED_RUN_ID if failed else str(index) * 32
        state = dict(series="v3", root=str(tmp_path), output=str(directory), cli_config=str(directory / "sessions.json"),
                     job=item, run_id=run_id, status="failed" if failed else "completed", released=True,
                     allocation_attempted=True, startup_ack=True, completed_execution=not failed, verified=not failed,
                     remote_startup_ack=not failed, owned_endpoint=None if failed else "gone-" + condition,
                     before_endpoints=[], unassigned_startup_reconciled=failed,
                     observed_gpu_seconds_upper_bound=1513.25, deadline_epoch=0)
        write(directory / "status.json", state)
        ledger["jobs"][run_id] = dict(key=item["key"], session=item["session"], output=str(directory), released=True,
            release_verified=True, release_evidence="backend_absent", actual_seconds=1513.25,
            endpoint=None if failed else state["owned_endpoint"], unassigned_startup_reconciled=failed)
    # Do not let the core historical scanner read anything outside synthetic fixtures.
    monkeypatch.setattr(core, "budget_ledger", lambda root: ledger)
    return m, tmp_path, options, parent, ledger


def test_target_only_and_immutable_receipt(setup):
    m, root, options, parent, ledger = setup
    c = m.RecoveryCoordinator(root, "full", **options)
    assert c.jobs == parent.jobs
    assert c.execute.__func__ is m.run_v3.Coordinator.execute
    assert c.run.__func__ is m.run_v3.Coordinator.run
    assert c.wait_operator.__func__ is m.RecoveryCoordinator.wait_operator
    for item in c.jobs:
        actual = c.job_options(item)
        expected = parent.job_options(item)
        if item["key"] == m.TARGET:
            expected["attempt_prefix"] = None
            assert c.job_directory(item) == root / "results/v3/recovery/infra03/smoke/B13_GREEDY"
        else:
            assert c.job_directory(item) == parent.job_directory(item)
        assert actual == expected
    before = c.receipt_path.read_bytes()
    receipt = json.loads(before)
    assert receipt["parent_sha256"] == m.run_v3.digest(parent.path)
    assert receipt["failed_lease"] == ledger["jobs"][m.FAILED_RUN_ID]
    assert len(receipt["prior_status_sha256"]) == 4
    assert "ledger_sha256" not in receipt
    # New charges evolve independently of the immutable predecessor receipt.
    ledger["jobs"]["probe"] = {"released": True, "actual_seconds": 15}
    m.RecoveryCoordinator(root, "full", **options)
    assert c.receipt_path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["release", "released", "release_evidence", "live", "remote", "executed",
                                      "config", "source", "consent", "parent", "parent_sha", "status_sha",
                                      "run_id", "charge", "prior", "artifact", "receipt", "missing_receipt", "cap"])
def test_fail_closed(setup, mutation):
    m, root, options, parent, ledger = setup
    c = m.RecoveryCoordinator(root, "full", **options)
    item = next(j for j in parent.jobs if j["key"] == m.TARGET)
    status = parent.job_directory(item) / "status.json"
    state = m.run_v3.read_json(status)
    if mutation == "release":
        ledger["jobs"][m.FAILED_RUN_ID]["release_verified"] = False
    elif mutation == "released":
        state["released"] = False
        write(status, state)
    elif mutation == "release_evidence":
        ledger["jobs"][m.FAILED_RUN_ID]["release_evidence"] = "unknown"
    elif mutation == "cap":
        ledger["max_gpu_seconds"] += 1
    elif mutation == "charge":
        ledger["jobs"][m.FAILED_RUN_ID]["actual_seconds"] = 0
    elif mutation in {"live", "remote", "executed", "run_id"}:
        key, value = {"live": ("owned_endpoint", "live"), "remote": ("remote_startup_ack", True),
                      "executed": ("completed_execution", True), "run_id": ("run_id", "foreign")}[mutation]
        state[key] = value
        write(status, state)
    elif mutation in {"config", "source", "consent"}:
        path = {"config": root / m.run_v3.CONFIG, "source": root / "tools/colab/run_v3.py",
                "consent": Path(options["initial_budget"])}[mutation]
        path.write_text("changed", encoding="utf-8")
    elif mutation == "parent":
        value = m.run_v3.read_json(parent.path)
        value["jobs"][0]["params"]["SEEDS"] = "29"
        write(parent.path, value)
    elif mutation == "parent_sha":
        value = m.run_v3.read_json(parent.path)
        value["extra"] = "mutation outside identity"
        write(parent.path, value)
    elif mutation == "status_sha":
        state["extra"] = "mutation outside identity"
        write(status, state)
    elif mutation == "prior":
        path = parent.job_directory(parent.jobs[0]) / "status.json"
        value = m.run_v3.read_json(path)
        value["verified"] = False
        write(path, value)
    elif mutation == "artifact":
        (status.parent / "raw.out.ipynb").write_text("synthetic", encoding="utf-8")
    elif mutation == "receipt":
        write(c.receipt_path, {"forged": True})
    else:
        c.receipt_path.unlink()
        write(c.path, c.state)
    with pytest.raises(m.run_v3.Blocked):
        m.RecoveryCoordinator(root, "full", **options)


def test_four_prior_smokes_are_observed_not_submitted(setup, monkeypatch):
    m, root, options, parent, ledger = setup
    c = m.RecoveryCoordinator(root, "full", **options)
    observed = []
    submissions = []
    monkeypatch.setattr(m.run_v3, "preflight", lambda *a, **k: {"synthetic": True})
    monkeypatch.setattr(m.run_v3.Backend, "identity", lambda self: {"local_endpoint": None, "assignments": []})
    monkeypatch.setattr(m.run_v3.Operator, "verify", lambda self: observed.append(self.state["job"]["key"]))
    for item in parent.jobs[:4]:
        write(parent.job_directory(item) / "verification.json", {"ok": True})

    def stop_after_first_new(root, output, item, options):
        submissions.append((item, output, options))
        raise m.run_v3.Blocked("blocked_execution", "Synthetic stop before allocation")

    monkeypatch.setattr(m.run_v3, "start_operator", stop_after_first_new)
    evidence = {j["key"]: {"synthetic": True} for j in c.jobs if j["phase"] in {"smoke", "pilot"}}
    assert c.execute(evidence, {}, {}) == 1
    assert observed == ["smoke/" + condition for condition in m.PRIOR]
    assert len(submissions) == 1 and submissions[0][0]["key"] == m.TARGET
    assert submissions[0][2]["attempt_prefix"] is None
    assert m.run_v3.digest(parent.path) == m.run_v3.read_json(c.receipt_path)["parent_sha256"]


def test_main_flags(setup, monkeypatch, capsys):
    m, root, options, parent, ledger = setup
    captured = {}

    def run(self):
        captured.update(self.options)
        return 0

    monkeypatch.setattr(m.RecoveryCoordinator, "run", run)
    args = ["--root", str(root), "--run", "--phase", "full", "--authorize-initial", "--initial-budget",
            options["initial_budget"], "--host-ack", str(root / "host.json"), "--attempt-prefix", "infra03",
            "--authorize-confirmatory", "--approval", str(root / "approval.json"), "--pilot-report", str(root / "pilot.json")]
    assert m.main(args) == 0
    assert captured["authorize_confirmatory"] is True
    assert captured["approval"] == str(root / "approval.json")
    assert captured["pilot_report"] == str(root / "pilot.json")
    assert json.loads(capsys.readouterr().out)["jobs"] == parent.jobs
    for flag, value in (("--phase", "smoke"), ("--attempt-prefix", "infra04")):
        bad = args.copy()
        bad[bad.index(flag) + 1] = value
        with pytest.raises(SystemExit):
            m.main(bad)


@pytest.fixture
def audit_setup(setup, monkeypatch):
    m, root, options, parent, ledger = setup
    core = m.run_v3
    verifier = root / "tools/verify_run_v3.py"
    verifier.write_text("def check_public_provenance(*args):\n    return {}\n", encoding="utf-8")
    (root / "tools/audit_inherited_v3.py").write_bytes((ROOT / "tools/audit_inherited_v3.py").read_bytes())
    frozen = core.read_json(parent.path)
    for value in frozen["frozen_evidence"].values():
        value["frozen_files"]["tools/verify_run_v3.py"] = core.digest(verifier)
    write(parent.path, frozen)
    c = m.RecoveryCoordinator(root, "full", **options)
    item = next(j for j in c.jobs if j["key"] == "pilot/G_SINGLE")
    directory = c.job_directory(item)
    archive = directory / "artifacts/G_SINGLE_pilot_v3_synthetic_final.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"synthetic archive, not model results")
    notebook = archive.with_name("01_G_SINGLE.out.synthetic.ipynb")
    notebook.write_bytes(b"synthetic notebook")
    inputs = root / "data/v3/pilot_inputs.jsonl"
    inputs.parent.mkdir(parents=True)
    inputs.write_bytes(b"synthetic numeric input fixture")
    state = dict(series="v3", root=str(root), output=str(directory), cli_config=str(directory / "sessions.json"),
        job=item, options=dict(approval=None, pilot_report=None), run_id="synthetic-pilot", status="failed",
        verified=False, completed_execution=True, released=True, owned_endpoint="gone-pilot",
        deadline_epoch=123, observed_gpu_seconds_upper_bound=987.25,
        health=dict(known=True, alive=False, run_id="synthetic-pilot", execution=dict(done=True, rc=0)),
        error_message="Original CLI failed; no retry")
    failure = dict(ok=False, runs={item["condition"]: dict(ok=False, error=m.PROVENANCE_ERROR)})
    write(directory / "status.json", state)
    write(directory / "verification.json", failure)
    (directory / "events.jsonl").write_bytes(b'{"status":"failed"}\n')
    ledger["jobs"][state["run_id"]] = dict(actual_seconds=987.25, released=True)
    report = dict(ok=True, runs={item["condition"]: dict(ok=True, condition=item["condition"],
        records=item["expected_rows"], certificate_schema="v3-offline-audit-1", archive=str(archive),
        archive_sha256=core.digest(archive), notebook_sha256=core.digest(notebook))})
    calls = []
    install = c.install_private_auditor
    # Retain real installation/hash guards; stub only offline synthetic audit I/O.
    monkeypatch.setitem(m.sys.modules, "verify_run_v3", m.sys.modules.get("verify_run_v3"))
    def synthetic_install(native_container=False):
        module = install(native_container=native_container)
        def verify(directory, **kwargs):
            calls.append(kwargs)
            return report
        module.verify_run = verify
        module.condition_notebooks = lambda *args: [notebook]
        return module
    monkeypatch.setattr(c, "install_private_auditor", synthetic_install)
    monkeypatch.setattr(core.Backend, "identity", lambda self: dict(assignments=[], local_endpoint=None))
    monkeypatch.setattr(core, "start_operator", lambda *a, **k: pytest.fail("No reinference/submission"))
    monkeypatch.setattr(core.Operator, "verify", lambda *a: pytest.fail("No frozen subprocess verifier"))
    return m, c, item, directory, state, failure, report, ledger, calls


def test_exact_boundary_revalidation_preserves_failure_and_charges(audit_setup):
    m, c, item, directory, state, failure, report, ledger, calls = audit_setup
    status_bytes = (directory / "status.json").read_bytes()
    failure_bytes = (directory / "verification.json").read_bytes()
    charges = json.dumps(ledger, sort_keys=True)
    assert c.wait_operator(item) == report
    current = m.run_v3.read_json(directory / "status.json")
    assert current["status"] == "completed" and current["verified"] is True
    for key in ("run_id", "deadline_epoch", "observed_gpu_seconds_upper_bound", "error_message", "health"):
        assert current[key] == state[key]
    history = directory / "revalidation_history"
    assert (history / "original_status.json").read_bytes() == status_bytes
    assert (history / "original_verification.json").read_bytes() == failure_bytes
    assert list(directory.rglob("status.json")) == [directory / "status.json"]
    assert (directory / "events.jsonl").read_bytes().startswith(b'{"status":"failed"}\n')
    assert json.dumps(ledger, sort_keys=True) == charges
    assert calls[0] == dict(conditions=[item["condition"]], inputs=c.root / "data/v3/pilot_inputs.jsonl",
        config=c.root / m.run_v3.CONFIG, prompts=c.root / "prompts", expected_count=item["count"],
        split="pilot", run_tag="pilot_v3", approval=None, pilot_report=None)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert c.revalidate_operator(item) == report
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mutation", ["other_error", "not_released", "rc1", "audit_failed", "live_endpoint",
    "row_gap", "schema", "verifier_sha", "archive", "health_unknown"])
def test_revalidation_refuses_unrelated_or_incomplete_execution(audit_setup, monkeypatch, mutation):
    m, c, item, directory, state, failure, report, ledger, calls = audit_setup
    if mutation == "other_error":
        failure["runs"][item["condition"]]["error"] = "ValueError: Integrity mismatch"
        write(directory / "verification.json", failure)
    elif mutation == "not_released":
        state["released"] = False
    elif mutation == "rc1":
        state["health"]["execution"]["rc"] = 1
    elif mutation == "health_unknown":
        state["health"]["known"] = False
    elif mutation == "audit_failed":
        report["ok"] = False
    elif mutation == "live_endpoint":
        monkeypatch.setattr(m.run_v3.Backend, "identity", lambda self: dict(assignments=[dict(endpoint="gone-pilot")]))
    elif mutation == "verifier_sha":
        (c.root / "tools/verify_run_v3.py").write_text("changed", encoding="utf-8")
    else:
        result = report["runs"][item["condition"]]
        if mutation == "row_gap":
            result["records"] -= 1
        elif mutation == "schema":
            result["certificate_schema"] = "unknown"
        else:
            result["archive_sha256"] = "0" * 64
    write(directory / "status.json", state)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(m.run_v3.Blocked):
        c.revalidate_operator(item)
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_waits_for_terminal_without_restarting(audit_setup, monkeypatch):
    m, c, item, directory, state, failure, report, ledger, calls = audit_setup
    running = {**state, "status": "running"}
    write(directory / "status.json", running)
    monkeypatch.setattr(m.run_v3.time, "time", lambda: 100)
    monkeypatch.setattr(m.run_v3.time, "sleep", lambda seconds: write(directory / "status.json", state))
    assert c.wait_operator(item) == report


def test_other_error_uses_original_wait(audit_setup, monkeypatch):
    m, c, item, directory, state, failure, report, ledger, calls = audit_setup
    failure["runs"][item["condition"]]["error"] = "ValueError: Other integrity failure"
    write(directory / "verification.json", failure)
    seen = []
    monkeypatch.setattr(m.run_v3.Coordinator, "wait_operator", lambda self, job: seen.append(job))
    c.wait_operator(item)
    assert seen == [item] and calls == []


@pytest.mark.parametrize("mutation", ["deadline", "charge", "failure_history", "receipt"])
def test_promoted_history_is_immutable(audit_setup, mutation):
    m, c, item, directory, state, failure, report, ledger, calls = audit_setup
    c.revalidate_operator(item)
    if mutation in {"deadline", "charge"}:
        current = m.run_v3.read_json(directory / "status.json")
        current["deadline_epoch" if mutation == "deadline" else "observed_gpu_seconds_upper_bound"] += 1
        write(directory / "status.json", current)
    else:
        path = directory / "revalidation_history" / (
            "original_verification.json" if mutation == "failure_history" else "receipt.json")
        path.chmod(0o644)
        write(path, {"tampered": True})
    with pytest.raises(m.run_v3.Blocked):
        c.revalidate_operator(item)


@pytest.fixture
def local_setup(setup, monkeypatch):
    m, root, options, parent, ledger = setup
    core = m.run_v3
    approval, pilot = root / "approval.json", root / "pilot.json"
    write(approval, {"synthetic": True})
    write(pilot, {"go": True})
    options.update(approval=str(approval), pilot_report=str(pilot), authorize_confirmatory=True)
    c = m.RecoveryCoordinator(root, "full", **options)
    approved = dict(experiment_id=core.EXPERIMENT_ID, no_additional_allowance_after_pilot=True,
        initial_authorization_sha256=core.hashlib.sha256(b'{}').hexdigest())
    gates = []
    def gate(root, phase, approval_path, report_path, authorized, **kwargs):
        gates.append((phase, approval_path, report_path, authorized, kwargs))
        core.require(authorized is True and approval_path == str(approval) and report_path == str(pilot),
                     "Actual approval/report required")
        core.require(core.read_json(pilot)["go"] is True, "Actual GO required")
        return approved, core.read_json(pilot)
    monkeypatch.setattr(core, "approval_gate", gate)
    monkeypatch.setattr(core, "initial_budget_gate", lambda *args: {})
    for entry in ledger["jobs"].values():
        entry["release_verified"] = True
    for item in c.jobs[:-1]:
        directory = c.job_directory(item)
        state = dict(series="v3", root=str(root), output=str(directory), job=item,
            cli_config=str(directory / "sessions.json"), status="completed", startup_ack=True,
            completed_execution=True, verified=True, released=True, deadline_epoch=0)
        # Preserve original smoke evidence used by the immutable recovery receipt.
        if not (directory / "status.json").exists():
            write(directory / "status.json", state)
        c.state["verified_jobs"][item["key"]] = {"ok": True}
    python = root / ".venv-analysis/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic executable")
    (root / "tools/local_analysis_v3.py").write_bytes(b"synthetic helper")
    item = c.jobs[-1]
    directory = c.job_directory(item)
    calls, checked = [], []
    monkeypatch.setitem(m.sys.modules, "verify_run_v3", SimpleNamespace(
        check_notebook=lambda path: checked.append(path)))
    def native(command, **kwargs):
        assert all(c.state["verified_jobs"].get(j["key"], {}).get("ok") is True for j in c.jobs[:-1])
        calls.append((command, kwargs))
        archive = directory / "artifacts/analysis_confirmatory_v3.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("analysis.json", json.dumps({"meta": dict(protocol_version="3", code_version="0.3.0",
                n_physical_cases=6000, n_test_problems=500, run_tag="confirmatory_v3", historical_compatibility=False)}))
            z.writestr("report.md", "synthetic report")
        notebook = archive.with_name("07_ANALYSIS.out.synthetic.ipynb")
        notebook.write_bytes(b"synthetic raw notebook; check_notebook mocked")
        helper_nb, inputs = directory / "local_analysis.ipynb", directory / "analysis_inputs_receipt.json"
        helper_nb.write_bytes(b"synthetic helper notebook")
        write(inputs, {"synthetic": True})
        result = dict(ok=True, archive_sha256=core.digest(archive), notebook_sha256=core.digest(notebook),
            archive=str(archive), notebook=str(notebook), receipt=str(directory / "local_analysis_receipt.json"),
            verification=str(directory / "verification.json"), analysis_inputs=str(directory / "analysis_inputs"))
        write(directory / "verification.json", dict(result, extra_metadata="preserve"))
        pin = dict(path=str(approval), sha256=core.digest(approval))
        write(directory / "local_analysis_receipt.json", dict(schema_version=1, execution="local_cpu_papermill",
            allocation_attempted=False, owned_endpoint=None, gpu_seconds=0, result=result,
            pins={"approval": pin}, sources={"synthetic": pin}, helper_notebook_sha256=core.digest(helper_nb),
            input_receipt_sha256=core.digest(inputs)))
        return 0, "archive path before final JSON is deliberately not parsed"
    monkeypatch.setattr(core, "bounded_call", native)
    return m, c, item, directory, calls, checked, gates, approved, ledger


def test_local_analysis_once_and_idempotent_original_verification(local_setup):
    m, c, item, directory, calls, checked, gates, approved, ledger = local_setup
    before = json.dumps(ledger, sort_keys=True)
    evidence = c.state.get("frozen_evidence")
    result = c.local_analysis()
    status = m.run_v3.status_gate(c.root, directory, item)
    assert all(status[k] is True for k in ("startup_ack", "completed_execution", "verified", "released"))
    assert status["allocation_attempted"] is False and status["owned_endpoint"] is None
    assert status["job"] == item and status["options"]["approval"] == c.options["approval"]
    files = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert c.wait_operator(item) == result
    assert files == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert len(calls) == 1 and len(checked) == 2 and len(c.state["verified_jobs"]) == 21
    assert calls[0][1] == dict(timeout=m.run_v3.JOB_SECONDS, log=directory / "local_analysis.log")
    assert calls[0][0] == [str(c.root / ".venv-analysis/Scripts/python.exe"),
        str(c.root / "tools/local_analysis_v3.py"), "--root", str(c.root),
        "--approval", c.options["approval"], "--pilot-report", c.options["pilot_report"]]
    assert gates[0] == ("analysis", c.options["approval"], c.options["pilot_report"], True, {"fresh": False})
    assert json.dumps(ledger, sort_keys=True) == before and c.state.get("frozen_evidence") == evidence


@pytest.mark.parametrize("mutation", ["missing_key", "same_count_foreign", "failed", "unverified", "unreleased",
    "not_executed", "no_authorization", "no_go", "consent", "live_lease", "partial", "rc1"])
def test_local_analysis_barriers_fail_closed(local_setup, monkeypatch, mutation):
    m, c, item, directory, calls, checked, gates, approved, ledger = local_setup
    if mutation in {"missing_key", "same_count_foreign"}:
        c.state["verified_jobs"].pop(c.jobs[5]["key"])
        if mutation == "same_count_foreign":
            c.state["verified_jobs"]["foreign/job"] = {"ok": True}
    elif mutation in {"failed", "unverified", "unreleased", "not_executed"}:
        previous = c.jobs[5]
        path = c.job_directory(previous) / "status.json"
        state = m.run_v3.read_json(path)
        key, value = {"failed": ("status", "failed"), "unverified": ("verified", False),
            "unreleased": ("released", False), "not_executed": ("completed_execution", False)}[mutation]
        state[key] = value
        write(path, state)
    elif mutation == "no_authorization":
        c.options["authorize_confirmatory"] = False
    elif mutation == "no_go":
        write(Path(c.options["pilot_report"]), {"go": False})
    elif mutation == "consent":
        approved["initial_authorization_sha256"] = "changed"
    elif mutation == "live_lease":
        ledger["jobs"]["live"] = {"released": False}
    elif mutation == "partial":
        write(directory / "verification.json", {"ok": False})
    else:
        monkeypatch.setattr(m.run_v3, "bounded_call", lambda *args, **kwargs: (1, ""))
    with pytest.raises(m.run_v3.Blocked):
        c.local_analysis()
    assert calls == [] and not (directory / "status.json").exists()


@pytest.mark.parametrize("mutation", ["archive", "notebook", "receipt", "verification", "pin", "status"])
def test_completed_local_analysis_reverifies_without_rerun(local_setup, mutation):
    m, c, item, directory, calls, checked, gates, approved, ledger = local_setup
    result = c.local_analysis()
    if mutation in {"archive", "notebook"}:
        Path(result[mutation]).write_bytes(b"tampered")
    elif mutation in {"receipt", "verification"}:
        path = directory / ("local_analysis_receipt.json" if mutation == "receipt" else "verification.json")
        value = m.run_v3.read_json(path)
        if mutation == "receipt":
            value["result"]["archive_sha256"] = "changed"
        else:
            value["notebook_sha256"] = "changed"
        write(path, value)
    elif mutation == "pin":
        write(Path(c.options["approval"]), {"changed": True})
    else:
        value = m.run_v3.read_json(directory / "status.json")
        value["owned_endpoint"] = "foreign"
        write(directory / "status.json", value)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises((m.run_v3.Blocked, zipfile.BadZipFile)):
        c.wait_operator(item)
    assert len(calls) == 1
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_latency_hook_preempts_remote_analysis_in_inherited_execute(local_setup, monkeypatch):
    m, c, item, directory, calls, checked, gates, approved, ledger = local_setup
    core = m.run_v3
    c.state["verified_jobs"] = {}
    def wait(self, job):
        self.state["verified_jobs"][job["key"]] = {"ok": True}
    monkeypatch.setattr(core.Coordinator, "wait_operator", wait)
    monkeypatch.setattr(core, "preflight", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(core, "estimate_jobs", lambda *args: {})
    monkeypatch.setattr(core, "remaining_confirmatory", lambda *args: [])
    monkeypatch.setattr(c, "decide_pilot", lambda: {"go": True})
    c.state.update(budget_ready=True, pilot_report_sha256=core.digest(c.options["pilot_report"]))
    monkeypatch.setattr(core, "start_operator", lambda *args: pytest.fail("Remote analysis submission forbidden"))
    assert c.execute({}, {}, {}) == 0
    assert len(calls) == 1 and len(checked) == 2
    assert c.state["outcome"] == "completed" and len(c.state["verified_jobs"]) == 21
    assert c.execute.__func__ is core.Coordinator.execute
