"""Synthetic fixtures only: never execute a kernel, statistics, gold, API or GPU."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import local_analysis_v3 as local


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def change(path, key, value):
    data = json.loads(path.read_text())
    data[key] = value
    put(path, data)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "synthetic"
    config = {"models": {"synthetic": {}}, "implementation_profile": "synthetic"}
    put(root / local.core.CONFIG, config)
    monkeypatch.setattr(local.core, "config_gate", lambda root: config)
    initial = dict(approved=True, experiment_id=local.core.EXPERIMENT_ID,
        scope="all_gpu_phases_conditional_technical_go", max_gpu_hours=25, max_total_assignments=2,
        gpu="A100", implementation_profile="synthetic", models=config["models"],
        config_sha256=local.digest(root / local.core.CONFIG), technical_go_required=True,
        approved_by="synthetic", approved_utc="2026-01-01T00:00:00+00:00", authorization_source="synthetic")
    put(root / "results/v3/initial_budget.json", initial)
    report = dict(protocol_version="3", code_version="0.3.0", go=True, decision="go", errors=[],
        budget_approval=False, conditions={c: dict(verified=True, denominator=150 if c in local.core.SAMPLED else 50,
            verification=dict(ok=True, condition=c, records=150 if c in local.core.SAMPLED else 50),
            rates=dict(denominator=150 if c in local.core.SAMPLED else 50, timeout_infra_rate=0))
            for c in local.core.CONDITIONS},
        aggregate=dict(complete=True, denominator=600, observed_cases=600, timeout_infra_rate=0))
    pilot = root / "results/v3/pilot/actual.json"
    put(pilot, report)
    approval = root / "results/v3/actual_approval.json"
    put(approval, dict(approved=True, scope="test_and_latency", experiment_id=local.core.EXPERIMENT_ID,
        config_sha256=local.digest(root / local.core.CONFIG), pilot_report_sha256=local.digest(pilot),
        max_gpu_hours=25, approved_by="synthetic", approved_utc="2026-01-01T00:00:00+00:00",
        initial_authorization_sha256=hashlib.sha256(json.dumps(initial, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(), no_additional_allowance_after_pilot=True))
    auditor_path = root / "tools/audit_native_container_v3.py"
    auditor_path.parent.mkdir()
    auditor_path.write_text("from fixture_auditor import auditor\ndef load_verifier():\n    return auditor\n")
    (root / "tools/verify_run_v3.py").write_text("# synthetic verifier identity\n")
    analysis = root / "src/jevlab/v3/analysis.py"
    analysis.parent.mkdir(parents=True)
    analysis.write_text("# synthetic original analysis identity; never imported\n")
    calls = []
    def verify(directory, **kwargs):
        calls.append((directory, kwargs))
        condition = kwargs["conditions"][0]
        archive = next(directory.glob("*_final.zip"))
        notebook = next(directory.glob("*.ipynb"))
        result = dict(ok=True, condition=condition, records=1800 if condition == "LATENCY" else
            1500 if condition in local.core.SAMPLED else 500, certificate_schema="v3-offline-audit-1",
            archive=str(archive), archive_sha256=local.digest(archive), notebook_sha256=local.digest(notebook))
        return dict(ok=True, runs={condition: result})
    auditor = SimpleNamespace(verify_run=verify,
        condition_notebooks=lambda directory, condition: list(directory.glob("*.out.*.ipynb")),
        AUDITOR_SHA256=local.digest(auditor_path), VERIFIER_SHA256=local.digest(root / "tools/verify_run_v3.py"),
        AUDITOR_ID="synthetic", AUDITOR_VERSION="synthetic")
    monkeypatch.setitem(sys.modules, "fixture_auditor", SimpleNamespace(auditor=auditor))
    jobs = {}
    for item in [*local.core.plan("test"), *local.core.plan("latency")]:
        directory = root / "results/v3" / item["key"]
        artifacts = directory / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / (item["condition"] + "_confirmatory_v3_synthetic_final.zip")).write_bytes(b"synthetic archive")
        (artifacts / (local.core.NOTEBOOKS[item["condition"]] + ".out.synthetic.ipynb")).write_bytes(b"synthetic raw notebook")
        put(directory / "status.json", dict(series="v3", root=local.wsl_path(root), output=local.wsl_path(directory),
            cli_config=local.wsl_path(directory / "sessions.json"), job=item, run_id=item["key"],
            allocation_attempted=True, status="completed", completed_execution=True, verified=True, released=True,
            options=dict(approval=local.wsl_path(approval), pilot_report=local.wsl_path(pilot), authorize_confirmatory=True)))
        put(directory / "verification.json", verify(artifacts, conditions=[item["condition"]]))
        jobs[item["key"]] = dict(key=item["key"], session=item["session"], output=local.wsl_path(directory),
            released=True, release_verified=True, release_evidence="backend_absent", actual_seconds=100)
    put(root / local.core.BUDGET_LEDGER, dict(experiment_id=local.core.EXPERIMENT_ID,
        max_gpu_seconds=90000, jobs=jobs))
    calls.clear()
    executions = []
    def execute_notebook(source, destination, **kwargs):
        executions.append((source, destination, kwargs))
        nb = nbformat.read(source, as_version=4)
        nb.metadata.papermill = dict(end_time="2026-01-01T00:00:01+00:00", exception=None)
        nb.cells[0].metadata.papermill = dict(status="completed")
        nb.cells[0].execution_count = 1
        nbformat.write(nb, destination)
        archive = Path(destination).parent / "analysis_confirmatory_v3.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name in local.OUTPUTS:
                z.writestr(name, json.dumps(dict(meta=dict(protocol_version="3", code_version="0.3.0",
                    n_physical_cases=6000, n_test_problems=500, run_tag="confirmatory_v3",
                    historical_compatibility=False))) if name == "analysis.json" else "synthetic report")
    monkeypatch.setitem(sys.modules, "papermill", SimpleNamespace(execute_notebook=execute_notebook))
    for name in ("status_gate", "budget_ledger"):
        monkeypatch.setattr(local.core, name, lambda *a, **k: pytest.fail("WSL-sensitive core gate forbidden"))
    return SimpleNamespace(root=root, approval=approval, pilot=pilot, auditor=auditor, calls=calls,
                           executions=executions, pm=execute_notebook)


def test_execute_contract(fixture):
    f = fixture
    before = {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    result = local.execute(f.root, f.approval, f.pilot)
    assert result["ok"] is True and len(f.calls) == 7 and len(f.executions) == 1
    staging = Path(result["analysis_inputs"])
    assert len(list(staging.iterdir())) == 14
    for p, raw in before.items():
        assert p.read_bytes() == raw
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["allocation_attempted"] is False and receipt["owned_endpoint"] is None
    assert len(receipt["sources"]) == 14 and set(receipt["pins"]) == {
        "helper", "auditor", "verifier", "original_analysis", "config", "approval", "pilot_report", "initial", "ledger"}
    assert result["archive_sha256"] == local.digest(result["archive"])
    assert result["notebook_sha256"] == local.digest(result["notebook"])
    assert Path(result["notebook"]).name.startswith("07_ANALYSIS.out.")
    source, destination, kwargs = f.executions[0]
    assert kwargs == dict(kernel_name="python3", cwd=str(f.root), progress_bar=False)
    nb = nbformat.read(source, as_version=4)
    assert len(nb.cells) == 1
    code = nb.cells[0].source
    compile(code, source, "exec")
    assert code.index("sys.path.insert(0, str(root / 'tools'))") < code.index("spec.loader.exec_module(helper)")
    assert "analysis._integrity_module = lambda: auditor" in code
    assert repr(str(f.root / "tools/audit_native_container_v3.py")) in code
    assert "analysis.run_analysis(" in code and "n_boot" not in code
    assert "sys.modules" not in code and "test_gold.jsonl" in code
    for directory, kwargs in f.calls:
        assert kwargs == dict(conditions=[directory.parent.name], inputs=f.root / "data/v3/test_inputs.jsonl",
            config=f.root / local.core.CONFIG, prompts=f.root / "prompts", approval=f.approval,
            pilot_report=f.pilot, expected_count=500, split="test", run_tag="confirmatory_v3")
    assert not (Path(result["receipt"]).parent / "status.json").exists()
    with pytest.raises(ValueError, match="collision"):
        local.execute(f.root, f.approval, f.pilot)


@pytest.fixture
def amended_fixture(fixture, monkeypatch):
    f = fixture
    change(f.approval, "operational_amendment", {"policy_id": "synthetic"})
    # Only the runtime amendment gate is out of scope for this synthetic fixture.
    monkeypatch.setattr(local.core, "runtime_approval_gate", lambda *a: {
        "operational_amendment": {"policy_id": "synthetic"}})
    v1 = f.root / "tools/audit_inherited_v3.py"
    v1.write_text("# synthetic V1 identity\n")
    v3 = f.root / "tools/audit_operational_latency_v3.py"
    v3.write_text("from fixture_auditor import auditor\n"
                  "AUDITOR_VERSION = '3.0.0'\n"
                  f"APPROVAL_SHA256 = {local.digest(f.approval)!r}\n"
                  f"VERIFIER_SHA256 = {f.auditor.VERIFIER_SHA256!r}\n"
                  f"INHERITED_AUDITOR_SHA256 = {f.auditor.AUDITOR_SHA256!r}\n"
                  f"V1_AUDITOR_SHA256 = {local.digest(v1)!r}\n"
                  "def load_verifier():\n    return auditor\n")
    f.auditor.INHERITED_AUDITOR_SHA256 = f.auditor.AUDITOR_SHA256
    f.auditor.V1_AUDITOR_SHA256 = local.digest(v1)
    f.auditor.AUDITOR_SHA256 = local.digest(v3)
    f.auditor.AUDITOR_VERSION = "3.0.0"
    f.latency = f.root / "results/v3/latency/LATENCY"
    change(f.latency / "status.json", "technical_phase_revalidation", {
        "auditor": {"AUDITOR_SHA256": f.auditor.AUDITOR_SHA256}})
    original = f.auditor.verify_run
    def verify(*args, **kwargs):
        report = original(*args, **kwargs)
        metadata = dict(id=f.auditor.AUDITOR_ID, version=f.auditor.AUDITOR_VERSION,
                        sha256=f.auditor.AUDITOR_SHA256)
        report["auditor"] = metadata
        if kwargs["conditions"] == ["LATENCY"]:
            report["runs"]["LATENCY"].update(auditor=metadata, proof_projection=dict(
                applied=True, memory_only=True, rows=8, phase_id="synthetic-H2",
                proof_sha256="synthetic-phase-proof", source="phases/H2/selector_equivalence.json"))
        return report
    f.auditor.verify_run = verify
    put(f.latency / "verification.json", verify(f.latency / "artifacts", conditions=["LATENCY"]))
    f.calls.clear()
    return f


def test_amended_v3_selection_and_receipt(amended_fixture):
    f = amended_fixture
    before = {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    result = local.execute(f.root, f.approval, f.pilot)
    assert result["ok"] and len(f.calls) == 7 and len(f.executions) == 1
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["auditor"]["AUDITOR_VERSION"] == "3.0.0"
    assert receipt["auditor"]["AUDITOR_SHA256"] == f.auditor.AUDITOR_SHA256
    assert receipt["pins"]["auditor"]["path"] == str(f.root / "tools/audit_operational_latency_v3.py")
    assert {"v1_auditor", "v2_auditor", "latency_status", "latency_verification"} <= receipt["pins"].keys()
    code = nbformat.read(f.executions[0][0], as_version=4).cells[0].source
    assert repr(str(f.root / "tools/audit_operational_latency_v3.py")) in code
    assert "helper.load_verifier()" in code
    for path, raw in before.items():
        assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["missing_receipt", "receipt_sha", "source_version", "v3_sha",
    "verifier_sha", "v2_sha", "v1_sha", "approval_sha", "stored_proof", "missing_proof",
    "stored_auditor", "fresh_proof", "fresh_version", "fresh_sha", "latency_records"])
def test_amended_v3_guards(amended_fixture, fault):
    f = amended_fixture
    status = f.latency / "status.json"
    stored = f.latency / "verification.json"
    if fault == "missing_receipt":
        change(status, "technical_phase_revalidation", {})
    elif fault == "receipt_sha":
        change(status, "technical_phase_revalidation", {"auditor": {"AUDITOR_SHA256": "wrong"}})
    elif fault in {"source_version", "v3_sha", "verifier_sha", "v2_sha", "v1_sha"}:
        setattr(f.auditor, {"source_version": "AUDITOR_VERSION", "v3_sha": "AUDITOR_SHA256",
            "verifier_sha": "VERIFIER_SHA256", "v2_sha": "INHERITED_AUDITOR_SHA256",
            "v1_sha": "V1_AUDITOR_SHA256"}[fault], "wrong")
    elif fault == "approval_sha":
        change(f.approval, "approved_by", "different synthetic approver")
    elif fault in {"stored_proof", "missing_proof", "stored_auditor"}:
        data = json.loads(stored.read_text())
        if fault == "stored_auditor":
            data["auditor"]["sha256"] = "wrong"
        elif fault == "missing_proof":
            del data["runs"]["LATENCY"]["proof_projection"]
        else:
            data["runs"]["LATENCY"]["proof_projection"]["proof_sha256"] = "wrong"
        put(stored, data)
    else:
        original = f.auditor.verify_run
        def broken(*args, **kwargs):
            report = original(*args, **kwargs)
            if kwargs["conditions"] == ["LATENCY"]:
                if fault == "fresh_proof":
                    report["runs"]["LATENCY"]["proof_projection"]["applied"] = False
                elif fault == "latency_records":
                    report["runs"]["LATENCY"]["records"] = 1799
                else:
                    report["auditor"]["version" if fault == "fresh_version" else "sha256"] = "wrong"
            return report
        f.auditor.verify_run = broken
    with pytest.raises(ValueError):
        local.execute(f.root, f.approval, f.pilot)
    assert not f.executions


def test_analysis_verifier_receives_state(fixture, monkeypatch):
    original = local.core.Operator.verify
    calls = []
    def verify(operator):
        assert operator.state == {}
        calls.append(operator)
        return original(operator)
    monkeypatch.setattr(local.core.Operator, "verify", verify)
    assert local.execute(fixture.root, fixture.approval, fixture.pilot)["ok"]
    assert len(calls) == 1


@pytest.mark.parametrize("target", ["tools/audit_native_container_v3.py", "tools/audit_inherited_v3.py",
    "results/v3/latency/LATENCY/status.json", "results/v3/latency/LATENCY/verification.json"])
def test_amended_pins_survive_execution(amended_fixture, monkeypatch, target):
    f = amended_fixture
    def changed(source, destination, **kwargs):
        f.pm(source, destination, **kwargs)
        (f.root / target).write_text("changed synthetic pin")
    monkeypatch.setitem(sys.modules, "papermill", SimpleNamespace(execute_notebook=changed))
    with pytest.raises(ValueError, match="Pinned evidence changed"):
        local.execute(f.root, f.approval, f.pilot)
    assert not (f.root / "results/v3/analysis/ANALYSIS/local_analysis_receipt.json").exists()


@pytest.mark.parametrize("fault", ["status", "completed", "verified", "released", "root", "output", "job",
    "cli_config", "status_approval", "status_pilot", "authorized", "stored_ok", "stored_hash", "stored_records",
    "extra", "extra_dir", "missing_notebook", "approval_config", "approval_pilot", "approval_scope", "approval_no",
    "initial_hash", "initial_cap", "pilot_no_go", "pilot_incomplete", "ledger_cap", "ledger_active", "ledger_unverified",
    "ledger_evidence", "ledger_nan", "ledger_negative", "ledger_overcap", "ledger_missing", "ledger_output",
    "staging_collision", "receipt_collision", "artifact_collision", "fresh_audit", "fresh_schema"])
def test_guards_before_papermill(fixture, fault):
    f = fixture
    directory = f.root / "results/v3/test/G_SINGLE"
    status = directory / "status.json"
    fields = {"status": ("status", "failed"), "completed": ("completed_execution", False),
        "verified": ("verified", False), "released": ("released", False), "root": ("root", "/wrong"),
        "output": ("output", "/wrong"), "job": ("job", {}), "cli_config": ("cli_config", "/wrong")}
    if fault in fields:
        change(status, *fields[fault])
    elif fault in {"status_approval", "status_pilot", "authorized"}:
        data = json.loads(status.read_text())
        data["options"][{"status_approval": "approval", "status_pilot": "pilot_report", "authorized": "authorize_confirmatory"}[fault]] = False
        put(status, data)
    elif fault.startswith("stored_"):
        path = directory / "verification.json"
        data = json.loads(path.read_text())
        if fault == "stored_ok":
            data["ok"] = False
        else:
            data["runs"]["G_SINGLE"][{"stored_hash": "archive_sha256", "stored_records": "records"}[fault]] = "wrong"
        put(path, data)
    elif fault in {"extra", "extra_dir", "missing_notebook"}:
        artifacts = directory / "artifacts"
        if fault == "missing_notebook":
            next(artifacts.glob("*.ipynb")).unlink()
        elif fault == "extra_dir":
            (artifacts / "extra").mkdir()
        else:
            (artifacts / "extra").write_bytes(b"extra")
    elif fault.startswith("approval_") or fault == "initial_hash":
        fields = {"approval_config": ("config_sha256", "wrong"), "approval_pilot": ("pilot_report_sha256", "wrong"),
            "approval_scope": ("scope", "test"), "approval_no": ("approved", False),
            "initial_hash": ("initial_authorization_sha256", "wrong")}
        change(f.approval, *fields[fault])
    elif fault == "initial_cap":
        change(f.root / "results/v3/initial_budget.json", "max_gpu_hours", 26)
    elif fault.startswith("pilot_"):
        change(f.pilot, "go" if fault == "pilot_no_go" else "aggregate", False if fault == "pilot_no_go" else {})
        change(f.approval, "pilot_report_sha256", local.digest(f.pilot))
    elif fault.startswith("ledger_"):
        path = f.root / local.core.BUDGET_LEDGER
        data = json.loads(path.read_text())
        entry = data["jobs"]["test/G_SINGLE"]
        if fault == "ledger_cap":
            data["max_gpu_seconds"] = 90001
        elif fault == "ledger_missing":
            del data["jobs"]["test/G_SINGLE"]
        else:
            fields = {"ledger_active": ("released", False), "ledger_unverified": ("release_verified", False),
                "ledger_evidence": ("release_evidence", "claimed"), "ledger_nan": ("actual_seconds", float("nan")),
                "ledger_negative": ("actual_seconds", -1), "ledger_overcap": ("actual_seconds", 90000),
                "ledger_output": ("output", "/wrong")}
            key, value = fields[fault]
            entry[key] = value
        put(path, data)
    elif fault.endswith("collision"):
        path = f.root / "results/v3/analysis/ANALYSIS" / {"staging_collision": "analysis_inputs",
            "receipt_collision": "analysis_inputs_receipt.json", "artifact_collision": "artifacts"}[fault]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"collision")
    else:
        original = f.auditor.verify_run
        def broken(*args, **kwargs):
            report = original(*args, **kwargs)
            if fault == "fresh_audit":
                report["ok"] = False
            else:
                next(iter(report["runs"].values()))["certificate_schema"] = "wrong"
            return report
        f.auditor.verify_run = broken
    with pytest.raises((ValueError, OSError)):
        local.execute(f.root, f.approval, f.pilot)
    assert not f.executions


@pytest.mark.parametrize("fault", ["exception", "incomplete", "zip_extra", "zip_coverage", "mutated_source",
                                  "changed_cell", "unexecuted_cell", "cell_error"])
def test_execution_failure_never_stamps_success(fixture, monkeypatch, fault):
    f = fixture
    def broken(source, destination, **kwargs):
        if fault == "exception":
            raise RuntimeError("synthetic papermill failure")
        f.pm(source, destination, **kwargs)
        if fault in {"incomplete", "changed_cell", "unexecuted_cell", "cell_error"}:
            nb = nbformat.read(destination, as_version=4)
            if fault == "incomplete":
                nb.metadata.papermill.end_time = None
            elif fault == "changed_cell":
                nb.cells[0].source = "pass"
            elif fault == "unexecuted_cell":
                nb.cells[0].execution_count = None
            else:
                nb.cells[0].outputs = [nbformat.v4.new_output("error", ename="synthetic", evalue="synthetic", traceback=[])]
            nbformat.write(nb, destination)
        elif fault == "mutated_source":
            (f.root / local.core.CONFIG).write_text("changed")
        else:
            archive = Path(destination).parent / "analysis_confirmatory_v3.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for name in local.OUTPUTS:
                    z.writestr(name, json.dumps(dict(meta={})) if name == "analysis.json" else "synthetic")
                if fault == "zip_extra":
                    z.writestr("extra", "synthetic")
    monkeypatch.setitem(sys.modules, "papermill", SimpleNamespace(execute_notebook=broken))
    with pytest.raises((RuntimeError, ValueError)):
        local.execute(f.root, f.approval, f.pilot)
    out = f.root / "results/v3/analysis/ANALYSIS"
    assert not (out / "local_analysis_receipt.json").exists()
    assert not (out / "status.json").exists()


def test_windows_identity():
    assert local.wsl_path(r"C:\projects\JEV-LLM") == "/mnt/c/projects/JEV-LLM"
    assert local.wsl_path("/synthetic/root") == "/synthetic/root"
    with pytest.raises(ValueError):
        local.wsl_path(r"\\server\share\root")


def test_symlink_guard(fixture):
    path = fixture.root / "results/v3/test/G_SINGLE/artifacts/extra"
    try:
        path.symlink_to(fixture.approval)
    except OSError:
        pytest.skip("Native Windows symlink privilege unavailable")
    with pytest.raises(ValueError, match="Symlink"):
        local.safe(path)


def test_symlink_rejected_before_execution(fixture, monkeypatch):
    target = fixture.root / "results/v3/test/G_SINGLE/artifacts"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError, match="Symlink"):
        local.execute(fixture.root, fixture.approval, fixture.pilot)
    assert not fixture.executions


def test_released_no_allocation_lease(fixture):
    path = fixture.root / local.core.BUDGET_LEDGER
    ledger = json.loads(path.read_text())
    ledger["jobs"]["synthetic_preflight"] = dict(released=True, release_verified=True,
        release_evidence="durable_no_allocation", allocation_attempted=False, actual_seconds=0, endpoint=None)
    put(path, ledger)
    assert local.execute(fixture.root, fixture.approval, fixture.pilot)["ok"]


def test_cli_contract(fixture, monkeypatch, capsys):
    expected = dict(ok=True, archive="synthetic", archive_sha256="synthetic", notebook_sha256="synthetic")
    def execute(root, approval, pilot):
        assert (root, approval, pilot) == (fixture.root, fixture.approval, fixture.pilot)
        return expected
    monkeypatch.setattr(local, "execute", execute)
    assert local.main(["--root", str(fixture.root), "--approval", str(fixture.approval),
                       "--pilot-report", str(fixture.pilot)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
