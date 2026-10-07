"""Synthetic CPU-only preparation tests; never prepare real approvals or bundles."""

import copy
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from test_colab_v3 import ROOT, m, bound_consent, write_json, forbidden
from test_postpilot_operator_v3 import amended


@pytest.fixture
def prep(m, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_postpilot_test", ROOT / "tools/colab/prepare_postpilot_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "r", m)
    monkeypatch.setattr(m.subprocess, "run", forbidden)
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m.bundle_tools(), "build", forbidden)
    import verify_run_v3
    monkeypatch.setattr(verify_run_v3, "read_archive", forbidden)
    return module


@pytest.fixture
def preparation(prep, amended, monkeypatch):
    fixture, approved, report, calendar = amended
    root = fixture.root.resolve()
    out = root / "results/v3"
    fixture.report = out / "pilot/decision.json"
    report["meta"] = {"generated_utc": "2020-01-01T00:00:00+00:00"}
    write_json(fixture.report, report)
    initial = {"approved": True, "max_gpu_hours": 25, "original": "synthetic consent \u00f1"}
    write_json(out / "initial_budget.json", initial)
    history = dict(root=str(root), pilot_report=str(fixture.report), pilot_report_sha256=prep.r.digest(fixture.report))
    write_json(out / "coordinator_full_infra03_recovery_state.json", history)
    write_json(out / "recovery/infra03/recovery_receipt.json", {"synthetic": True})
    asset = out / "operational_amendment/pre_amendment_assets.zip"
    asset.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset, "w") as archive:
        archive.writestr("synthetic", "Never actual evidence")
    write_json(root / "tools/verify_run_v3.py", {"synthetic": True})
    ledger = dict(experiment_id=prep.r.EXPERIMENT_ID, max_gpu_seconds=90000,
                  jobs={"prior": dict(released=True, release_verified=True, actual_seconds=10000)})
    write_json(root / prep.r.BUDGET_LEDGER, ledger)
    monkeypatch.setattr(prep, "PILOT_SHA256", prep.r.digest(fixture.report))
    monkeypatch.setattr(prep, "ASSETS_SHA256", prep.r.digest(asset))
    monkeypatch.setattr(prep.r, "initial_budget_gate", lambda *args: initial)
    monkeypatch.setattr(prep.r, "dataset_gate", lambda *args: {"sealed": True})
    monkeypatch.setattr(prep.r, "budget_ledger", lambda *args: copy.deepcopy(ledger))
    calls = []

    def history_gate(checker):
        # A synthetic stand-in for the separately tested thirteen-job history gate.
        value = prep.r.read_json(checker.options["approval"])
        assert value["recovery_state_sha256"] == prep.r.digest(checker.recovery_path)
        assert value["historical_recovery_receipt_sha256"] == prep.r.digest(checker.receipt_path)
        assert value["preserved_assets_sha256"] == prep.r.digest(checker.assets_path)
        calls.append("history")
        return value["operational_amendment"]

    monkeypatch.setattr(prep.PostPilotCoordinator, "validate_history", history_gate)
    return root, ledger, calls, approved, report, calendar


def test_prepare_write_once_bound_and_offline(prep, preparation):
    root, ledger, calls, _, report, calendar = preparation
    before = prep.r.utc()
    results = prep.prepare(root, True)
    approval = prep.r.read_json(results["approval"])
    record = prep.r.read_json(results["preregistration_eta"])
    assert calls == ["history"]
    assert prep.r.epoch(before) <= prep.r.epoch(approval["approved_utc"]) <= prep.r.epoch(prep.r.utc())
    assert approval["approved_utc"] == approval["operational_amendment"]["approved_utc"]
    assert approval["new_verifier_sha256"] == prep.r.digest(root / "tools/verify_run_v3.py")
    assert "new_verifier_sha256" not in approval["operational_amendment"]
    assert approval["initial_authorization_sha256"] == prep.hashlib.sha256(json.dumps(
        prep.r.read_json(root / "results/v3/initial_budget.json"), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert record["approval_sha256"] == prep.r.digest(results["approval"])
    assert record["calendar"] == calendar and len(calendar) == 1800
    assert sum(j["expected_rows"] for j in prep.r.plan("test")) == 6000
    assert record["remaining_estimates"]["test/B13"] == 1.5 * (7600 + 6 * 400) + 900 + 600
    assert record["remaining_estimates"]["latency/LATENCY"] == 1.5 * (200 + 1600 + 3600) + 900 + 600
    assert record["projected_total_gpu_seconds"] == 10000 + sum(record["remaining_estimates"].values()) <= 90000
    assert record["cpu_only"] is True and record["allocation_permitted"] is False
    assert prep.r.read_json(root / prep.r.BUDGET_LEDGER) == ledger
    contents = {p: p.read_bytes() for p in (Path(results["approval"]), Path(results["preregistration_eta"]))}
    with pytest.raises(ValueError, match="already exists"):
        prep.prepare(root, True)
    assert all(p.read_bytes() == raw for p, raw in contents.items())


@pytest.mark.parametrize("failure", ["consent", "budget", "live", "pilot_pin", "asset_pin", "history", "runtime", "existing_job", "changed_source"])
def test_fail_closed_before_any_publication(prep, preparation, monkeypatch, failure):
    root, ledger, calls, _, _, _ = preparation
    if failure == "budget":
        ledger["jobs"]["prior"]["actual_seconds"] = 89999
    elif failure == "live":
        ledger["jobs"]["prior"]["released"] = False
    elif failure == "pilot_pin":
        monkeypatch.setattr(prep, "PILOT_SHA256", "f" * 64)
    elif failure == "asset_pin":
        monkeypatch.setattr(prep, "ASSETS_SHA256", "f" * 64)
    elif failure in {"history", "runtime"}:
        def reject(*args):
            raise ValueError("Synthetic gate failed")
        if failure == "history":
            monkeypatch.setattr(prep.PostPilotCoordinator, "validate_history", reject)
        else:
            monkeypatch.setattr(prep.r, "runtime_approval_gate", reject)
    elif failure == "existing_job":
        write_json(root / "results/v3/test/G_SINGLE/status.json", {})
    elif failure == "changed_source":
        original = prep.r.runtime_approval_gate
        def changing(*args):
            value = original(*args)
            with (root / "tools/verify_run_v3.py").open("a") as stream:
                stream.write(" ")
            return value
        monkeypatch.setattr(prep.r, "runtime_approval_gate", changing)
    with pytest.raises(ValueError):
        prep.prepare(root, failure != "consent")
    assert not (root / "results/v3/operational_amendment/approval.json").exists()
    assert not (root / "results/v3/operational_amendment/preregistration_eta.json").exists()


def test_exact_original_eta_formula_without_original_reader(prep, amended, monkeypatch):
    fixture, approved, report, _ = amended
    manifests = {c: prep.manifest_snapshot(p, report["conditions"][c]["verification"]["archive_sha256"])
                 for c, p in fixture.archives.items()}
    jobs = [*prep.r.plan("test"), *prep.r.plan("latency")]
    actual, _ = prep.historical_estimates(fixture.root, jobs, approved, report, manifests)
    # The controller arithmetic must match; replace only its offending reader with synthetic snapshots.
    import verify_run_v3
    by_path = {p.resolve(): {"manifest": manifests[c], "archive_sha256": prep.r.digest(p)}
               for c, p in fixture.archives.items()}
    monkeypatch.setattr(verify_run_v3, "read_archive", lambda p: by_path[Path(p).resolve()])
    assert actual == prep.r.estimate_jobs(fixture.root, jobs, approved, report)


@pytest.mark.parametrize("change", ["nan", "negative", "failed_stage", "deadline", "missing_hot"])
def test_invalid_measured_timings_rejected(prep, amended, change):
    fixture, approved, report, _ = amended
    manifests = {c: prep.manifest_snapshot(p, report["conditions"][c]["verification"]["archive_sha256"])
                 for c, p in fixture.archives.items()}
    stage = manifests["G_SINGLE"]["stages"]["environment"]
    if change == "failed_stage":
        stage["ok"] = False
    elif change in {"nan", "negative"}:
        stage["seconds"] = float("nan") if change == "nan" else -1
    else:
        report["conditions"]["G_SINGLE"]["estimate"] = {} if change == "missing_hot" else {"estimated_test_walltime_s": 20000}
    with pytest.raises(ValueError):
        prep.historical_estimates(fixture.root, prep.r.plan("test"), approved, report, manifests)


def test_manifest_pin_and_duplicate_guard(prep, tmp_path):
    path = tmp_path / "synthetic_final.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="SHA/size"):
        prep.manifest_snapshot(path, "a" * 64)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("MANIFEST.JSON", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        prep.manifest_snapshot(path, prep.r.digest(path))


def test_existing_record_never_overwritten_or_completed(prep, preparation):
    root, _, _, _, _, _ = preparation
    path = root / "results/v3/operational_amendment/preregistration_eta.json"
    path.write_bytes(b"Existing or interrupted immutable record")
    with pytest.raises(ValueError, match="already exists"):
        prep.prepare(root, True)
    assert path.read_bytes() == b"Existing or interrupted immutable record"
    assert not (path.parent / "approval.json").exists()
