"""Synthetic local evidence only; never launch the supervisor or a subprocess."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Subprocess launch forbidden in synthetic completion tests")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def supervisor(forbid_processes):
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab"))
        return importlib.import_module("complete_v3")
    finally:
        sys.path[:] = previous


@pytest.fixture
def evidence(tmp_path, supervisor):
    config_path = tmp_path / "config/experiment_v3.json"
    config = {"models": {"G": {"repo": "synthetic-only", "revision": "a" * 40}}}
    write_json(config_path, config)
    initial = dict(
        approved=True, scope="all_gpu_phases_conditional_technical_go",
        experiment_id=supervisor.EXPERIMENT_ID, max_gpu_hours=25,
        max_total_assignments=2, technical_go_required=True,
        config_sha256=supervisor.digest(config_path), models=config["models"],
        synthetic_fixture_only=True,
    )
    write_json(tmp_path / "results/v3/initial_budget.json", initial)
    report_path = tmp_path / "results/v3/pilot/synthetic-report.json"
    report = dict(
        go=True, errors=[], budget_approval=False,
        conditions={condition: {"verified": True} for condition in supervisor.CONDITIONS},
        aggregate={"complete": True, "observed_cases": 600},
        meta={"generated_utc": "2026-01-01T00:00:00+00:00"},
    )
    write_json(report_path, report)
    state = dict(
        pilot_report=str(report_path), pilot_report_sha256=supervisor.digest(report_path),
        technical_go=True, budget_ready=True,
        remaining_estimates={"synthetic/test": 70000, "synthetic/latency": 10000},
    )
    ledger = dict(
        experiment_id=supervisor.EXPERIMENT_ID, max_gpu_seconds=supervisor.MAX_GPU_SECONDS,
        jobs={"synthetic/pilot": dict(released=True, release_verified=True, actual_seconds=10000)},
    )
    return tmp_path, initial, report_path, report, state, ledger


def test_confirm_binds_actual_canonical_authorization_and_preserves_confirmation(evidence, supervisor):
    root, initial, report_path, report, state, ledger = evidence
    before = datetime.now(timezone.utc)
    approval, actual_report = supervisor.confirm(root, state, ledger)
    after = datetime.now(timezone.utc)
    value = json.loads(approval.read_text(encoding="utf-8"))
    canonical = json.dumps(initial, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert value["initial_authorization_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert value["initial_authorization_sha256"] != supervisor.digest(root / "results/v3/initial_budget.json")
    assert actual_report == report_path.resolve()
    assert approval == root / "results/v3/confirmatory_approval_infra03.json"
    assert value["config_sha256"] == initial["config_sha256"]
    assert value["pilot_report_sha256"] == supervisor.digest(report_path)
    approved = datetime.fromisoformat(value["approved_utc"])
    assert before <= approved <= after
    assert approved.utcoffset().total_seconds() == 0
    assert approved >= datetime.fromisoformat(report["meta"]["generated_utc"])
    assert value["approved"] is True and value["scope"] == "test_and_latency"
    assert value["max_gpu_hours"] == 25
    assert value["no_additional_allowance_after_pilot"] is True
    assert value["approved_by"] == "user (existing conditional all-phase authorization)"
    assert "No new consent or allowance" in value["authorization_source"]
    original = approval.read_bytes()
    supervisor.confirm(root, state, ledger)
    assert approval.read_bytes() == original
    value["max_gpu_hours"] = 26
    write_json(approval, value)
    changed = approval.read_bytes()
    with pytest.raises(ValueError, match="Existing confirmation changed"):
        supervisor.confirm(root, state, ledger)
    assert approval.read_bytes() == changed


@pytest.mark.parametrize("error_type", [PermissionError, FileNotFoundError])
@pytest.mark.parametrize("denials", [1, 4, 5])
def test_live_json_retries_windows_replacement_without_stale_fallback(supervisor, monkeypatch, tmp_path, denials, error_type):
    path = tmp_path / "host_ack.json"
    write_json(path, {"heartbeat": "old"})
    original_read = Path.read_text
    calls, sleeps = [], []
    failure = error_type("Synthetic Windows replacement")

    def read(actual, *, encoding):
        assert actual == path
        assert encoding == "utf-8"
        calls.append(actual)
        if len(calls) <= denials:
            raise failure
        return original_read(actual, encoding=encoding)

    def sleep(seconds):
        sleeps.append(seconds)
        write_json(path, {"heartbeat": "fresh"})

    def forbidden(*args, **kwargs):
        pytest.fail("Live JSON must not use a stat precheck or inherited read_json")

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(supervisor, "read_json", forbidden)
    monkeypatch.setattr(supervisor.time, "sleep", sleep)
    if denials == 5:
        with pytest.raises(error_type) as caught:
            supervisor.read_live_json(str(path))
        assert caught.value is failure
    else:
        assert supervisor.read_live_json(str(path)) == {"heartbeat": "fresh"}
    assert len(calls) == min(denials + 1, 5)
    assert sleeps == [0.2] * min(denials, 4)


def test_live_json_real_missing_file_fails_after_bounded_retries(supervisor, monkeypatch, tmp_path):
    path = tmp_path / "missing.json"
    sleeps = []
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)
    with pytest.raises(FileNotFoundError) as caught:
        supervisor.read_live_json(path)
    assert caught.value.filename == str(path)
    assert sleeps == [0.2] * 4


@pytest.mark.parametrize("failure", ["invalid_json", "other_os_error"])
def test_live_json_does_not_retry_or_hide_other_errors(supervisor, monkeypatch, tmp_path, failure):
    path = tmp_path / "host_ack.json"
    path.write_text("{invalid", encoding="utf-8")
    sleeps, calls = [], []
    original_read = Path.read_text

    def read(actual, *, encoding):
        calls.append(actual)
        if failure == "other_os_error":
            raise OSError("Synthetic non-retryable failure")
        return original_read(actual, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)
    with pytest.raises(json.JSONDecodeError if failure == "invalid_json" else OSError):
        supervisor.read_live_json(path)
    assert calls == [path] and sleeps == []


@pytest.mark.parametrize("case", [
    "success", "backend_first", "backend_second", "local_first", "local_second", "owned_endpoint",
])
def test_reconcile_unassigned_smoke_requires_two_empty_observations(supervisor, monkeypatch, tmp_path, case):
    run_v3 = sys.modules["run_v3"]
    root = tmp_path.resolve()
    directory = root / "results/v3/smoke/synthetic"
    ack = root / "synthetic-host-ack.json"
    state = dict(
        job={"phase": "smoke", "session": "synthetic-session"}, run_id="synthetic-smoke",
        status="failed", allocation_attempted=True, before_endpoints=[], owned_endpoint=None,
        remote_startup_ack=False, completed_execution=False, deadline_epoch=1000,
        operator_lock="synthetic-operator.lock", error="Colab503 uncertain allocation",
    )
    entry = dict(session=state["job"]["session"], output=str(directory), endpoint=None,
                 released=False, release_verified=False, actual_seconds=12)
    ledger = {"jobs": {state["run_id"]: entry}}
    observations = [{"assignments": [], "local_endpoint": None} for _ in range(2)]
    if case.startswith("backend"):
        observations[case.endswith("second")]["assignments"] = [
            {"endpoint": "unknown-synthetic-endpoint", "accelerator": "synthetic"},
        ]
    elif case.startswith("local"):
        observations[case.endswith("second")]["local_endpoint"] = "synthetic-local-endpoint"
    elif case == "owned_endpoint":
        state["owned_endpoint"] = "synthetic-owned-endpoint"
    original_state, original_ledger = deepcopy(state), deepcopy(ledger)
    events, writes, saves, charges, ledger_reads = [], [], [], [], []
    actual = 317.25

    def host_gate(actual_ack, actual_root, *, cleanup):
        assert (actual_ack, actual_root, cleanup) == (ack, root, True)

    def status_gate(actual_root, actual_directory, item):
        assert (actual_root, actual_directory) == (root, directory)
        assert item is state["job"]

    def read_json(path):
        assert path == directory / "status.json"
        return state

    def identity(backend):
        assert (backend.out, backend.session) == (directory, state["job"]["session"])
        observed = observations[len([event for event in events if event[0] == "identity"])]
        events.append(("identity", deepcopy(observed)))
        return observed

    def budget_ledger(actual_root):
        assert actual_root == root
        ledger_reads.append(actual_root)
        return ledger

    def lease_elapsed(actual_entry):
        assert actual_entry is entry
        charges.append(actual_entry)
        return actual

    def save(operator, status, **fields):
        assert (operator.root, operator.out) == (root, directory)
        saves.append((status, deepcopy(fields)))
        operator.state.update(status=status, **fields)

    def forbidden(*args, **kwargs):
        pytest.fail("Reconciliation must not release endpoints or invoke CLI operations")

    monkeypatch.setattr(run_v3, "host_gate", host_gate)
    monkeypatch.setattr(run_v3, "status_gate", status_gate)
    monkeypatch.setattr(run_v3, "flock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_v3, "budget_ledger", budget_ledger)
    monkeypatch.setattr(run_v3, "lease_elapsed", lease_elapsed)
    monkeypatch.setattr(run_v3.Backend, "identity", identity)
    monkeypatch.setattr(run_v3.Backend, "call", forbidden)
    monkeypatch.setattr(run_v3.Operator, "release", forbidden)
    monkeypatch.setattr(run_v3.Operator, "save", save)
    monkeypatch.setattr(supervisor, "read_json", read_json)
    monkeypatch.setattr(supervisor, "atomic_json", lambda path, value: writes.append((path, deepcopy(value))))
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    if case != "success":
        message = "Not an unassigned diagnostic startup failure" if case == "owned_endpoint" else "Backend is not empty"
        with pytest.raises(ValueError, match=message):
            supervisor.reconcile_unassigned_smoke(root, directory, ack)
        assert writes == saves == charges == ledger_reads == []
        assert state == original_state and ledger == original_ledger
        assert events == ([] if case == "owned_endpoint" else
                          [("identity", observations[0])] if case.endswith("first") else
                          [("identity", observations[0]), ("sleep", 5), ("identity", observations[1])])
        return

    result = supervisor.reconcile_unassigned_smoke(root, directory, ack)
    assert events == [("identity", observations[0]), ("sleep", 5), ("identity", observations[1])]
    assert result == dict(released=True, charged_seconds_upper_bound=actual, backend_observations=2)
    assert ledger_reads == [root] and charges == [entry]
    assert writes == [(root / "results/v3/budget_ledger.json", ledger)]
    assert entry["released"] is True and entry["release_verified"] is True
    assert entry["actual_seconds"] == entry["observed_seconds"] == actual
    assert entry["release_evidence"] == "backend_absent" and entry["unassigned_startup_reconciled"] is True
    assert saves == [("failed", dict(released=True, release_evidence="backend_absent",
                                    observed_gpu_seconds_upper_bound=actual, unassigned_startup_reconciled=True))]
    assert state["status"] == "failed" and state["released"] is True
    assert state["completed_execution"] is False
    assert not directory.exists()


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("outcome", ["blocked_budget_approval", "blocked_execution"])
def test_complete_orchestrates_closed_host_handoff(evidence, supervisor, monkeypatch, outcome, recovery):
    root, initial, report_path, report, state, ledger = evidence
    state.update(outcome=outcome, verified_jobs={})
    completed = dict(state, outcome="completed", verified_jobs={
        f"synthetic/job/{i}": {"ok": True} for i in range(21)
    })
    states = iter([state, completed])
    host = dict(state="closed", safe_to_stop_host=True, live_leases=0, pending_operators=0)
    out = root / "results/v3"
    approval = out / "confirmatory_approval_infra03.json"
    summary = str(out / "final_analysis/resumen_es.md")
    confirmations, exports, launches = [], [], []

    def read_json(path):
        if path == out / ("coordinator_full_infra03" + ("_recovery" if recovery else "") + "_state.json"):
            return next(states)
        if path == out / "hoststatus.json":
            return host
        assert path == out / "budget_ledger.json"
        return ledger

    def confirm(actual_root, actual_state, actual_ledger):
        assert actual_root == root
        assert actual_state is state and actual_ledger is ledger
        assert actual_state["technical_go"] is True and actual_state["budget_ready"] is True
        assert all(entry["released"] and entry["release_verified"]
                   for entry in actual_ledger["jobs"].values())
        confirmations.append(actual_state)
        return approval, report_path

    def export_analysis(actual_root, actual_state):
        assert actual_root == root and actual_state is completed
        assert len(actual_state["verified_jobs"]) == 21
        exports.append(actual_state)
        return summary

    class Child:
        def poll(self):
            return None

    def popen(command, **kwargs):
        launches.append(command)
        assert kwargs["cwd"] == root
        assert kwargs["stdout"].name == str(out / "completion_resume_host.log")
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["creationflags"] == subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        return Child()

    ticks = iter(range(10))
    monkeypatch.setattr(supervisor, "read_live_json", read_json)
    monkeypatch.setattr(supervisor, "confirm", confirm)
    monkeypatch.setattr(supervisor, "export_analysis", export_analysis)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(ticks))

    if outcome == "blocked_execution":
        with pytest.raises(ValueError, match="Execution stopped: blocked_execution"):
            supervisor.complete(root, recovery=recovery)
        assert launches == confirmations == exports == []
        return

    assert supervisor.complete(root, recovery=recovery) == 0
    assert confirmations == [state] and exports == [completed]
    assert launches == [[
        "powershell.exe", "-NoProfile", "-NonInteractive", "-File",
        str(root / "tools/colab/host_v3.ps1"), "-Root", str(root),
        "-Python", "/root/.local/share/uv/tools/google-colab-cli/bin/python",
        "-InitialBudget", str(out / "initial_budget.json"),
        "-HostAck", str(out / "host_ack.json"), "-AttemptPrefix", "infra03",
        "-Approval", str(approval), "-PilotReport", str(report_path), "-Hours", "24",
    ] + (["-Recovery"] if recovery else [])]
    status = json.loads((out / "completion_status.json").read_text(encoding="utf-8"))
    assert status["outcome"] == "completed" and status["report"] == summary
    assert status["verified_jobs"] == list(completed["verified_jobs"])


@pytest.mark.parametrize("case", [
    "consent", "scope", "config", "models", "report_hash", "report_source",
    "technical_go", "budget_ready", "go", "errors", "budget_approval",
    "incomplete", "cases", "missing_condition", "unverified", "future_report",
    "released", "release_verified", "ledger_experiment", "ledger_budget", "expensive",
])
def test_confirm_rejects_invalid_evidence_without_writing(evidence, supervisor, case):
    root, initial, report_path, report, state, ledger = evidence
    if case in {"consent", "scope", "models"}:
        initial[{"consent": "approved", "scope": "scope", "models": "models"}[case]] = {
            "consent": False, "scope": "pilot_only", "models": {},
        }[case]
        write_json(root / "results/v3/initial_budget.json", initial)
    elif case == "config":
        write_json(root / "config/experiment_v3.json", {"models": initial["models"], "changed": True})
    elif case == "report_hash":
        state["pilot_report_sha256"] = "0" * 64
    elif case == "report_source":
        report_path = root / "results/v3/not-pilot.json"
        write_json(report_path, report)
        state.update(pilot_report=str(report_path), pilot_report_sha256=supervisor.digest(report_path))
    elif case in {"technical_go", "budget_ready"}:
        state[case] = False
    elif case in {"released", "release_verified"}:
        ledger["jobs"]["synthetic/pilot"][case] = False
    elif case == "ledger_experiment":
        ledger["experiment_id"] = "synthetic-wrong-experiment"
    elif case == "ledger_budget":
        ledger["max_gpu_seconds"] += 1
    elif case == "expensive":
        state["remaining_estimates"]["synthetic/test"] += 1
    else:
        if case in {"go", "errors", "budget_approval"}:
            report[case] = {"go": False, "errors": ["synthetic failure"], "budget_approval": True}[case]
        elif case == "incomplete":
            report["aggregate"]["complete"] = False
        elif case == "cases":
            report["aggregate"]["observed_cases"] = 599
        elif case == "missing_condition":
            report["conditions"].pop(supervisor.CONDITIONS[-1])
        elif case == "unverified":
            report["conditions"][supervisor.CONDITIONS[0]]["verified"] = False
        elif case == "future_report":
            report["meta"]["generated_utc"] = "2099-01-01T00:00:00+00:00"
        write_json(report_path, report)
        state["pilot_report_sha256"] = supervisor.digest(report_path)
    with pytest.raises(ValueError):
        supervisor.confirm(root, state, ledger)
    assert not (root / "results/v3/confirmatory_approval_infra03.json").exists()


@pytest.fixture
def analysis(tmp_path, supervisor):
    report = dict(
        meta={"n_physical_cases": 6000, "n_test_problems": 500},
        parser_audit={"n_disagreements": 0},
        primary={
            "quality": {"acc_diff_pp": 2.5, "ci97_5_bonferroni": [1.0, 4.0]},
            "latency": {"speedup": 1.25, "ci97_5_bonferroni": [1.1, 1.4]},
        },
        table=[{"condition": "SYNTHETIC_ONLY", "n_cases": 6000,
                "n_problems": 500, "accuracy_pct": 75.0}],
        declarations={"synthetic_fixture_only": True, "not_actual_results": True},
    )
    archive = tmp_path / "results/v3/analysis/ANALYSIS/artifacts/analysis_confirmatory_v3.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("analysis.json", json.dumps(report))
        stream.writestr("report.md", "# Synthetic report only\n")
    state = {"verified_jobs": {"analysis/ANALYSIS": {
        "ok": True, "archive_sha256": supervisor.digest(archive),
    }}}
    return tmp_path, archive, report, state


def test_export_analysis_generates_report_and_spanish_summary(analysis, supervisor):
    root, archive, report, state = analysis
    summary = Path(supervisor.export_analysis(root, state))
    destination = root / "results/v3/final_analysis"
    assert summary == destination / "resumen_es.md"
    assert json.loads((destination / "analysis.json").read_text()) == report
    assert (destination / "report.md").read_text() == "# Synthetic report only\n"
    text = summary.read_text(encoding="utf-8")
    assert "500 problemas de test; 6000 inferencias" in text
    assert "2.5 puntos porcentuales" in text and "[1.0, 4.0]" in text
    assert "Speedup geometrico pareado: 1.25" in text and "[1.1, 1.4]" in text
    assert "| SYNTHETIC_ONLY | 6000 | 500 | 75 |" in text
    assert '"synthetic_fixture_only": true' in text
    assert '"not_actual_results": true' in text
    original = summary.read_bytes()
    assert supervisor.export_analysis(root, state) == str(summary)
    assert summary.read_bytes() == original


@pytest.mark.parametrize("case", ["unsafe", "hash", "unverified", "cases", "problems", "parser"])
def test_export_analysis_rejects_invalid_archive_before_export(analysis, supervisor, case):
    root, archive, report, state = analysis
    verification = state["verified_jobs"]["analysis/ANALYSIS"]
    if case == "hash":
        verification["archive_sha256"] = "0" * 64
    elif case == "unverified":
        verification["ok"] = False
    else:
        if case == "cases":
            report["meta"]["n_physical_cases"] = 5999
        elif case == "problems":
            report["meta"]["n_test_problems"] = 499
        elif case == "parser":
            report["parser_audit"]["n_disagreements"] = 1
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("analysis.json", json.dumps(report))
            stream.writestr("report.md", "Synthetic only")
            if case == "unsafe":
                stream.writestr("../escaped.txt", "Synthetic unsafe member")
        verification["archive_sha256"] = supervisor.digest(archive)
    with pytest.raises(ValueError, match={
        "unsafe": "Unsafe analysis export path", "hash": "Analysis archive verification changed",
        "unverified": "Analysis archive verification changed",
        "cases": "Incomplete or disputed analysis", "problems": "Incomplete or disputed analysis",
        "parser": "Incomplete or disputed analysis",
    }[case]):
        supervisor.export_analysis(root, state)
    assert not (root / "results/v3/final_analysis").exists()
    assert not (root / "results/v3/escaped.txt").exists()
