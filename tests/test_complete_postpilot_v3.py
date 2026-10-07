"""Offline synthetic supervision checks; all process launches are replaced."""

from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows process flags required")
def test_real_hidden_cmd_executes_powershell_without_host_or_gpu(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools/colab"))
    s = importlib.import_module("complete_postpilot_v3")
    log = tmp_path / "host launch & probe.log"
    command = s.cmd_arguments(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                               "[Console]::WriteLine(12345)"])
    wrapper = 'cmd.exe /d /v:off /s /c "' + command + ' >>' + s.cmd_arguments([str(log)]) + ' 2>&1"'
    result = subprocess.run(wrapper, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, creationflags=s.HOST_FLAGS, timeout=20)
    assert result.returncode == 0 and log.read_text().strip() == "12345"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def supervisor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools/colab"))
    module = importlib.import_module("complete_postpilot_v3")
    def forbidden(*args, **kwargs):
        pytest.fail("Real subprocess forbidden")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    return module


@pytest.fixture
def scenario(tmp_path, supervisor, monkeypatch):
    s, root = supervisor, tmp_path / "workspace & receipts"
    out = root / "results/v3"
    ack_path = out / "host_ack.json"
    clock = [1000.0]
    monkeypatch.setattr(s.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(s.time, "time", lambda: clock[0])
    monkeypatch.setattr(s.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    ledger = dict(experiment_id=s.EXPERIMENT_ID, max_gpu_seconds=s.MAX_GPU_SECONDS,
                  jobs={"historical": dict(released=True, release_verified=True, actual_seconds=7,
                                            release_evidence="backend_absent")})
    write(out / "budget_ledger.json", ledger)
    host = dict(state="closed", host_lease_id="a" * 32, host_process_id=10,
                safe_to_stop_host=True, live_leases=0, pending_operators=0,
                coordinator_exit_code=0, helper_exit_code=0)
    ack = dict(state="closed", host_lease_id="a" * 32, host_process_id=10, accept_allocations=False)
    write(out / "hoststatus.json", host)
    write(ack_path, ack)
    old = {p.name: p.read_bytes() for p in (ack_path, out / "hoststatus.json")}
    launches, analyses, exports, polls, wrappers, proofs = [], [], [], [], [], []
    native = {}
    case = {"value": "success"}

    def popen(command, **kwargs):
        wrappers.append(command)
        host_command = command.split('/c "', 1)[1].rsplit(' >>', 1)[0]
        command = re.findall(r'"([^"]*)"', host_command)
        phase = command[command.index("-PostPilotPhase") + 1]
        wrapper_pid = 20 + len(launches)
        pid = 120 + len(launches)
        lease = ("b" if phase == "test" else "c") * 32
        launched = clock[0]
        native.update(ProcessId=pid, ParentProcessId=wrapper_pid, Name="powershell.exe",
                      CommandLine=host_command,
                      started_utc=datetime.fromtimestamp(launched, timezone.utc).isoformat())
        launches.append((command, kwargs))
        def publish():
            stamp = datetime.fromtimestamp(launched, timezone.utc).isoformat()
            fresh_ack = dict(ack, host_lease_id=lease, host_process_id=pid,
                             issued_utc=stamp, host_process_started_utc=stamp, wsl_distro="Ubuntu")
            fresh_host = dict(host, host_lease_id=lease, host_process_id=pid)
            state = dict(series="v3", phase=phase, outcome="completed",
                         host_identity=dict(host_lease_id=lease, wsl_distro="Ubuntu",
                                            accept_allocations=True, verified_utc=stamp),
                         verified_jobs={i["key"]: {"ok": True} for i in s.plan(phase)})
            if case["value"] == "stale_state":
                state["host_identity"]["host_lease_id"] = "a" * 32
            elif case["value"] == "unsafe":
                fresh_host["safe_to_stop_host"] = False
            elif case["value"] == "live":
                ledger["jobs"]["historical"]["released"] = False
                write(out / "budget_ledger.json", ledger)
            elif case["value"] == "incomplete":
                state["verified_jobs"].pop(next(iter(state["verified_jobs"])))
            elif case["value"] == "blocked":
                state["outcome"] = "blocked_execution"
            elif case["value"] == "wrong_pid":
                fresh_ack["host_process_id"] = 999
            elif case["value"] == "wrong_host_pid":
                fresh_host["host_process_id"] = wrapper_pid
            elif case["value"] == "old_lease":
                fresh_ack["host_lease_id"] = fresh_host["host_lease_id"] = "a" * 32
            elif case["value"] == "invalid_lease":
                fresh_ack["host_lease_id"] = fresh_host["host_lease_id"] = "unverified"
            elif case["value"] == "wrong_start":
                fresh_ack["host_process_started_utc"] = datetime.fromtimestamp(launched + 0.01, timezone.utc).isoformat()
            elif case["value"] == "future_ack":
                fresh_ack["issued_utc"] = datetime.fromtimestamp(clock[0] + 20, timezone.utc).isoformat()
            elif case["value"] == "active_success" and polls.count(phase) == 1:
                fresh_ack.update(state="active", acknowledged=True, keep_host_awake=True,
                                 keep_wsl_alive=True, accept_allocations=True)
                fresh_host["state"] = "active"
            write(ack_path, fresh_ack)
            write(out / "hoststatus.json", fresh_host)
            write(out / ("coordinator_postpilot_" + phase + "_state.json"), state)
        def poll():
            polls.append(phase)
            if case["value"] == "stale_exit":
                return 0
            if case["value"] == "timeout":
                clock[0] += s.WALL_SECONDS
            elif case["value"] == "readiness_timeout":
                pass
            else:
                publish()
            return None
        # First read intentionally still sees the historical closed receipt.
        return SimpleNamespace(pid=wrapper_pid, poll=poll, wait=lambda timeout: 0)

    verification = dict(ok=True, archive_sha256="synthetic-only")
    def run(command, **kwargs):
        if command[0] == "powershell.exe":
            proofs.append((command, kwargs))
            if "AND Name = 'cmd.exe'" in command[-1]:
                candidate = dict(ProcessId=native["ParentProcessId"], Name="cmd.exe",
                                 CommandLine=wrappers[0], started_utc=native["started_utc"])
                if case["value"] == "wrong_wrapper":
                    candidate["CommandLine"] = "cmd.exe /c unrelated.cmd"
                return SimpleNamespace(returncode=0, stdout=json.dumps([candidate]))
            candidate = dict(native)
            if case["value"] == "wrong_parent":
                candidate["ParentProcessId"] = 999
            elif case["value"] == "stale_native":
                candidate["started_utc"] = datetime.fromtimestamp(clock[0] - 10, timezone.utc).isoformat()
            elif case["value"] == "future_native":
                candidate["started_utc"] = datetime.fromtimestamp(clock[0] + 10, timezone.utc).isoformat()
            elif case["value"] == "wrong_command":
                candidate["CommandLine"] = "powershell.exe -File unrelated.ps1"
            elif case["value"] == "wrapper_identity":
                candidate["ProcessId"] = candidate["ParentProcessId"]
            elif case["value"] == "unquoted_executable":
                candidate["CommandLine"] = 'powershell.exe' + candidate["CommandLine"][len('"powershell.exe"'):]
            elif case["value"] == "cmd_whitespace":
                candidate["CommandLine"] = candidate["CommandLine"].replace('"powershell.exe" ', '"powershell.exe"  ') + ' '
            elif case["value"] == "cim_timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            elif case["value"] == "cim_failed":
                raise subprocess.CalledProcessError(1, command)
            candidates = [] if case["value"] == "no_native" else [candidate]
            if case["value"] == "ambiguous_native":
                candidates.append(candidate)
            return SimpleNamespace(returncode=0, stdout=json.dumps(candidates))
        analyses.append((command, kwargs))
        if case["value"] == "analysis_timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        write(out / "analysis/ANALYSIS/verification.json", verification)
        write(out / "analysis/ANALYSIS/local_analysis_receipt.json",
              dict(execution="local_cpu_papermill", allocation_attempted=False, gpu_seconds=0,
                   result=verification if case["value"] != "fake_analysis" else dict(ok=False)))
        return SimpleNamespace(returncode=0)
    def export(root_arg, state):
        exports.append(state)
        assert root_arg == root
        assert state == {"verified_jobs": {"analysis/ANALYSIS": verification}}
        return str(out / "final_analysis/resumen_es.md")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(s, "export_analysis", export)
    def complete(resume_test_wrapper=None):
        return s.complete(root, out / "initial_budget.json", ack_path,
                          out / "provided_approval.json", out / "pilot/real_report.json", "/isolated/bin/python",
                          resume_test_wrapper)
    return SimpleNamespace(**locals())


def test_order_freshness_receipt_preservation_and_real_analysis(scenario):
    f = scenario
    assert f.complete() == 0
    assert f.polls == ["test", "latency"]
    assert len(f.launches) == 2 and len(f.analyses) == len(f.exports) == 1
    for (command, options), phase in zip(f.launches, ("test", "latency")):
        assert command[command.index("-PostPilotPhase") + 1] == phase
        assert command[command.index("-Distro") + 1] == "Ubuntu"
        assert command[command.index("-Python") + 1] == "/isolated/bin/python"
        assert command[command.index("-InitialBudget") + 1] == str(f.out / "initial_budget.json")
        assert command[command.index("-Approval") + 1] == str(f.out / "provided_approval.json")
        assert "-Recovery" not in command and "-AttemptPrefix" not in command
        assert 4 < int(command[command.index("-Hours") + 1]) <= 24
        assert options["creationflags"] == 134218240 and options["close_fds"] is True
        assert options["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
        assert options["startupinfo"].wShowWindow == 0
        assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL
        wrapper = f.wrappers[("test", "latency").index(phase)]
        assert wrapper.startswith('cmd.exe /d /v:off /s /c ""powershell.exe" ')
        assert wrapper.endswith(' >>"' + str(f.out / ("completion_postpilot_" + phase + "_host.log")) + '" 2>&1"')
    assert len(f.proofs) == 2
    for command, options in f.proofs:
        assert "Get-CimInstance Win32_Process" in command[-1]
        assert "ParentProcessId = " in command[-1]
        assert options["check"] is True and 0 < options["timeout"] <= 10
    for name, content in f.old.items():
        assert (f.out / "completion_postpilot_receipts" / ("a" * 32) / name).read_bytes() == content
    assert len(list((f.out / "completion_postpilot_receipts").iterdir())) == 3
    command, options = f.analyses[0]
    assert command[0] == str(f.root / ".venv-analysis/Scripts/python.exe")
    assert command[1] == str(f.root / "tools/local_analysis_v3.py")
    assert 0 < options["timeout"] < f.s.WALL_SECONDS
    status = json.loads((f.out / "completion_postpilot_status.json").read_text())
    assert status["outcome"] == "completed"
    assert len(status["phases"]["test"]["verified_jobs"]) == 6
    assert status["phases"]["latency"]["verified_jobs"] == ["latency/LATENCY"]
    assert status["phases"]["test"]["host_process_id"] == 120
    assert status["phases"]["test"]["host_wrapper_process_id"] == 20
    assert not (f.out / "completion_status.json").exists()
    assert not (f.out / "coordinator_full_infra03_recovery_state.json").exists()


@pytest.mark.parametrize("case", ["stale_exit", "stale_state", "unsafe", "live", "incomplete",
                                  "blocked", "wrong_pid", "timeout", "readiness_timeout",
                                 "fake_analysis", "analysis_timeout", "wrong_parent", "stale_native",
                                 "wrong_command", "wrapper_identity", "no_native", "ambiguous_native",
                                 "wrong_start", "future_ack", "cim_timeout", "cim_failed",
                                 "wrong_host_pid", "old_lease", "invalid_lease", "future_native"])
def test_fail_closed_without_advancing_or_export(scenario, case):
    f = scenario
    f.case["value"] = case
    with pytest.raises((ValueError, subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        f.complete()
    assert not f.exports
    if case not in {"fake_analysis", "analysis_timeout"}:
        assert len(f.launches) == 1 and not f.analyses
    status = json.loads((f.out / "completion_postpilot_status.json").read_text())
    assert status["outcome"] == "failed" and status["needs_parent_attention"] is True


@pytest.mark.parametrize("case", ["active_success", "unquoted_executable", "cmd_whitespace"])
def test_native_identity_accepts_live_readiness_and_cmd_executable_quoting(scenario, case):
    scenario.case["value"] = case
    assert scenario.complete() == 0
    assert len(scenario.proofs) == 2


def test_cmd_arguments_quotes_shell_metacharacters_and_trailing_slashes(supervisor):
    assert supervisor.cmd_arguments(["powershell.exe", "C:\\root&name\\", "two words!"]) == (
        '"powershell.exe" "C:\\root&name\\\\" "two words!"')


@pytest.mark.parametrize("value", ['percent%PATH%', 'quote"', 'line\n', 'return\r'])
def test_cmd_arguments_rejects_expansion_and_command_injection(supervisor, value):
    with pytest.raises(ValueError, match="Unsafe CMD argument"):
        supervisor.cmd_arguments([value])


@pytest.mark.parametrize("change", [dict(released=False), dict(release_verified=False),
                                   dict(actual_seconds=float("nan")), dict(actual_seconds=True),
                                   dict(release_evidence="unverified")])
def test_live_or_invalid_initial_ledger_never_launches(scenario, change):
    f = scenario
    f.ledger["jobs"]["historical"].update(change)
    write(f.out / "budget_ledger.json", f.ledger)
    with pytest.raises(ValueError):
        f.complete()
    assert f.launches == f.analyses == f.exports == []


def test_global_windows_lock_uses_same_byte_and_releases(supervisor, tmp_path, monkeypatch):
    (tmp_path / "results/v3").mkdir(parents=True)
    events = []
    fake = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2,
                           locking=lambda fd, mode, count: events.append((mode, count)))
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    with pytest.raises(RuntimeError):
        with supervisor.completion_lock(tmp_path):
            raise RuntimeError("synthetic failure")
    assert events == [(1, 1), (2, 1)]
    fake.locking = lambda *args: (_ for _ in ()).throw(OSError("already locked"))
    with pytest.raises(OSError, match="already locked"):
        with supervisor.completion_lock(tmp_path):
            pytest.fail("Parallel supervisor entered")


def test_native_detached_cli_redirects_handles_without_running_work(supervisor, tmp_path, monkeypatch):
    root = tmp_path
    for name in ("results/v3/initial_budget.json", "results/v3/approval.json", "results/v3/pilot.json",
                 ".venv-analysis/Scripts/python.exe", "tools/local_analysis_v3.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    launches = []
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(subprocess, "Popen", popen)
    def forbidden(*args, **kwargs):
        pytest.fail("Detached parent must not run work or acquire child's lock")
    monkeypatch.setattr(supervisor, "complete", forbidden)
    monkeypatch.setattr(supervisor, "completion_lock", forbidden)
    assert supervisor.main(["--root", str(root), "--initial-budget", str(root / "results/v3/initial_budget.json"),
                            "--host-ack", str(root / "results/v3/host_ack.json"),
                            "--approval", str(root / "results/v3/approval.json"),
                            "--pilot-report", str(root / "results/v3/pilot.json"), "--detach",
                            "--resume-test-wrapper", "43116"]) == 0
    command, options = launches[0]
    assert command[0] == sys.executable and "--detach" not in command
    assert command[-2:] == ["--resume-test-wrapper", "43116"]
    assert options["creationflags"] == 520 and options["close_fds"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"].name.endswith("completion_postpilot_supervisor.log")
    assert options["stderr"] == subprocess.STDOUT


def test_native_argv_preserves_path_whitespace_quotes_and_backslashes(supervisor):
    argv = ["powershell.exe", "-File", "C:\\two  spaces\\script.ps1", "C:\\trailing path\\", 'literal"quote']
    command = subprocess.list2cmdline(argv)
    assert supervisor.native_argv(command.replace("powershell.exe ", '"powershell.exe"  ') + " ") == argv
    assert supervisor.native_argv(subprocess.list2cmdline(["powershell.exe", "C:\\two spaces\\script.ps1"])) != argv[:3]


@pytest.fixture
def resume_scenario(scenario, monkeypatch):
    f = scenario
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
               str(f.root / "tools/colab/host_v3.ps1"), "-PostPilotPhase", "test",
               "-Root", str(f.root), "-Distro", "Ubuntu", "-Python", "/isolated/bin/python",
               "-InitialBudget", str(f.out / "initial_budget.json"), "-HostAck", str(f.ack_path),
               "-Approval", str(f.out / "provided_approval.json"), "-PilotReport", str(f.out / "pilot/real_report.json"),
               "-Hours", "23"]
    wrapper = 'cmd.exe /d /v:off /s /c "' + f.s.cmd_arguments(command) + ' >>' + f.s.cmd_arguments([str(f.out / "completion_postpilot_test_host.log")]) + ' 2>&1"'
    existing = f.popen(wrapper)
    f.case["value"] = "active_success"
    existing.poll()
    f.case["value"] = "success"
    f.launches.clear()
    f.polls.clear()
    failed = dict(outcome="failed", phase="test", host_wrapper_process_id=existing.pid,
                  host_process_id=None, phases={}, error_message="Native host process identity mismatch")
    write(f.out / "completion_postpilot_status.json", failed)
    f.failed_bytes = (f.out / "completion_postpilot_status.json").read_bytes()
    f.observed = []
    f.closed_handles = []
    def observe(pid):
        f.observed.append(pid)
        return SimpleNamespace(pid=existing.pid, started=f.s.epoch(f.native["started_utc"]),
                               poll=existing.poll, wait=existing.wait,
                               close=lambda: f.closed_handles.append(pid))
    monkeypatch.setattr(f.s, "OwnedProcessObserver", observe)
    f.clock[0] += 20
    f.existing = existing
    return f


def test_resume_observes_original_test_then_only_launches_latency(resume_scenario):
    f = resume_scenario
    # A live TEST budget must not trigger the new-host released-ledger precondition.
    f.ledger["jobs"]["historical"]["released"] = False
    write(f.out / "budget_ledger.json", f.ledger)
    original_poll = f.existing.poll
    def poll():
        f.ledger["jobs"]["historical"]["released"] = True
        write(f.out / "budget_ledger.json", f.ledger)
        return original_poll()
    f.existing.poll = poll
    f.case["value"] = "cmd_whitespace"
    assert f.complete(f.existing.pid) == 0
    assert f.observed == f.closed_handles == [f.existing.pid]
    assert len(f.launches) == 1
    assert f.launches[0][0][f.launches[0][0].index("-PostPilotPhase") + 1] == "latency"
    assert len(f.analyses) == len(f.exports) == 1
    directory = f.out / "completion_postpilot_receipts" / ("b" * 32)
    assert next(directory.glob("failed_supervisor_*.json")).read_bytes() == f.failed_bytes
    repair = json.loads(next(directory.glob("resume_infra_*.json")).read_text())
    assert repair["wrapper_process_id"] == f.existing.pid and len(repair["supervisor_sha256"]) == 64


@pytest.mark.parametrize("case", ["wrong_command", "wrong_parent", "wrong_start", "wrong_wrapper", "blocked", "incomplete", "unsafe", "wrong_host_pid"])
def test_resume_mismatch_or_real_failure_never_replays_test_or_launches_latency(resume_scenario, case):
    f = resume_scenario
    f.case["value"] = case
    with pytest.raises(ValueError):
        f.complete(f.existing.pid)
    assert not f.launches and not f.analyses and not f.exports
    assert f.closed_handles == [f.existing.pid]
    status = json.loads((f.out / "completion_postpilot_status.json").read_text())
    assert status["outcome"] == "failed"
    if case == "blocked":
        assert "blocked_execution" in status["error_message"]


@pytest.mark.parametrize("change", [dict(outcome="completed"), dict(phase="latency"), dict(host_wrapper_process_id=999), dict(host_process_id=999), dict(host_lease_id="d" * 32)])
def test_resume_requires_exact_failed_receipt_without_overwriting_it(resume_scenario, change):
    f = resume_scenario
    path = f.out / "completion_postpilot_status.json"
    receipt = json.loads(path.read_text())
    receipt.update(change)
    write(path, receipt)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        f.complete(f.existing.pid)
    assert path.read_bytes() == before
    assert not f.observed and not f.launches


def test_resume_already_closed_blocked_host_reports_real_error_without_start(resume_scenario):
    f = resume_scenario
    f.case["value"] = "blocked"
    f.existing.poll()
    state_path = f.out / "coordinator_postpilot_test_state.json"
    state = json.loads(state_path.read_text())
    state["error_message"] = "actual local execution error"
    write(state_path, state)
    with pytest.raises(ValueError, match="actual local execution error"):
        f.complete(f.existing.pid)
    assert not f.observed and not f.launches and not f.analyses and not f.exports
    directory = f.out / "completion_postpilot_receipts" / ("b" * 32)
    assert next(directory.glob("failed_supervisor_*.json")).read_bytes() == f.failed_bytes


@pytest.mark.parametrize("change", [dict(host_process_started_utc="1969-12-30T00:00:00+00:00"),
                                  dict(host_lease_id="d" * 32), dict(host_process_id=999)])
def test_resume_rejects_expired_or_unmatched_ack_without_start(resume_scenario, change):
    f = resume_scenario
    ack = json.loads(f.ack_path.read_text())
    ack.update(change)
    write(f.ack_path, ack)
    with pytest.raises(ValueError):
        f.complete(f.existing.pid)
    assert not f.observed and not f.launches
    assert (f.out / "completion_postpilot_status.json").read_bytes() == f.failed_bytes


def test_read_only_observer_waits_reports_exit_and_closes_handle(supervisor, monkeypatch):
    import _winapi
    observer = supervisor.OwnedProcessObserver.__new__(supervisor.OwnedProcessObserver)
    observer.pid, observer.handle = 123, 456
    waits, closes = [], []
    def wait(handle, milliseconds):
        waits.append((handle, milliseconds))
        return _winapi.WAIT_TIMEOUT if len(waits) < 3 else 0
    monkeypatch.setattr(_winapi, "WaitForSingleObject", wait)
    monkeypatch.setattr(_winapi, "GetExitCodeProcess", lambda handle: 7)
    monkeypatch.setattr(_winapi, "CloseHandle", closes.append)
    assert observer.poll() is None
    with pytest.raises(subprocess.TimeoutExpired):
        observer.wait(1.25)
    assert observer.poll() == 7
    assert observer.wait(2) == 7
    observer.close()
    assert waits == [(456, 0), (456, 1250), (456, 0), (456, 2000)]
    assert closes == [456]


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows process handle required")
def test_real_read_only_observer_of_current_test_process(supervisor):
    observer = supervisor.OwnedProcessObserver(os.getpid())
    try:
        assert observer.pid == os.getpid() and observer.started > 0
        assert observer.poll() is None
        with pytest.raises(subprocess.TimeoutExpired):
            observer.wait(0.001)
    finally:
        observer.close()
