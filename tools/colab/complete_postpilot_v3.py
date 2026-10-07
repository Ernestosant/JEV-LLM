"""Native Windows, approval-bound test -> latency -> local CPU completion."""

import argparse
import ctypes
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from complete_v3 import read_live_json, export_analysis
from run_v3 import EXPERIMENT_ID, MAX_GPU_SECONDS, atomic_json, epoch, plan, require

WALL_SECONDS = 24 * 3600
DETACHED_FLAGS = 0x00000008 | 0x00000200
HOST_FLAGS = 0x08000000 | 0x00000200


def native_argv(command):
    require(isinstance(command, str) and bool(command), "Missing native command line")
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not pointer:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return list(pointer[:count.value])
    finally:
        kernel.LocalFree(pointer)


class OwnedProcessObserver:
    """Read-only handle pins the original process identity, even after exit."""
    def __init__(self, pid):
        import _winapi
        self.pid = pid
        self.handle = _winapi.OpenProcess(0x00100000 | 0x1000, False, pid)
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
            kernel.GetProcessTimes.restype = ctypes.c_int
            times = [ctypes.c_ulonglong() for _ in range(4)]
            if not kernel.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.started = times[0].value / 10000000 - 11644473600
        except Exception:
            self.close()
            raise

    def poll(self):
        import _winapi
        if _winapi.WaitForSingleObject(self.handle, 0) == _winapi.WAIT_TIMEOUT:
            return None
        return _winapi.GetExitCodeProcess(self.handle)

    def wait(self, timeout):
        import _winapi
        if _winapi.WaitForSingleObject(self.handle, min(int(timeout * 1000), 0xfffffffe)) == _winapi.WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired("owned CMD observer", timeout)
        return _winapi.GetExitCodeProcess(self.handle)

    def close(self):
        import _winapi
        _winapi.CloseHandle(self.handle)


def cmd_arguments(command):
    # CMD owns explicit log redirection; quote every path before shell parsing.
    arguments = []
    for value in command:
        require(not any(c in value for c in '%"\r\n'), "Unsafe CMD argument")
        quoted = subprocess.list2cmdline([value])
        if not quoted.startswith('"'):
            quoted = '"' + value + '\\' * (len(value) - len(value.rstrip('\\'))) + '"'
        arguments.append(quoted)
    return ' '.join(arguments)


@contextmanager
def completion_lock(root):
    import msvcrt
    with (root / "results/v3/completion.lock").open("a+b") as lock:
        lock.write(b"\0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def released_ledger(root):
    ledger = read_live_json(root / "results/v3/budget_ledger.json")
    require(ledger.get("experiment_id") == EXPERIMENT_ID
            and ledger.get("max_gpu_seconds") == MAX_GPU_SECONDS
            and isinstance(ledger.get("jobs"), dict) and ledger["jobs"], "Missing original budget ledger")
    for entry in ledger["jobs"].values():
        charge = entry.get("actual_seconds")
        require(entry.get("released") is True and entry.get("release_verified") is True
                and type(charge) in (int, float) and math.isfinite(charge) and charge >= 0
                and (entry.get("release_evidence") == "backend_absent" or
                     (entry.get("release_evidence") == "durable_no_allocation"
                      and entry.get("allocation_attempted") is False and charge == 0
                      and not entry.get("endpoint"))), "Unverified or live budget lease")
    require(sum(e["actual_seconds"] for e in ledger["jobs"].values()) <= MAX_GPU_SECONDS,
            "Original 25h GPU budget exceeded")


def closed(host):
    require(host.get("state") == "closed" and host.get("safe_to_stop_host") is True
            and type(host.get("live_leases")) is int and host["live_leases"] == 0
            and type(host.get("pending_operators")) is int and host["pending_operators"] == 0,
            "Host closure is not verified")


def preserve_receipts(out, ack_path):
    """Keep exact historical receipts before the host replaces its output paths."""
    host_path = ack_path.parent / "hoststatus.json"
    if not host_path.exists():
        require(not ack_path.exists(), "Host acknowledgement has no closure receipt")
        return None
    host, ack = read_live_json(host_path), read_live_json(ack_path)
    closed(host)
    lease = host.get("host_lease_id", "")
    require(len(lease) == 32 and all(c in "0123456789abcdef" for c in lease)
            and ack.get("host_lease_id") == lease and ack.get("state") == "closed"
            and ack.get("accept_allocations") is False, "Historical host identity is unconfirmed")
    directory = out / "completion_postpilot_receipts" / lease
    directory.mkdir(parents=True, exist_ok=True)
    for source in (ack_path, host_path):
        target = directory / source.name
        content = source.read_bytes()
        if target.exists():
            require(target.read_bytes() == content, "Preserved historical receipt differs")
        else:
            with target.open("xb") as stream:
                stream.write(content)
    return lease


def complete(root, initial_budget, host_ack, approval, pilot_report, python, resume_test_wrapper=None):
    out = root / "results/v3"
    deadline = time.monotonic() + WALL_SECONDS
    status_path = out / "completion_postpilot_status.json"
    status = dict(supervisor_pid=os.getpid(), outcome="starting", phases={})
    observer = None
    resume_ack = None
    may_save = resume_test_wrapper is None

    def remaining():
        seconds = deadline - time.monotonic()
        require(seconds > 0, "Post-pilot completion exceeded its 24-hour wall-clock bound")
        return seconds

    def save(outcome, **fields):
        status.update(outcome=outcome, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        atomic_json(status_path, status)

    try:
        if resume_test_wrapper is not None:
            require(type(resume_test_wrapper) is int and resume_test_wrapper > 0, "Invalid resume wrapper PID")
            failed_bytes = status_path.read_bytes()
            failed = json.loads(failed_bytes)
            require(failed.get("outcome") == "failed" and failed.get("phase") == "test"
                    and failed.get("host_wrapper_process_id") == resume_test_wrapper
                    and not failed.get("phases"), "Resume requires exact failed TEST wrapper receipt")
            resume_ack = read_live_json(host_ack)
            resume_host = read_live_json(host_ack.parent / "hoststatus.json")
            lease = resume_ack.get("host_lease_id", "")
            require(type(resume_ack.get("host_process_id")) is int
                    and resume_ack["host_process_id"] > 0
                    and resume_host.get("host_process_id") == resume_ack["host_process_id"]
                    and len(lease) == 32 and all(c in "0123456789abcdef" for c in lease)
                    and resume_host.get("host_lease_id") == lease
                    and failed.get("host_process_id") in (None, resume_ack["host_process_id"])
                    and failed.get("host_lease_id") in (None, lease), "Resume host acknowledgement identity mismatch")
            original = epoch(resume_ack["host_process_started_utc"])
            require(original <= time.time() + 1 and time.time() - original < WALL_SECONDS,
                    "Original resume host wall-clock bound expired")
            deadline = time.monotonic() + WALL_SECONDS - max(0, time.time() - original)
            directory = out / "completion_postpilot_receipts" / lease
            directory.mkdir(parents=True, exist_ok=True)
            snapshot = directory / ("failed_supervisor_" + hashlib.sha256(failed_bytes).hexdigest() + ".json")
            if snapshot.exists():
                require(snapshot.read_bytes() == failed_bytes, "Failed receipt snapshot differs")
            else:
                with snapshot.open("xb") as stream:
                    stream.write(failed_bytes)
            may_save = True
            status.update(phase="test", host_wrapper_process_id=resume_test_wrapper,
                          host_process_id=resume_ack["host_process_id"], host_lease_id=lease,
                          host_process_started_utc=resume_ack["host_process_started_utc"])
            # A distinct infrastructure receipt never replaces a parent's earlier repair receipt.
            repair = directory / ("resume_infra_" + str(os.getpid()) + "_" + str(time.time_ns()) + ".json")
            with repair.open("x", encoding="utf-8") as stream:
                json.dump(dict(supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               failed_receipt=str(snapshot), wrapper_process_id=resume_test_wrapper,
                               host_process_id=resume_ack["host_process_id"], host_lease_id=lease,
                               original_started_utc=resume_ack["host_process_started_utc"]), stream)
            if resume_host.get("state") == "closed":
                state = read_live_json(out / "coordinator_postpilot_test_state.json")
                require(state.get("outcome") == "completed",
                        "Existing TEST host closed with failure: " + json.dumps(dict(host=resume_host, coordinator=state)))
            observer = OwnedProcessObserver(resume_test_wrapper)
        for phase in ("test", "latency"):
            resuming = phase == "test" and resume_ack is not None
            if not resuming:
                released_ledger(root)
            old_lease = None if resuming else preserve_receipts(out, host_ack)
            hours = min(24, int(remaining() // 3600))
            require(resuming or (hours > 4 and hours * 3600 >= max(j["deadline_seconds"] for j in plan(phase)) + 90),
                    "Insufficient remaining host custody within total 24h bound")
            launched = epoch(resume_ack["host_process_started_utc"]) if resuming else time.time()
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
                       str(root / "tools/colab/host_v3.ps1"), "-PostPilotPhase", phase,
                       "-Root", str(root), "-Distro", "Ubuntu", "-Python", python,
                       "-InitialBudget", str(initial_budget), "-HostAck", str(host_ack),
                       "-Approval", str(approval), "-PilotReport", str(pilot_report), "-Hours", str(hours)]
            log_path = out / ("completion_postpilot_" + phase + "_host.log")
            host_command = cmd_arguments(command)
            wrapper_command = 'cmd.exe /d /v:off /s /c "' + host_command + ' >>' + cmd_arguments([str(log_path)]) + ' 2>&1"'
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            child = observer if resuming else subprocess.Popen(wrapper_command, cwd=root,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=startup, creationflags=HOST_FLAGS, close_fds=True)
            if not resuming:
                save("waiting_host", phase=phase, host_wrapper_process_id=child.pid, host_process_id=None)
            else:
                status.update(phase=phase, host_wrapper_process_id=child.pid,
                              host_process_id=resume_ack["host_process_id"], host_lease_id=resume_ack["host_lease_id"])
            lease, ready, process = None, False, None
            startup_until = min(deadline, time.monotonic() + 180)
            while True:
                remaining()
                if process is None:
                    # Native, read-only process proof, never a WSL/GPU API invocation.
                    query = ("$ErrorActionPreference='Stop'; "
                             "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
                             f"$p=@(Get-CimInstance Win32_Process -Filter \"ParentProcessId = {child.pid} "
                             "AND Name = 'powershell.exe'\" | Select-Object ProcessId, ParentProcessId, "
                             "Name, CommandLine, @{n='started_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}}); "
                             "ConvertTo-Json -InputObject $p -Compress")
                    proof = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        encoding="utf-8", timeout=min(10, remaining()), check=True, creationflags=HOST_FLAGS)
                    candidates = json.loads(proof.stdout)
                    require(isinstance(candidates, list) and len(candidates) <= 1,
                            "Ambiguous native host process identity")
                    if candidates:
                        process = candidates[0]
                        argv = native_argv(process.get("CommandLine"))
                        if resuming:
                            require(len(argv) == len(command) and argv[-2] == "-Hours"
                                    and argv[-1].isdigit() and 4 < int(argv[-1]) <= 24,
                                    "Resume original host hours mismatch")
                            command[-1] = argv[-1]
                            require(process.get("ProcessId") == resume_ack["host_process_id"]
                                    and abs(epoch(process["started_utc"]) - launched) <= 0.001
                                    and observer.started <= launched + 0.001
                                    and launched - observer.started < 180,
                                    "Resume original process birth mismatch")
                            require(failed.get("host_process_started_utc") is None
                                    or abs(epoch(failed["host_process_started_utc"]) - launched) <= 0.001,
                                    "Resume failed receipt birth mismatch")
                            wrapper_query = query.replace(
                                f"ParentProcessId = {child.pid} AND Name = 'powershell.exe'",
                                f"ProcessId = {child.pid} AND Name = 'cmd.exe'")
                            wrapper_proof = subprocess.run(
                                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", wrapper_query],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", timeout=min(10, remaining()), check=True, creationflags=HOST_FLAGS)
                            wrappers = json.loads(wrapper_proof.stdout)
                            expected_wrapper = 'cmd.exe /d /v:off /s /c "' + cmd_arguments(command) + ' >>' + cmd_arguments([str(log_path)]) + ' 2>&1"'
                            require(isinstance(wrappers, list) and len(wrappers) == 1
                                    and wrappers[0].get("ProcessId") == child.pid
                                    and wrappers[0].get("Name", "").lower() == "cmd.exe"
                                    and abs(epoch(wrappers[0]["started_utc"]) - observer.started) <= 0.001
                                    and native_argv(wrappers[0].get("CommandLine")) == native_argv(expected_wrapper),
                                    "Resume CMD wrapper identity mismatch")
                        require(type(process.get("ProcessId")) is int and process["ProcessId"] > 0
                                and process["ProcessId"] != child.pid
                                and process.get("ParentProcessId") == child.pid
                                and process.get("Name", "").lower() == "powershell.exe"
                                and argv == command
                                and launched - 1 <= epoch(process["started_utc"]) <= time.time() + 1,
                                "Native host process identity mismatch")
                        save("waiting_host", host_process_id=process["ProcessId"],
                             host_process_started_utc=process["started_utc"])
                try:
                    ack = read_live_json(host_ack)
                    host = read_live_json(host_ack.parent / "hoststatus.json")
                except FileNotFoundError:
                    ack, host = {}, {}
                if resuming:
                    require(ack.get("host_process_id") == resume_ack["host_process_id"]
                            and host.get("host_process_id") == resume_ack["host_process_id"]
                            and ack.get("host_lease_id") == resume_ack["host_lease_id"]
                            and host.get("host_lease_id") == resume_ack["host_lease_id"],
                            "Existing resume host identity changed")
                fresh = (process is not None
                         and ack.get("host_process_id") == process["ProcessId"]
                         and host.get("host_process_id") == process["ProcessId"]
                         and ack.get("host_lease_id") == host.get("host_lease_id")
                         and bool(ack.get("host_lease_id")) and ack["host_lease_id"] != old_lease)
                if fresh:
                    require(len(ack["host_lease_id"]) == 32
                            and all(c in "0123456789abcdef" for c in ack["host_lease_id"])
                            and launched - 1 <= epoch(ack["issued_utc"]) <= time.time() + 1
                            and abs(epoch(ack["host_process_started_utc"]) - epoch(process["started_utc"])) <= 0.001
                            and ack.get("wsl_distro") == "Ubuntu", "New host process identity mismatch")
                    require(lease in (None, ack["host_lease_id"])
                            and (not resuming or ack["host_lease_id"] == resume_ack["host_lease_id"]), "New host lease changed")
                    lease = ack["host_lease_id"]
                    ready = ready or (ack.get("state") == "active" and ack.get("accept_allocations") is True
                                      and ack.get("acknowledged") is True
                                     and ack.get("keep_host_awake") is True and ack.get("keep_wsl_alive") is True)
                    if host.get("state") == "closed":
                        closed(host)
                        require(ack.get("state") == "closed" and ack.get("accept_allocations") is False,
                                "Closed host acknowledgement mismatch")
                        state = read_live_json(out / ("coordinator_postpilot_" + phase + "_state.json"))
                        require(state.get("outcome") == "completed", "Post-pilot phase failed: " + json.dumps(
                            {k: state.get(k) for k in ("outcome", "error_type", "error_message")}))
                        identity = state.get("host_identity", {})
                        # A fast phase may close between polls; its coordinator records real readiness.
                        require(identity.get("host_lease_id") == lease
                                and identity.get("wsl_distro") == "Ubuntu"
                                and identity.get("accept_allocations") is True
                                and launched - 1 <= epoch(identity["verified_utc"]) <= time.time() + 1,
                                "Completed coordinator is not bound to newly ready host")
                        expected = {item["key"] for item in plan(phase)}
                        require(state.get("series") == "v3" and state.get("phase") == phase
                                and state.get("outcome") == "completed"
                                and set(state.get("verified_jobs", {})) == expected
                                and all(v.get("ok") is True for v in state["verified_jobs"].values()),
                                "Post-pilot phase is not genuinely completed/verified: " + json.dumps(state))
                        require(child.wait(timeout=remaining()) == 0
                                and host.get("coordinator_exit_code") == 0
                                and host.get("helper_exit_code") == 0, "Host process failed: " + json.dumps(host))
                        released_ledger(root)
                        status["phases"][phase] = dict(host_lease_id=lease, host_process_id=process["ProcessId"],
                                                      host_wrapper_process_id=child.pid,
                                                      verified_jobs=sorted(expected))
                        save("phase_completed")
                        break
                require(child.poll() is None, "Host exited without verified phase closure: " + json.dumps(host))
                require(ready or time.monotonic() < startup_until, "New host readiness timed out")
                time.sleep(min(5, remaining()))

        # Preserve LAT's final closure as well, without writing any coordinator/worker states.
        preserve_receipts(out, host_ack)
        released_ledger(root)
        save("local_cpu_analysis", phase="analysis")
        command = [str(root / ".venv-analysis/Scripts/python.exe"),
                   str(root / "tools/local_analysis_v3.py"), "--root", str(root),
                   "--approval", str(approval), "--pilot-report", str(pilot_report)]
        with (out / "completion_postpilot_analysis.log").open("ab") as log:
            result = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=remaining(), check=True)
        require(result.returncode == 0, "Local analysis failed")
        remaining()
        directory = out / "analysis/ANALYSIS"
        verification = read_live_json(directory / "verification.json")
        receipt = read_live_json(directory / "local_analysis_receipt.json")
        require(receipt.get("execution") == "local_cpu_papermill"
                and receipt.get("allocation_attempted") is False and receipt.get("gpu_seconds") == 0
                and verification.get("ok") is True
                and all(receipt.get("result", {}).get(k) == v for k, v in verification.items()),
                "Genuine local CPU verification/receipt missing")
        released_ledger(root)
        report = export_analysis(root, {"verified_jobs": {"analysis/ANALYSIS": verification}})
        remaining()
        save("completed", report=report, analysis_verification=verification)
        return 0
    except Exception as error:
        if may_save:
            save("failed", error_type=type(error).__name__, error_message=str(error),
                 needs_parent_attention=True)
        # Never kill a GPU host/caretaker or manufacture release evidence on failure.
        raise
    finally:
        if observer is not None:
            observer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "initial-budget", "host-ack", "approval", "pilot-report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--python", default="/root/.local/share/uv/tools/google-colab-cli/bin/python")
    parser.add_argument("--detach", action="store_true", help="Launch native detached supervisor; child takes global lock")
    parser.add_argument("--resume-test-wrapper", type=int,
                        help="Observe only this failed TEST's existing owned CMD; never relaunch TEST")
    args = parser.parse_args(argv)
    require(os.name == "nt", "This supervisor requires native Windows Python")
    for name in ("root", "initial_budget", "host_ack", "approval", "pilot_report"):
        path = getattr(args, name).absolute()
        require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                        for p in (path, *path.parents)), "Symlink/junction supervisor path forbidden")
        setattr(args, name, path.resolve())
    require(args.python.startswith("/") and not args.python.startswith("/mnt/"),
            "An isolated WSL Python absolute path is required")
    require(args.host_ack.name != "hoststatus.json", "Host acknowledgement/status collision")
    out = args.root / "results/v3"
    outputs = (args.host_ack, args.host_ack.parent / "hoststatus.json",
               out / "completion_postpilot_status.json", out / "completion.lock",
               out / "completion_postpilot_receipts")
    for path in outputs:
        require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                        for p in (path, *path.parents)), "Symlink/junction output forbidden")
    require(not any(path in outputs for path in (args.initial_budget, args.approval, args.pilot_report)),
            "Supervisor outputs must not replace authorization evidence")
    for path in (args.initial_budget, args.approval, args.pilot_report,
                 args.root / ".venv-analysis/Scripts/python.exe", args.root / "tools/local_analysis_v3.py"):
        require(path.is_file(), "Required input missing: " + str(path))
    require(args.host_ack.parent.is_dir(), "Host acknowledgement parent missing")
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in ("root", "initial_budget", "host_ack", "approval", "pilot_report", "python"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.resume_test_wrapper is not None:
            command.extend(["--resume-test-wrapper", str(args.resume_test_wrapper)])
        with (args.root / "results/v3/completion_postpilot_supervisor.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=args.root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, creationflags=DETACHED_FLAGS, close_fds=True)
        print(json.dumps(dict(supervisor_pid=child.pid, status=str(args.root / "results/v3/completion_postpilot_status.json"))))
        return 0
    with completion_lock(args.root):
        return complete(args.root, args.initial_budget, args.host_ack, args.approval, args.pilot_report,
                        args.python, args.resume_test_wrapper)


if __name__ == "__main__":
    raise SystemExit(main())
