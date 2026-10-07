"""PowerShell AST and simulated custody tests. No native power/CIM/WSL/Colab calls."""

import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from contextlib import nullcontext

import pytest
from test_colab_v3 import m, operator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/colab/host_v3.ps1"


def ps_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(source):
    exe = shutil.which("powershell.exe")
    if not exe:
        pytest.skip("Windows PowerShell 5.1 required for AST/simulated call-flow tests")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def definitions():
    return f"""
$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile({ps_string(SCRIPT)},[ref]$tokens,[ref]$errors)
if ($errors.Count) {{ throw ($errors | ConvertTo-Json) }}
foreach ($node in $ast.EndBlock.Statements) {{
    if ($node -is [Management.Automation.Language.FunctionDefinitionAst]) {{ Invoke-Expression $node.Extent.Text }}
}}
$script:V3HostScript={ps_string(SCRIPT)}
"""


def test_powershell_ast_and_native_argument_roundtrip():
    report = powershell(definitions() + r'''
$value=Join-NativeArguments @('C:\root with spaces\host.ps1', 'quote"inside', 'C:\trailing\', "apostrophe's path", '')
@{errors=$errors.Count; arguments=$value; functions=@($ast.EndBlock.Statements | Where-Object {$_ -is [Management.Automation.Language.FunctionDefinitionAst]}).Count} | ConvertTo-Json -Compress
''')
    assert report["errors"] == 0 and report["functions"] < 15
    assert report["arguments"].startswith('"C:\\root with spaces\\host.ps1"')
    assert 'quote\\\"inside' in report["arguments"]
    assert report["arguments"].endswith('""')
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Set-Content" not in text and "powercfg" not in text.lower()
    assert "--shutdown" not in text and "--terminate" not in text and "Stop-Process" not in text


def test_atomic_json_has_no_bom_and_old_or_new_complete_bytes(tmp_path):
    path = tmp_path / "host ack.json"
    report = powershell(definitions() + f"""
Write-HostJson {ps_string(path)} @{{acknowledged=$true; text='first'}}
$old=[IO.File]::ReadAllBytes({ps_string(path)})
Write-HostJson {ps_string(path)} @{{acknowledged=$false; text='second'}}
@{{old=[Convert]::ToBase64String($old); current=[Convert]::ToBase64String([IO.File]::ReadAllBytes({ps_string(path)}))}} | ConvertTo-Json -Compress
""")
    old, current = (base64.b64decode(report[k]) for k in ("old", "current"))
    assert not old.startswith(b"\xef\xbb\xbf") and not current.startswith(b"\xef\xbb\xbf")
    assert json.loads(old.decode("utf-8"))["acknowledged"] is True
    assert json.loads(current.decode("utf-8"))["acknowledged"] is False
    assert not list(tmp_path.glob("*.tmp"))


def test_four_hour_allocation_helper_is_rejected_before_any_native_call():
    result = powershell(definitions() + r'''
function Set-HostPower { throw 'Native power must not run' }
function Get-HostIdentity { throw 'Native CIM must not run' }
try { [void](Invoke-V3Host 'missing' 'missing' 'C:\missing' '' '' '' 'python3' 4 $false); throw 'Expected validation failure' }
catch { @{message=$_.Exception.Message} | ConvertTo-Json -Compress }
''')
    assert "more than four hours" in result["message"]


def test_postpilot_route_uses_explicit_phase_and_requires_approval(tmp_path):
    root = tmp_path / "project"
    (root / "tools/colab").mkdir(parents=True)
    (root / "tools/colab/run_v3.py").write_text("# fixture\n")
    initial = root / "initial.json"
    initial.write_text("{}")
    result = powershell(definitions() + f'''
function Set-HostPower {{ throw 'Native power must not run' }}
try {{ [void](Invoke-V3Host {ps_string(initial)} {ps_string(root / 'ack.json')} {ps_string(root)} '' '' '' 'python3' 24 $false '' $false 'test'); throw 'Expected failure' }}
catch {{ @{{message=$_.Exception.Message}} | ConvertTo-Json -Compress }}
''')
    assert "Post-pilot requires" in result["message"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'tools\\colab\\postpilot_v3.py' in source
    assert '"--phase", $phase' in source
    assert '$phase = if ($PostPilotPhase) { $PostPilotPhase } else { "full" }' in source


@pytest.mark.parametrize("mode", ["closed", "pending", "live", "offline", "power-zero", "power-unknown", "wrong-distro", "bad-evidence", "reporting-failure"])
def test_helper_callflow_never_executes_native_host_or_allocates(tmp_path, mode):
    root = tmp_path / "root with spaces and 'quote"
    (root / "tools/colab").mkdir(parents=True)
    (root / "tools/colab/run_v3.py").write_text("# synthetic test-only coordinator path\n")
    out = root / "results/v3"
    out.mkdir(parents=True)
    initial = out / "initial-fixture.json"
    initial.write_text('{"synthetic_fixture_only":true}')
    ack_path = out / "host_ack.json"
    ledger = out / "budget_ledger.json"
    ledger.write_text(json.dumps(dict(experiment_id="jev-llm-v3-20261003", max_gpu_seconds=90000, jobs={"test-only":
        dict(released=True, release_verified=True, actual_seconds=123, allocation_attempted=True,
             release_evidence="durable_no_allocation" if mode == "bad-evidence" else "backend_absent")})))
    source = definitions() + f"""
$script:Mode={ps_string(mode)}; $script:Events=[Collections.Generic.List[string]]::new(); $script:Clock=0
$script:Root={ps_string(root)}; $script:Ledger={ps_string(ledger)}; $script:AckPath={ps_string(ack_path)}
$script:RealWriter=${{function:Write-HostJson}}
function Write-HostJson([string]$Path,$Value) {{
    if ($script:Mode -eq 'reporting-failure' -and $Path -eq $script:AckPath -and $Value.state -eq 'draining') {{ throw [IO.IOException]::new('Synthetic disk failure') }}
    & $script:RealWriter $Path $Value
}}
function Add-Type {{ throw 'Native Add-Type forbidden in simulated tests' }}
function Start-Process {{ throw 'Native process launch forbidden in simulated tests' }}
function Get-CimInstance {{ throw 'Actual CIM query forbidden in simulated tests' }}
function Get-HostIdentity {{ return @{{host_id='12345678-1234-1234-1234-123456789abc'; host_process_started_utc='2026-10-04T00:00:00.0000000Z'}} }}
function Get-HostUtc {{ return [DateTime]::Parse('2026-10-04T00:00:00Z').ToUniversalTime() }}
function Get-HostMonotonic {{ return $script:Clock }}
function Start-Sleep {{ $script:Clock += 3600 }}
function Set-HostPower([uint32]$Flags) {{
    $script:Events.Add('power:'+ $Flags)
    if ($script:Mode -eq 'power-zero') {{ return 0 }}
    if ($script:Mode -eq 'power-unknown') {{ return 'UNKNOWN' }}
    return [uint32]2147483648
}}
function Read-HostWslJson([string[]]$Arguments,[string]$Directory,[string]$Prefix,[int]$TimeoutSeconds=30,[scriptblock]$Heartbeat) {{
    if ($Arguments -contains '--reconcile-budget') {{
        $script:Events.Add('reconcile')
        $a=[IO.File]::ReadAllText($script:AckPath) | ConvertFrom-Json
        if ($a.accept_allocations) {{ throw 'Admission must drain before reconciliation' }}
        if ($script:Mode -eq 'offline') {{ throw 'Synthetic backend offline' }}
        return [pscustomobject]@{{schema_version=1; host_lease_id=$a.host_lease_id; experiment_id='jev-llm-v3-20261003';
            max_gpu_seconds=90000; live_leases=$(if ($script:Mode -eq 'live') {{1}} else {{0}}); pending_operators=$(if ($script:Mode -eq 'pending') {{1}} else {{0}});
            safe_to_stop_host=$($script:Mode -notin @('pending','live')); allocation_permitted=$false;
            ledger=(Convert-ToWslPath $script:Ledger); ledger_sha256=(Get-FileHash $script:Ledger -Algorithm SHA256).Hash.ToLowerInvariant()}}
    }}
    $script:Events.Add('probe')
    return [pscustomobject]@{{wsl_distro=$(if ($script:Mode -eq 'wrong-distro') {{'foreign'}} else {{'synthetic-distro'}})}}
}}
function Start-HostWsl([string[]]$Arguments,[string]$Mode,[string]$Directory,[string]$Prefix) {{
    $script:Events.Add('start:'+ $Mode)
    if ($Mode -eq 'caretaker') {{
        $note=$Arguments[$Arguments.Count-4] -replace '^/mnt/([a-z])/', '$1:/'
        Write-HostJson $note @{{host_lease_id=$Arguments[$Arguments.Count-2]; pid=77; starttime='1234'; wsl_distro='synthetic-distro'; deadline_monotonic=18000; boot_id='synthetic-boot'}}
    }}
    $p=[pscustomobject]@{{Id=42; HasExited=$($Mode -eq 'coordinator'); ExitCode=1}}
    return @{{process=$p; output=$Prefix+'.out.log'; errors=$Prefix+'.err.log'; started_utc='2026-10-04T00:00:00.0000000Z'}}
}}
function Stop-HostCaretaker($Caretaker,[string]$StopPath,[string]$LeaseId) {{ $script:Events.Add('stop:caretaker'); return $true }}
$code=Invoke-V3Host {ps_string(initial)} {ps_string(ack_path)} {ps_string(root)} '' '' 'synthetic-distro' 'python3' 5 $false
@{{code=$code; clock=$script:Clock; events=@($script:Events); ack=([IO.File]::ReadAllText({ps_string(ack_path)}) | ConvertFrom-Json);
   status=([IO.File]::ReadAllText({ps_string(out / 'hoststatus.json')}) | ConvertFrom-Json)}} | ConvertTo-Json -Depth 12 -Compress
"""
    result = powershell(source)
    events = result["events"]
    assert result["ack"]["acknowledged"] is False
    assert result["ack"]["keep_host_awake"] is False and result["ack"]["keep_wsl_alive"] is False
    assert result["ack"]["issued_utc"] == "2026-10-04T00:00:00.0000000Z"
    assert result["ack"]["expires_utc"] == "2026-10-04T05:00:00.0000000Z"
    assert initial.read_text() == '{"synthetic_fixture_only":true}'
    assert not ack_path.read_bytes().startswith(b"\xef\xbb\xbf")
    if mode == "closed":
        assert result["code"] == 1, json.dumps(result, indent=2)  # Pilot/parent-confirmation exit is not fabricated success.
        assert result["status"]["state"] == "closed" and result["status"]["safe_to_stop_host"]
        assert events.index("start:caretaker") < events.index("start:coordinator")
        assert events.index("reconcile") < events.index("stop:caretaker") < events.index("power:2147483648")
    else:
        assert result["code"] != 0 and not result["status"]["safe_to_stop_host"]
        assert "stop:caretaker" not in events and "power:2147483648" not in events
        assert result["status"]["needs_parent_attention"] is True
        if mode in {"pending", "live", "offline", "bad-evidence", "reporting-failure"}:
            assert result["status"]["state"] == "expired_unconfirmed"
            assert result["clock"] >= 5 * 3600
        if mode in {"power-zero", "power-unknown", "wrong-distro"}:
            assert "start:coordinator" not in events


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("host_runner_test", ROOT / "tools/colab/run_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def host_receipt(runner, tmp_path, monkeypatch):
    root = tmp_path / "project"
    script = root / "tools/colab/host_v3.ps1"
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    proc = tmp_path / "proc/77"
    proc.mkdir(parents=True)
    (proc / "stat").write_text("77 (synthetic caretaker) " + " ".join(["S"] + ["0"] * 18 + ["1234"]))
    boot = tmp_path / "proc/sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("synthetic-boot")
    now = datetime(2026, 10, 4, 12, tzinfo=timezone.utc).timestamp()
    def stamp(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    ack = dict(schema_version=1, acknowledged=True, accept_allocations=True, state="active", host_lease_id="a" * 32,
        host_id="12345678-1234-1234-1234-123456789abc", wsl_distro="synthetic-distro", keep_host_awake=True, keep_wsl_alive=True,
        issued_utc=stamp(now), probed_utc=stamp(now), expires_utc=stamp(now + 86400),
        host_process_id=11, host_process_started_utc=stamp(now - 60), host_script=str(script), host_script_wsl=str(script),
        host_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), caretaker_process_id=42,
        caretaker_process_started_utc=stamp(now - 30), caretaker_pid=77, caretaker_starttime="1234", power_request_return=2147483648,
        caretaker_deadline_monotonic=86400, host_remaining_seconds=86400)
    note = root / "caretaker.json"
    note.write_text(json.dumps(dict(host_lease_id=ack["host_lease_id"], pid=77, starttime="1234", wsl_distro="synthetic-distro",
                                    deadline_monotonic=86400, boot_id="synthetic-boot")))
    ack["caretaker_note_wsl"] = str(note)
    path = root / "host_ack.json"
    path.write_text(json.dumps(ack))
    identity = dict(host_id=ack["host_id"], host_name="powershell", host_start=ack["host_process_started_utc"],
        host_command='powershell.exe -File "' + str(script) + '"', caretaker_name="wsl", caretaker_start=ack["caretaker_process_started_utc"],
        caretaker_command="wsl.exe --exec python3 -c synthetic " + ack["host_lease_id"])
    real_path = Path
    monkeypatch.setattr(runner, "Path", lambda value: tmp_path / "proc" if str(value) == "/proc" else
                        tmp_path / str(value).lstrip("/") if str(value).startswith("/proc/") else real_path(value))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", environ={"WSL_DISTRO_NAME": "synthetic-distro"}))
    monkeypatch.setattr(runner.time, "time", lambda: now)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0)
    calls = []
    def native(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity))
    monkeypatch.setattr(runner.subprocess, "run", native)
    return SimpleNamespace(root=root, path=path, ack=ack, identity=identity, calls=calls, now=now, stamp=stamp)


def test_host_gate_checks_actual_birth_identity_and_cleanup_vs_admission(runner, host_receipt):
    h = host_receipt
    assert runner.host_gate(h.path, h.root)["host_lease_id"] == h.ack["host_lease_id"]
    assert "-EncodedCommand" in h.calls[0]
    h.ack.update(state="draining", accept_allocations=False, expires_utc=h.stamp(h.now + 60))
    h.path.write_text(json.dumps(h.ack))
    with pytest.raises(runner.Blocked):
        runner.host_gate(h.path, h.root)
    assert runner.host_gate(h.path, h.root, True)["accept_allocations"] is False


@pytest.mark.parametrize("damage", ["zero", "unknown", "stale", "future", "expired", "renewed", "reused-pid", "foreign-helper", "foreign-caretaker", "linux-birth", "distro"])
def test_host_gate_fails_closed_without_real_native_calls(runner, host_receipt, damage):
    h = host_receipt
    if damage in {"zero", "unknown"}:
        h.ack["power_request_return"] = 0 if damage == "zero" else "UNKNOWN"
    elif damage == "stale":
        h.ack["probed_utc"] = h.stamp(h.now - 91)
    elif damage == "future":
        h.ack["probed_utc"] = h.stamp(h.now + 1)
    elif damage == "expired":
        h.ack["expires_utc"] = h.stamp(h.now - 1)
    elif damage == "renewed":
        h.ack["expires_utc"] = h.stamp(h.now + 86401)
    elif damage == "reused-pid":
        h.identity["host_start"] = h.stamp(h.now - 59)
    elif damage == "foreign-helper":
        h.identity["host_command"] = "powershell.exe -File unrelated.ps1"
    elif damage == "foreign-caretaker":
        h.identity["caretaker_command"] = "wsl.exe --exec unrelated"
    elif damage == "linux-birth":
        h.ack["caretaker_starttime"] = "9999"
    elif damage == "distro":
        h.ack["wsl_distro"] = "foreign"
    h.path.write_text(json.dumps(h.ack))
    with pytest.raises(runner.Blocked):
        runner.host_gate(h.path, h.root)


def test_clock_rollback_cannot_extend_monotonic_host_admission(runner, host_receipt, monkeypatch):
    h = host_receipt
    monkeypatch.setattr(runner.time, "monotonic", lambda: 23 * 3600)
    # Wall clock and heartbeat still appear fresh at the original start.
    with pytest.raises(runner.Blocked, match="Monotonic"):
        runner.host_gate(h.path, h.root)
    assert runner.host_gate(h.path, h.root, True)["accept_allocations"] is True


def test_draining_receipt_blocks_allocation_before_backend_or_new_reservation(operator, monkeypatch):
    s = operator
    def draining(*args, **kwargs):
        raise s.m.Blocked("blocked_host_identity", "Synthetic draining admission barrier")
    monkeypatch.setattr(s.m, "host_gate", draining)
    with pytest.raises(s.m.Blocked, match="admission barrier"):
        s.op.allocate()
    assert not s.backend.calls and not (s.op.root / s.m.BUDGET_LEDGER).exists()


def test_verified_budget_closure_is_terminal_for_late_operator_writes(operator):
    s = operator
    s.op.allocate()
    s.clock["now"] += 100
    assert s.op.release()
    s.op.finish_budget(True)
    path = Path(s.op.state["budget_ledger"])
    closed = s.m.read_json(path)
    s.clock["now"] += 900
    s.op.observe_budget()
    s.op.finish_budget(False)
    assert s.m.read_json(path) == closed
    assert s.m.budget_spent(closed) == 100


def test_active_lease_elapsed_and_deadline_survive_wall_clock_rollback(operator, monkeypatch):
    s = operator
    s.op.allocate()
    original_deadline = s.op.monotonic_deadline
    original_start = s.clock["now"]
    s.clock["now"] = original_start - 3600
    monkeypatch.setattr(s.m.time, "monotonic", lambda: original_start + 1500)
    s.op.observe_budget()
    ledger = s.m.read_json(s.op.state["budget_ledger"])
    assert ledger["jobs"][s.op.state["run_id"]]["observed_seconds"] == 1500
    assert s.op.monotonic_deadline == original_deadline
    monkeypatch.setattr(s.m.time, "monotonic", lambda: original_deadline + 1)
    with pytest.raises(s.m.Blocked, match="deadline"):
        s.op.remaining(cleanup=True)


def test_lost_wsl_clock_domain_preserves_consumption_without_fresh_allowance(operator, monkeypatch):
    s = operator
    s.op.allocate()
    ledger = s.m.read_json(s.op.state["budget_ledger"])
    monkeypatch.setattr(s.m, "BOOT_ID", "synthetic-new-wsl-boot")
    assert s.m.budget_spent(ledger) >= s.m.MAX_GPU_SECONDS


def test_unconfirmed_release_keeps_guard_and_guard_start_has_independent_session(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.backend.release_confirmed = False
    with pytest.raises(s.m.Blocked, match="unconfirmed"):
        s.op.release()
    assert not (s.op.guard / "session_guard.stop").exists()
    commands = []
    monkeypatch.setattr(s.m, "bounded_call", lambda command, **kwargs: commands.append(command) or (0, ""))
    s.op.guard_call("start")
    assert commands[0][2:5] == ["setsid", "--wait", "bash"]


def test_reconcile_does_not_steal_busy_expired_operator_or_report_safe_stop(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.clock["now"] += s.m.JOB_SECONDS + 1
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"host_lease_id": "synthetic-test-host", "accept_allocations": False})
    def lock(path, **kwargs):
        if "jev-v3-operator-" in str(path):
            raise s.m.Blocked("blocked_ownership", "Synthetic live operator retains lock")
        return nullcontext(9)
    monkeypatch.setattr(s.m, "flock", lock)
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-draining-ack")
    assert report["live_leases"] == report["pending_operators"] == 1
    assert not report["safe_to_stop_host"] and not s.backend.releases
    assert s.m.read_json(s.op.root / s.m.BUDGET_LEDGER)["jobs"][s.op.state["run_id"]]["reserved_seconds"] == s.op.item["deadline_seconds"]


def test_unallocated_detached_intent_keeps_host_custody_even_with_zero_gpu_ledger(operator, monkeypatch):
    s = operator
    s.op.save("submission_intent", startup_ack=False)
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"host_lease_id": "synthetic-test-host", "accept_allocations": False})
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-draining-ack")
    assert report["live_leases"] == 0 and report["pending_operators"] == 1
    assert not report["safe_to_stop_host"] and not s.backend.calls


def test_unknown_cpu_allocation_cannot_signal_all_owned_closed(operator, monkeypatch):
    s = operator
    s.op.item = s.m.job("ANALYSIS", "analysis")
    s.op.state["job"] = s.op.item
    s.op.save("failed", allocation_attempted=True, owned_endpoint=None, released=False)
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"host_lease_id": "synthetic-test-host", "accept_allocations": False})
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-draining-ack")
    assert report["live_leases"] == 0 and report["pending_operators"] == 1
    assert not report["safe_to_stop_host"] and not s.backend.releases


def test_authoritatively_closed_orphan_becomes_terminal_without_new_work(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.clock["now"] += 200
    # Simulate a lost worker and already absent backend; no status completion claim.
    s.backend.rows = [r for r in s.backend.rows if r["endpoint"] != s.op.owner]
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"host_lease_id": "synthetic-test-host", "accept_allocations": False})
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-draining-ack")
    assert report["safe_to_stop_host"] and report["pending_operators"] == report["live_leases"] == 0
    state = s.m.read_json(s.out / "status.json")
    assert state["status"] == "failed" and state["released"] is True
    assert not state["verified"] and not state["completed_execution"]
    assert state["observed_gpu_seconds_upper_bound"] == 200


def test_missing_terminal_end_bound_never_discards_monotonic_consumption(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.op.save("failed", released=True, completed_execution=False, verified=False)
    state = s.m.read_json(s.out / "status.json")
    state["updated_utc"] = "1970-01-01T00:16:40+00:00"
    s.m.atomic_json(s.out / "status.json", state)
    s.backend.rows = [r for r in s.backend.rows if r["endpoint"] != s.op.owner]
    monkeypatch.setattr(s.m.time, "time", lambda: 900.0)
    monkeypatch.setattr(s.m.time, "monotonic", lambda: 1500.0)
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    ledger = s.m.budget_ledger(s.op.root)
    s.op.reconcile_budget(ledger)
    assert ledger["jobs"][s.op.state["run_id"]]["actual_seconds"] >= 500


def test_worker_preflight_failure_closes_only_its_unallocated_intent(operator, monkeypatch):
    s = operator
    lock = s.op.root / "synthetic-worker-lock"
    lock.write_text("synthetic")
    expected = "/tmp/jev-v3-operator-" + s.op.item["session"] + ".lock"
    s.op.state.update(operator_lock=expected, operator_lock_fd=9)
    s.op.save("submission_intent", startup_ack=False)
    real_path = Path
    monkeypatch.setattr(s.m, "Path", lambda value: lock if str(value) == expected else real_path(value))
    monkeypatch.setattr(s.m.os, "fstat", lambda descriptor: lock.stat())
    def rejected(*args, **kwargs):
        raise s.m.Blocked("blocked_integrity", "Synthetic source mismatch before allocation")
    monkeypatch.setattr(s.m, "preflight", rejected)
    assert s.m.main(["--root", str(s.op.root), "--_worker", str(s.out)]) == 1
    state = s.m.read_json(s.out / "status.json")
    assert state["status"] == "failed" and state["allocation_attempted"] is False
    assert state["released"] is True and state["release_evidence"] == "durable_no_allocation"
    assert state["completed_execution"] is False and not s.backend.calls


@pytest.mark.parametrize("admitting", [False, True])
def test_safe_stop_receipt_binds_closed_ledger_and_draining_admission(operator, monkeypatch, admitting):
    s = operator
    s.op.allocate()
    s.clock["now"] += 100
    assert s.op.release()
    s.op.finish_budget(True)
    s.op.save("completed", completed_execution=True, verified=True, released=True)
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"host_lease_id": "synthetic-test-host", "accept_allocations": admitting})
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-host-ack")
    assert report["live_leases"] == report["pending_operators"] == 0
    assert report["safe_to_stop_host"] is (not admitting)
    assert report["ledger_sha256"] == s.m.digest(s.op.root / s.m.BUDGET_LEDGER)
    assert report["host_lease_id"] == "synthetic-test-host" and report["max_gpu_seconds"] == 25 * 3600
