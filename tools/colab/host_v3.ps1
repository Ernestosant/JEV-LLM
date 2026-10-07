param(
    [Parameter(Mandatory=$true)][string]$InitialBudget,
    [Parameter(Mandatory=$true)][string]$HostAck,
    [string]$Approval = "",
    [string]$PilotReport = "",
    [string]$AttemptPrefix = "",
    [string]$Distro = "",
    [string]$Python = "python3",
    [string]$Root = "",
    [ValidateRange(4,24)][int]$Hours = 24,
    [switch]$ReconcileOnly,
    [switch]$Recovery,
    [ValidateSet("", "test", "latency")][string]$PostPilotPhase = ""
)

$ErrorActionPreference = "Stop"
$script:V3HostScript = $PSCommandPath

function Convert-ToWslPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if ($full -notmatch '^[A-Za-z]:\\') { throw "A local Windows drive path is required" }
    return "/mnt/" + $full.Substring(0,1).ToLowerInvariant() + "/" + $full.Substring(3).Replace("\", "/")
}

function Join-NativeArguments([string[]]$Values) {
    # CommandLineToArgvW escaping, including embedded quotes and trailing slashes.
    return (($Values | ForEach-Object {
        if ($_ -match '^[A-Za-z0-9_./:=\\-]+$') { $_ }
        else { '"' + [regex]::Replace([regex]::Replace($_, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"' }
    }) -join ' ')
}

function Write-HostJson([string]$Path, $Value) {
    $temporary = $Path + "." + [Guid]::NewGuid().ToString("N") + ".tmp"
    try {
        [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 12) + "`n"), [Text.UTF8Encoding]::new($false))
        if ([IO.File]::Exists($Path)) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
        else { [IO.File]::Move($temporary, $Path) }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Get-HostUtc { return [DateTime]::UtcNow }
function Get-HostMonotonic { return [Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency }

function Set-HostPower([uint32]$Flags) {
    if (-not ("JevV3PowerLease" -as [type])) {
        Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class JevV3PowerLease { [DllImport("kernel32.dll", SetLastError=true)] public static extern uint SetThreadExecutionState(uint flags); }'
    }
    return [JevV3PowerLease]::SetThreadExecutionState($Flags)
}

function Get-HostIdentity {
    $identity = [string](Get-CimInstance Win32_ComputerSystemProduct).UUID
    if ($identity -notmatch '^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$' -or
        $identity -match '^(00000000-0000-0000-0000-000000000000|FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF)$') {
        throw "Actual Windows CIM identity is unavailable"
    }
    return @{ host_id = $identity.ToLowerInvariant(); host_process_started_utc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString("o") }
}

function Start-HostWsl([string[]]$Arguments, [string]$Mode, [string]$Directory, [string]$Prefix) {
    $output = $Prefix + ".out.log"
    $errors = $Prefix + ".err.log"
    $options = @{ FilePath = "wsl.exe"; ArgumentList = (Join-NativeArguments $Arguments); WorkingDirectory = $Directory;
                  PassThru = $true; RedirectStandardOutput = $output; RedirectStandardError = $errors }
    if ($Mode -eq "coordinator") { $options.NoNewWindow = $true }
    else { $options.WindowStyle = "Hidden" }
    $process = Start-Process @options
    # Windows PowerShell must retain the handle before a short WSL probe exits.
    $null = $process.Handle
    $born = if ($Mode -eq "caretaker") { $process.StartTime.ToUniversalTime().ToString("o") } else { $null }
    return @{ process = $process; output = $output; errors = $errors;
              started_utc = $born }
}

function Read-HostWslJson([string[]]$Arguments, [string]$Directory, [string]$Prefix,
                          [int]$TimeoutSeconds = 30, [scriptblock]$Heartbeat) {
    $client = Start-HostWsl $Arguments "probe" $Directory $Prefix
    $deadline = (Get-HostMonotonic) + $TimeoutSeconds
    try {
        while (-not $client.process.WaitForExit(1000)) {
            if ($Heartbeat) { & $Heartbeat }
            if ((Get-HostMonotonic) -ge $deadline) { throw "Owned WSL request timed out" }
        }
        if ($client.process.ExitCode -ne 0) { throw "Owned WSL request failed; details suppressed" }
        return ([IO.File]::ReadAllText($client.output, [Text.Encoding]::UTF8) | ConvertFrom-Json)
    } finally {
        # This handle belongs only to the short-lived request we just created.
        if (-not $client.process.HasExited) { $client.process.Kill() }
    }
}

function Stop-HostCaretaker($Caretaker, [string]$StopPath, [string]$LeaseId) {
    Write-HostJson $StopPath @{ host_lease_id = $LeaseId; stop = $true }
    return $Caretaker.process.WaitForExit(15000)
}

function Test-HostClosure($Report, [string]$LedgerPath, [string]$LeaseId) {
    if ($Report.schema_version -ne 1 -or $Report.host_lease_id -ne $LeaseId -or
        $Report.ledger -ne (Convert-ToWslPath $LedgerPath) -or
        $Report.safe_to_stop_host -isnot [bool] -or $Report.allocation_permitted -isnot [bool] -or
        $Report.experiment_id -ne "jev-llm-v3-20261003" -or $Report.max_gpu_seconds -ne 90000 -or
        $Report.allocation_permitted -ne $false -or $Report.live_leases -ne 0 -or
        $Report.pending_operators -ne 0 -or $Report.safe_to_stop_host -ne $true -or
        -not [IO.File]::Exists($LedgerPath) -or
        (Get-FileHash -LiteralPath $LedgerPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Report.ledger_sha256) { return $false }
    $ledger = [IO.File]::ReadAllText($LedgerPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
    if ($ledger.experiment_id -ne "jev-llm-v3-20261003" -or $ledger.max_gpu_seconds -ne 90000 -or $null -eq $ledger.jobs) { return $false }
    foreach ($property in $ledger.jobs.PSObject.Properties) {
        $entry = $property.Value
        if ($entry.released -isnot [bool] -or $entry.release_verified -isnot [bool] -or
            $entry.released -ne $true -or $entry.release_verified -ne $true -or $null -eq $entry.actual_seconds -or
            $entry.actual_seconds -is [bool] -or $entry.actual_seconds -is [string] -or [double]$entry.actual_seconds -lt 0 -or
            [double]::IsNaN([double]$entry.actual_seconds) -or [double]::IsInfinity([double]$entry.actual_seconds) -or
            $entry.release_evidence -notin @("backend_absent", "durable_no_allocation") -or
            ($entry.allocation_attempted -eq $true -and $entry.release_evidence -ne "backend_absent")) { return $false }
    }
    return $true
}

function Invoke-V3Host([string]$InitialBudget, [string]$HostAck, [string]$Root,
                       [string]$Approval = "", [string]$PilotReport = "", [string]$Distro = "",
                       [string]$Python = "python3", [int]$Hours = 24, [bool]$ReconcileOnly = $false,
                        [string]$AttemptPrefix = "", [bool]$Recovery = $false,
                        [string]$PostPilotPhase = "") {
    if ($AttemptPrefix -and $AttemptPrefix -notmatch '^infra[0-9]+$') { throw "Infrastructure attempt prefix must be infra followed by digits" }
    if ($Hours -lt 4 -or $Hours -gt 24 -or ($Hours -eq 4 -and -not $ReconcileOnly)) {
        throw "Allocation mode needs more than four hours of custody; maximum 24. Four hours is cleanup-only."
    }
    if (-not $Root) { $Root = Join-Path ([IO.Path]::GetDirectoryName($script:V3HostScript)) "..\.." }
    $Root = [IO.Path]::GetFullPath($Root)
    $HostAck = [IO.Path]::GetFullPath($HostAck)
    $InitialBudget = [IO.Path]::GetFullPath($InitialBudget)
    foreach ($path in @($InitialBudget, (Join-Path $Root "tools\colab\run_v3.py"))) {
        if (-not [IO.File]::Exists($path)) { throw "Required parent input or coordinator is missing" }
    }
    if ($Recovery -and ($AttemptPrefix -ne "infra03" -or -not [IO.File]::Exists((Join-Path $Root "tools\colab\resume_v3.py")))) {
        throw "Recovery requires the isolated infra03 coordinator"
    }
    if ($PostPilotPhase -and ($PostPilotPhase -notin @("test", "latency") -or $Recovery -or $ReconcileOnly -or
        -not $Approval -or -not $PilotReport -or -not [IO.File]::Exists((Join-Path $Root "tools\colab\postpilot_v3.py")))) {
        throw "Post-pilot requires an approved test/latency phase without recovery or cleanup-only mode"
    }
    if ([bool]$Approval -ne [bool]$PilotReport) { throw "Confirmation and pilot report must be supplied together" }
    if ($Approval -and (-not [IO.File]::Exists($Approval) -or -not [IO.File]::Exists($PilotReport))) { throw "Actual confirmation evidence is missing" }
    $directory = [IO.Path]::GetDirectoryName($HostAck)
    if (-not [IO.Directory]::Exists($directory)) { throw "Host acknowledgement parent is missing" }
    $protected = @($InitialBudget, $Approval, $PilotReport, (Join-Path $Root "results\v3\budget_ledger.json"), (Join-Path $Root "config\experiment_v3.json"))
    foreach ($path in $protected) {
        if ($path -and ([IO.Path]::GetFullPath($path) -in @($HostAck, (Join-Path $directory "hoststatus.json")))) { throw "Host outputs must not replace consent, config or budget evidence" }
    }
    [void](Convert-ToWslPath $Root)
    foreach ($path in @($Root, $directory, $HostAck)) {
        $cursor = $path
        while ($cursor) {
            if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Reparse host path forbidden" }
            $cursor = [IO.Path]::GetDirectoryName($cursor)
        }
    }
    $mutexPath = Join-Path $Root "results\v3\host_v3.lock"
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($mutexPath))
    $mutex = [IO.File]::Open($mutexPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $leaseId = [Guid]::NewGuid().ToString("N")
    $prefix = Join-Path $directory ("host_" + $leaseId)
    $statusPath = Join-Path $directory "hoststatus.json"
    $notePath = $prefix + ".caretaker.json"
    $stopPath = $prefix + ".stop.json"
    $issued = Get-HostUtc
    $deadline = (Get-HostMonotonic) + $Hours * 3600
    $ack = [ordered]@{ schema_version = 1; host_lease_id = $leaseId; acknowledged = $false; accept_allocations = $false;
        keep_host_awake = $false; keep_wsl_alive = $false; issued_utc = $issued.ToString("o");
        expires_utc = $issued.AddHours($Hours).ToString("o"); host_process_id = $PID;
        host_script = $script:V3HostScript; host_script_wsl = (Convert-ToWslPath $script:V3HostScript);
        host_script_sha256 = (Get-FileHash -LiteralPath $script:V3HostScript -Algorithm SHA256).Hash.ToLowerInvariant();
        state = "starting"; power_request_return = 0 }
    $status = [ordered]@{ schema_version = 1; host_lease_id = $leaseId; host_process_id = $PID; host_ack = $HostAck;
        host_log = $prefix + ".jsonl"; state = "starting"; needs_parent_attention = $false; safe_to_stop_host = $false;
        expires_utc = $ack.expires_utc; coordinator_exit_code = $null;
        limitations = "Scoped request cannot prevent forced shutdown, logout, process termination or offline backend uncertainty." }
    $caretaker, $coordinator, $pulse = $null, $null, $null
    $exitCode, $safe, $powerRequested = 2, $false, $false
    try {
        $identity = Get-HostIdentity
        $ack.host_id, $ack.host_process_started_utc = $identity.host_id, $identity.host_process_started_utc
        $ack.power_request_return = [uint32](Set-HostPower ([uint32]2147483649)) # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if ($ack.power_request_return -eq 0) { throw "Windows refused the scoped power request" }
        $powerRequested = $true
        $probeArgs = @("--exec", $Python, "-B", "-c", "import json,os; print(json.dumps({'wsl_distro':os.environ.get('WSL_DISTRO_NAME')}))")
        if ($Distro) { $probeArgs = @("--distribution", $Distro) + $probeArgs }
        $probe = Read-HostWslJson $probeArgs $Root ($prefix + ".identity")
        if (-not $probe.wsl_distro -or ($Distro -and $Distro -ne $probe.wsl_distro)) { throw "Actual WSL distro is unknown or mismatched" }
        $Distro = [string]$probe.wsl_distro
        $ack.wsl_distro = $Distro
        $base = @("--distribution", $Distro, "--exec", $Python, "-B")
        $caretakerCode = @'
import json,os,pathlib,sys,time
note,stop=map(pathlib.Path,sys.argv[1:3]); lease=sys.argv[3]
deadline=time.monotonic()+max(0,float(sys.argv[4]))
birth=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
tmp=note.with_name(note.name+'.pending')
tmp.write_text(json.dumps(dict(host_lease_id=lease,pid=os.getpid(),starttime=birth,wsl_distro=os.environ.get('WSL_DISTRO_NAME'),deadline_monotonic=deadline,started_monotonic=time.monotonic(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())),encoding='utf-8'); tmp.replace(note)
while time.monotonic()<deadline:
    if stop.exists():
        request=json.loads(stop.read_text(encoding='utf-8'))
        if request.get('host_lease_id')==lease and request.get('stop') is True: break
    time.sleep(min(2,max(0,deadline-time.monotonic())))
'@
        $caretaker = Start-HostWsl ($base + @("-c", $caretakerCode, (Convert-ToWslPath $notePath), (Convert-ToWslPath $stopPath), $leaseId, [string]($deadline - (Get-HostMonotonic)))) "caretaker" $Root ($prefix + ".caretaker")
        $readyUntil = [Math]::Min($deadline, (Get-HostMonotonic) + 30)
        while (-not [IO.File]::Exists($notePath) -and -not $caretaker.process.HasExited -and (Get-HostMonotonic) -lt $readyUntil) { Start-Sleep -Milliseconds 100 }
        if (-not [IO.File]::Exists($notePath) -or $caretaker.process.HasExited) { throw "Owned caretaker readiness is unproved" }
        $note = [IO.File]::ReadAllText($notePath, [Text.Encoding]::UTF8) | ConvertFrom-Json
        if ($note.host_lease_id -ne $leaseId -or $note.wsl_distro -ne $Distro -or $note.pid -le 0 -or -not $note.starttime) { throw "Owned caretaker identity mismatch" }
        $ack.caretaker_process_id, $ack.caretaker_process_started_utc = $caretaker.process.Id, $caretaker.started_utc
        $ack.caretaker_pid, $ack.caretaker_starttime = $note.pid, [string]$note.starttime
        $ack.caretaker_deadline_monotonic, $ack.caretaker_boot_id = $note.deadline_monotonic, $note.boot_id
        $ack.caretaker_note_wsl, $ack.caretaker_stop = (Convert-ToWslPath $notePath), $stopPath
        $status.caretaker_process_id, $status.caretaker_process_started_utc = $ack.caretaker_process_id, $ack.caretaker_process_started_utc
        $status.caretaker_pid, $status.caretaker_starttime, $status.caretaker_stop = $ack.caretaker_pid, $ack.caretaker_starttime, $stopPath
        $pulse = {
            try {
                $ack.power_request_return = [uint32](Set-HostPower ([uint32]2147483649))
                $current = Read-HostWslJson (@("--distribution", $Distro) + @("--exec", $Python, "-B", "-c", "import json,os; print(json.dumps({'wsl_distro':os.environ.get('WSL_DISTRO_NAME')}))")) $Root ($prefix + ".probe")
                $ack.keep_host_awake = $ack.power_request_return -ne 0
                $ack.keep_wsl_alive = -not $caretaker.process.HasExited -and $current.wsl_distro -eq $Distro
                $ack.acknowledged = $ack.keep_host_awake -and $ack.keep_wsl_alive
            } catch { $ack.acknowledged = $false; $ack.keep_host_awake = $false; $ack.keep_wsl_alive = $false; $status.error_type = $_.Exception.GetType().Name }
            $ack.probed_utc = (Get-HostUtc).ToString("o")
            $ack.host_remaining_seconds = [Math]::Max(0, $deadline - (Get-HostMonotonic))
            $status.state, $status.keep_host_awake, $status.keep_wsl_alive = $ack.state, $ack.keep_host_awake, $ack.keep_wsl_alive
            $status.updated_utc = $ack.probed_utc
            Write-HostJson $HostAck $ack
            Write-HostJson $statusPath $status
            [IO.File]::AppendAllText($status.host_log, (($status | ConvertTo-Json -Depth 12 -Compress) + "`n"), [Text.UTF8Encoding]::new($false))
        }.GetNewClosure()
        $ack.state, $ack.accept_allocations = "active", -not $ReconcileOnly
        & $pulse
        if (-not $ack.acknowledged) { throw "Actual host/caretaker request is unconfirmed" }
        $runner = Convert-ToWslPath (Join-Path $Root "tools\colab\run_v3.py")
        if (-not $ReconcileOnly) {
            $coordinatorRunner = if ($Recovery) { Convert-ToWslPath (Join-Path $Root "tools\colab\resume_v3.py") } else { $runner }
            if ($PostPilotPhase) { $coordinatorRunner = Convert-ToWslPath (Join-Path $Root "tools\colab\postpilot_v3.py") }
            $phase = if ($PostPilotPhase) { $PostPilotPhase } else { "full" }
            $arguments = $base + @($coordinatorRunner, "--root", (Convert-ToWslPath $Root), "--run", "--phase", $phase, "--authorize-initial",
                                  "--initial-budget", (Convert-ToWslPath $InitialBudget), "--host-ack", (Convert-ToWslPath $HostAck))
            if ($Approval) { $arguments += @("--authorize-confirmatory", "--approval", (Convert-ToWslPath $Approval), "--pilot-report", (Convert-ToWslPath $PilotReport)) }
            if ($AttemptPrefix) { $arguments += @("--attempt-prefix", $AttemptPrefix) }
            $coordinator = Start-HostWsl $arguments "coordinator" $Root ($prefix + ".coordinator")
            $status.coordinator_process_id, $status.coordinator_log = $coordinator.process.Id, $coordinator.output
        }
        while ((Get-HostMonotonic) -lt $deadline) {
            if ($null -eq $coordinator -or $coordinator.process.HasExited) {
                if ($coordinator) { $status.coordinator_exit_code = $coordinator.process.ExitCode }
                $ack.state, $ack.accept_allocations = "draining", $false
                & $pulse # Close admission before acquiring the allocation/reconciliation lock.
                try {
                    $report = Read-HostWslJson ($base + @($runner, "--root", (Convert-ToWslPath $Root), "--reconcile-budget", "--host-ack", (Convert-ToWslPath $HostAck))) $Root ($prefix + ".reconcile") ([int][Math]::Min(180, [Math]::Max(1, $deadline - (Get-HostMonotonic)))) $pulse
                    $status.live_leases, $status.pending_operators = $report.live_leases, $report.pending_operators
                    $safe = Test-HostClosure $report (Join-Path $Root "results\v3\budget_ledger.json") $leaseId
                } catch { $safe = $false; $status.error_type = $_.Exception.GetType().Name }
                if ($safe) { break }
                $status.needs_parent_attention = $true
            }
            & $pulse
            Start-Sleep -Seconds 15
        }
        if ($safe -and (Stop-HostCaretaker $caretaker $stopPath $leaseId)) {
            $status.power_clear_return = Set-HostPower ([uint32]2147483648) # Clear this thread's request only.
            $status.safe_to_stop_host = $true
            $status.state = "closed"
            $exitCode = if ($ReconcileOnly) { 0 } else { [int]$status.coordinator_exit_code }
            if ($status.power_clear_return -eq 0) { $status.state = "power_clear_unconfirmed"; $status.needs_parent_attention = $true; $exitCode = 2 }
        } else { $status.state = "expired_unconfirmed"; $status.needs_parent_attention = $true; $exitCode = 3 }
    } catch {
        $status.error_type = $_.Exception.GetType().Name
        $status.state, $status.needs_parent_attention = "startup_unconfirmed", $true
        # Never stop a caretaker or clear power on coordinator/startup failure.
        if ($powerRequested -or $caretaker) {
            $ack.accept_allocations = $false
            $ack.keep_host_awake, $ack.keep_wsl_alive = $false, $false
            while ((Get-HostMonotonic) -lt $deadline) {
                $ack.acknowledged = $false
                try { [void](Set-HostPower ([uint32]2147483649)) } catch { }
                try { Write-HostJson $HostAck $ack; Write-HostJson $statusPath $status }
                catch { [Console]::Error.WriteLine("V3_HOST: reporting failed; custody retained until fixed expiry.") }
                Start-Sleep -Seconds 15
            }
            $status.state = "expired_unconfirmed"
            $exitCode = 3
        }
    } finally {
        $ack.acknowledged, $ack.accept_allocations, $ack.keep_host_awake, $ack.keep_wsl_alive = $false, $false, $false, $false
        $ack.state, $ack.host_finished_utc = $status.state, (Get-HostUtc).ToString("o")
        $status.keep_host_awake, $status.keep_wsl_alive = $false, $false
        $status.finished_utc, $status.helper_exit_code = $ack.host_finished_utc, $exitCode
        try { Write-HostJson $HostAck $ack; Write-HostJson $statusPath $status }
        catch { [Console]::Error.WriteLine("V3_HOST: final receipt unavailable; do not infer successful cleanup."); $exitCode = [Math]::Max(2, $exitCode) }
        finally { $mutex.Dispose() }
    }
    return $exitCode
}

if ($MyInvocation.InvocationName -ne ".") {
    exit (Invoke-V3Host $InitialBudget $HostAck $Root $Approval $PilotReport $Distro $Python $Hours ([bool]$ReconcileOnly) $AttemptPrefix ([bool]$Recovery) $PostPilotPhase)
}
