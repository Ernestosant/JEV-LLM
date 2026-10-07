"""Finish an already authorized v3 run across the real post-pilot gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

from run_v3 import CONDITIONS, EXPERIMENT_ID, MAX_GPU_SECONDS, atomic_json, digest, epoch, read_json, require


def reconcile_unassigned_smoke(root, directory, host_ack):
    """Close an uncertain startup charge only after two empty backend observations."""
    from run_v3 import Backend, Operator, budget_ledger, flock, host_gate, lease_elapsed, status_gate
    root, directory = Path(root).resolve(), Path(directory).resolve()
    host_gate(host_ack, root, cleanup=True)
    require(directory.is_relative_to(root / "results/v3"), "Startup output escaped results")
    with flock("/tmp/jev-colab-allocation.lock", wait=30), flock("/tmp/jev-v3-budget.lock", wait=30):
        state = read_json(directory / "status.json")
        item = state["job"]
        status_gate(root, directory, item)
        require(item["phase"] == "smoke" and state.get("status") == "failed"
                and state.get("allocation_attempted") is True and state.get("before_endpoints") == []
                and not state.get("owned_endpoint") and not state.get("remote_startup_ack")
                and state.get("completed_execution") is False, "Not an unassigned diagnostic startup failure")
        with flock(state["operator_lock"], wait=0):
            backend = Backend(directory, item["session"])
            for attempt in range(2):
                observed = backend.identity()
                require(observed["assignments"] == [] and observed["local_endpoint"] is None,
                        "Backend is not empty; unknown assignments must remain untouched")
                if attempt == 0:
                    time.sleep(5)
            ledger = budget_ledger(root)
            entry = ledger["jobs"][state["run_id"]]
            require(entry["session"] == item["session"] and entry["output"] == str(directory)
                    and not entry.get("endpoint") and entry.get("released") is False, "Startup lease identity changed")
            actual = lease_elapsed(entry)
            entry.update(released=True, release_verified=True, actual_seconds=actual, observed_seconds=actual,
                         observed_end_epoch=time.time(), release_evidence="backend_absent", unassigned_startup_reconciled=True)
            atomic_json(root / "results/v3/budget_ledger.json", ledger)
            Operator(root, directory, state).save("failed", released=True, release_evidence="backend_absent",
                observed_gpu_seconds_upper_bound=actual, unassigned_startup_reconciled=True)
            return dict(released=True, charged_seconds_upper_bound=actual, backend_observations=2)


def confirm(root, state, ledger):
    initial = read_json(root / "results/v3/initial_budget.json")
    config = read_json(root / "config/experiment_v3.json")
    require(initial.get("approved") is True and initial.get("scope") == "all_gpu_phases_conditional_technical_go"
            and initial.get("experiment_id") == EXPERIMENT_ID and initial.get("max_gpu_hours") == 25
            and initial.get("max_total_assignments") == 2 and initial.get("technical_go_required") is True
            and initial.get("config_sha256") == digest(root / "config/experiment_v3.json")
            and initial.get("models") == config["models"], "Existing conditional user authorization changed")
    name = state["pilot_report"]
    if name.startswith("/mnt/"):
        name = name[5].upper() + ":/" + name[7:]
    report_path = Path(name).resolve()
    require(report_path.is_relative_to(root / "results/v3/pilot")
            and digest(report_path) == state["pilot_report_sha256"], "Actual pilot report binding changed")
    report = read_json(report_path)
    require(state.get("technical_go") is True and state.get("budget_ready") is True
            and report.get("go") is True and report.get("errors") == []
            and report.get("budget_approval") is False and set(report["conditions"]) == set(CONDITIONS)
            and report["aggregate"].get("complete") is True and report["aggregate"].get("observed_cases") == 600
            and all(entry.get("verified") is True for entry in report["conditions"].values()), "No real complete technical GO")
    require(ledger.get("experiment_id") == EXPERIMENT_ID and ledger.get("max_gpu_seconds") == MAX_GPU_SECONDS
            and all(e.get("released") is True and e.get("release_verified") is True for e in ledger["jobs"].values()),
            "Prior GPU leases must be authoritatively released")
    spent = sum(e["actual_seconds"] for e in ledger["jobs"].values())
    require(sum(state["remaining_estimates"].values()) <= MAX_GPU_SECONDS - spent, "Remaining matrix exceeds original total budget")
    value = dict(approved=True, scope="test_and_latency", experiment_id=EXPERIMENT_ID, max_gpu_hours=25,
                 config_sha256=initial["config_sha256"], pilot_report_sha256=state["pilot_report_sha256"],
                 initial_authorization_sha256=hashlib.sha256(json.dumps(initial, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                 approved_by="user (existing conditional all-phase authorization)",
                 approved_utc=datetime.now(timezone.utc).isoformat(), no_additional_allowance_after_pilot=True,
                 authorization_source="Records existing user authorization to finish v3, download and analyze results; actual pilot GO and remaining total budget now verified. No new consent or allowance is created.")
    require(epoch(value["approved_utc"]) >= epoch(report["meta"]["generated_utc"]), "Confirmation predates actual pilot decision")
    approval = root / "results/v3/confirmatory_approval_infra03.json"
    if approval.exists():
        previous = read_json(approval)
        require(all(previous.get(k) == v for k, v in value.items() if k != "approved_utc"), "Existing confirmation changed")
    else:
        atomic_json(approval, value)
    return approval, report_path


def export_analysis(root, state):
    directory = root / "results/v3/analysis/ANALYSIS"
    archive = directory / "artifacts/analysis_confirmatory_v3.zip"
    verification = state["verified_jobs"]["analysis/ANALYSIS"]
    require(verification.get("ok") is True and digest(archive) == verification["archive_sha256"], "Analysis archive verification changed")
    destination = root / "results/v3/final_analysis"
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())) and z.testzip() is None
                and sum(i.file_size for i in z.infolist()) <= 256 * 1024 * 1024, "Invalid or oversized analysis archive")
        for name in z.namelist():
            p = PurePosixPath(name)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name,
                    "Unsafe analysis export path")
        report = json.loads(z.read("analysis.json"))
        require(report["meta"].get("n_physical_cases") == 6000 and report["meta"].get("n_test_problems") == 500
                and report["parser_audit"].get("n_disagreements") == 0, "Incomplete or disputed analysis")
        require(not any(p.is_symlink() for p in (destination, *destination.parents)), "Symlink analysis destination forbidden")
        destination.mkdir(exist_ok=True)
        for name in z.namelist():
            target = destination / name
            require(not any(p.is_symlink() for p in (target, *target.parents)), "Symlink analysis export forbidden")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = z.read(name)
            if target.exists():
                require(target.read_bytes() == content, "Existing analysis export differs")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    quality, latency = report["primary"]["quality"], report["primary"]["latency"]
    lines = ["# Resultados JEV-LLM v3", "", "500 problemas de test; 6000 inferencias y 1800 mediciones de latencia.",
             "Las semillas no son problemas independientes. El baseline principal es el modelo de 13B greedy.", "",
             "## Comparacion Principal", "",
             f"- Diferencia de exactitud JFINAL - B13_GREEDY: {quality['acc_diff_pp']:.6g} puntos porcentuales.",
             f"- IC 97.5% de exactitud: {quality['ci97_5_bonferroni']}.",
             f"- Speedup geometrico pareado: {latency['speedup']:.6g}; mayor que 1 favorece al hibrido.",
             f"- IC 97.5% de speedup: {latency['ci97_5_bonferroni']}.", "",
             "Los IC coprincipales usan bootstrap por problema, estratificado por dominio, y ajuste Bonferroni.",
             "La latencia usa 100 problemas y la mediana de nueve mediciones por sistema y problema.", "",
             "## Exactitud", "", "| Condicion | Inferencias | Problemas | Exactitud % |", "|---|---|---|---|"]
    lines.extend(f"| {r['condition']} | {r['n_cases']} | {r['n_problems']} | {r['accuracy_pct']:.6g} |" for r in report["table"])
    lines.extend(["", "## Conclusiones Verificadas", "", "```json", json.dumps(report["declarations"], indent=2), "```", "",
                  "El balance es por nivel de fuente, no por dificultad intrinseca. La revision fue agentica, no humana.",
                  "Ver report.md y analysis.json para controles, contrastes secundarios, auditoria y limitaciones.", ""])
    target = destination / "resumen_es.md"
    content = "\n".join(lines).encode("utf-8")
    require(not target.exists() or target.read_bytes() == content, "Existing Spanish summary differs")
    target.write_bytes(content)
    return str(target)


def read_live_json(path):
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError):
            if attempt == 4:
                raise
            time.sleep(0.2)  # Windows replacement can briefly deny or hide the file.


def complete(root, prefix="infra03", recovery=False):
    root = Path(root).resolve()
    out = root / "results/v3"
    resumed = False
    started = time.monotonic()
    while time.monotonic() - started < 24 * 3600:
        state = read_live_json(out / ("coordinator_full_" + prefix + ("_recovery" if recovery else "") + "_state.json"))
        host = read_live_json(out / "hoststatus.json")
        outcome = state["outcome"]
        status = dict(updated_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
                      outcome=outcome, host_state=host["state"], verified_jobs=list(state.get("verified_jobs", {})))
        atomic_json(out / "completion_status.json", status)
        if host["state"] != "closed":
            time.sleep(30)
            continue
        require(host.get("safe_to_stop_host") is True and host.get("live_leases") == 0
                and host.get("pending_operators") == 0, "Host closure is not verified")
        if outcome == "completed":
            require(len(state["verified_jobs"]) == 21, "Full plan is not verified")
            status.update(outcome="completed", report=export_analysis(root, state))
            atomic_json(out / "completion_status.json", status)
            return 0
        require(outcome == "blocked_budget_approval" and not resumed, "Execution stopped: " + outcome)
        approval, report = confirm(root, state, read_live_json(out / "budget_ledger.json"))
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(root / "tools/colab/host_v3.ps1"),
                   "-Root", str(root), "-Python", "/root/.local/share/uv/tools/google-colab-cli/bin/python",
                   "-InitialBudget", str(out / "initial_budget.json"), "-HostAck", str(out / "host_ack.json"),
                   "-AttemptPrefix", prefix, "-Approval", str(approval), "-PilotReport", str(report), "-Hours", "24"]
        if recovery:
            command.append("-Recovery")
        with (out / "completion_resume_host.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        resumed = True
        # Allow the new host to replace the closed receipt before observing it.
        time.sleep(30)
        require(child.poll() is None, "Confirmatory host exited during startup")
    raise RuntimeError("Completion supervision exceeded its 24-hour wall-clock bound")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    import msvcrt
    with (args.root / "results/v3/completion.lock").open("a+b") as lock:
        lock.write(b"\0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            raise SystemExit(complete(args.root, recovery=args.recovery))
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            atomic_json(args.root / "results/v3/completion_status.json", dict(outcome="failed",
                        updated_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error_message=str(error)))
            raise
