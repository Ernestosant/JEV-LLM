"""Offline-first v3 coordinator and durable, owned-only Colab operator.

No API/auth call is made by --plan or by a failed local readiness gate. --run
defaults to development smoke + technical pilot, never automatic test promotion.
The host must remain awake with WSL alive; a detached process is not a host lease.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
# Read-only utility reuse; no legacy launch, pull, status, verifier or retry path.
from run_v2 import CoordinatorError, atomic_json, digest, diagnosis, local_path, read_json, source_version

CONFIG = "config/experiment_v3.json"
REMOTE_ROOT = "/content/jev_llm_v3"
CLI_PYTHON = "/root/.local/share/uv/tools/google-colab-cli/bin/python"
CONDITIONS = ("G_SINGLE", "G_GREEDY", "JFINAL", "B13", "B13_GREEDY", "Q9_GREEDY")
SAMPLED = {"G_SINGLE", "JFINAL", "B13"}
NOTEBOOKS = {c: f"{i:02d}_{c}" for i, c in enumerate(CONDITIONS, 1)}
NOTEBOOKS.update(ANALYSIS="07_ANALYSIS", LATENCY="08_LATENCY")
POLL_SECONDS = 25
RECOVERY_SECONDS = 180
LEGACY_JOB_SECONDS = 14400
JOB_SECONDS = 21600
CLEANUP_SECONDS = 600
CONTROLS = (2, 5, 10, 30)
EXPERIMENT_ID = "jev-llm-v3-20261003"
BUDGET_LEDGER = "results/v3/budget_ledger.json"
MAX_GPU_SECONDS = 25 * 3600
MAX_COLLECTION_BYTES = 512 * 1024 * 1024
PRIVATE_AUDITOR_SHA256 = "66f965f110e15155b9eadf18dbfc6481534dbca264f161522763af92e38796e7"
INHERITED_AUDITOR_SHA256 = "3a2299c8da375f215e07eb64da905163abd9f0bec47986c23da2d973e14d67a4"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip() if os.name == "posix" else None


class Blocked(ValueError):
    def __init__(self, state, message):
        super().__init__(message)
        self.state = state


def require(ok, message, state="blocked_integrity"):
    if not ok:
        raise Blocked(state, message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Evidence timestamps require a timezone")
    return parsed.timestamp()


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def job(condition, phase, session=None):
    require(condition in NOTEBOOKS, "Unknown v3 condition")
    require(phase in {"smoke", "pilot", "test", "latency", "analysis"}, "Unknown v3 phase")
    require((condition == "ANALYSIS") == (phase == "analysis"), "Analysis must be a CPU-only phase")
    require(condition != "LATENCY" or phase in {"smoke", "latency"}, "LATENCY is test or explicit dev smoke only")
    require(phase != "latency" or condition == "LATENCY", "Latency phase requires LATENCY")
    smoke = phase == "smoke"
    seeds = "17" if smoke or condition not in SAMPLED | {"LATENCY"} else "17,29,43"
    count = 1 if smoke else 50 if phase == "pilot" else 500
    params = dict(CONDITION=condition, CONFIG_FILE=CONFIG, DATA_SUBDIR="data/v3", ROOT=REMOTE_ROOT,
                  SPLIT="dev" if smoke else "pilot" if phase == "pilot" else "test",
                  RUN_TAG="smoke" if smoke else "pilot_v3" if phase == "pilot" else "confirmatory_v3",
                  N_PROBLEMS=count, SEEDS=seeds, PREFLIGHT_MODE="full", HASH_WEIGHTS=True,
                  REQUIRED_GPU="A100")
    if condition == "LATENCY":
        params.update(LAT_N_ITEMS=2 if smoke else 100, LAT_REPETITIONS=1 if smoke else 3,
                      LAT_CONDITIONS="JFINAL,B13_GREEDY", LATENCY_SWAP_MODE="reload")
    if condition == "ANALYSIS":
        # Only parameters actually declared by the CPU notebook.
        params = {k: v for k, v in params.items() if k in {"CONFIG_FILE", "DATA_SUBDIR", "ROOT", "RUN_TAG", "SPLIT"}}
    params.update(APPROVAL_FILE="", PILOT_REPORT_FILE="", CONFIRMATORY_AUTHORIZED=False)
    session = session or "jev-v3-" + phase + "-" + condition.lower()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", session) and session != "all", "Unsafe session name")
    return dict(key=phase + "/" + condition, session=session, condition=condition, phase=phase,
                count=count, params=params, gpu="CPU" if condition == "ANALYSIS" else "A100",
                expected_rows=4 if smoke and condition == "LATENCY" else 1800 if condition == "LATENCY"
                else count * len(seeds.split(",")),
                deadline_seconds=LEGACY_JOB_SECONDS if phase in {"smoke", "pilot"} else JOB_SECONDS, sharding=False)


def plan(phase="prepared"):
    phases = ("smoke", "pilot", "test", "latency", "analysis") if phase == "full" else ("smoke", "pilot") if phase == "prepared" else (phase,)
    return [job(c, p) for p in phases for c in
            (("LATENCY",) if p == "latency" else ("ANALYSIS",) if p == "analysis"
             else (*CONDITIONS, "LATENCY") if p == "smoke" else CONDITIONS)]


def bundle_tools():
    import make_bundle_v3
    return make_bundle_v3


def dataset_gate(root):
    try:
        return bundle_tools().validate_dataset(root=Path(root))
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise Blocked("blocked_dataset_review", "Full v3 dataset/policy/review seal unavailable: "
                      + (str(error)[:300] if isinstance(error, ValueError) else type(error).__name__)) from None


def config_gate(root):
    root = Path(root)
    cfg = read_json(root / CONFIG)
    require(cfg.get("protocol_version") == "3" and cfg.get("code_version") == "0.3.0"
            and cfg.get("implementation_profile") == "1COPY-G4-VLLM-GRAPH-V3", "Unknown v3 config/profile/version")
    require(set(cfg.get("models", {})) == {"G", "J", "B", "O"}
            and cfg["models"]["O"].get("repo") == "Qwen/Qwen3.5-9B"
            and cfg["models"]["O"].get("revision") == bundle_tools().OFFICIAL_Q9_REVISION, "Missing exact official9 model role")
    require(all(re.fullmatch(r"[0-9a-f]{40}", m.get("revision", "")) for m in cfg["models"].values()), "Mutable model pin")
    require(source_version((root / "src/jevlab/__init__.py").read_text(encoding="utf-8")) == "0.2.1"
            and source_version((root / "src/jevlab/v3/__init__.py").read_text(encoding="utf-8")) == "0.3.0",
            "Unknown historical code version; no legacy fallback")
    return cfg


def pilot_gate(root, report_path, *, fresh=True):
    require(report_path is not None, "Pinned technical pilot GO report required", "blocked_pilot_go")
    report = read_json(report_path)
    require(report.get("protocol_version") == "3" and report.get("code_version") == "0.3.0"
            and report.get("go") is True and report.get("decision") == "go" and report.get("errors") == []
            and report.get("budget_approval") is False, "Pilot is not a technical-only GO", "blocked_pilot_go")
    entries = report.get("conditions", {})
    require(set(entries) == set(CONDITIONS), "Pilot must cover all six physical arms", "blocked_pilot_go")
    for c, entry in entries.items():
        n = 150 if c in SAMPLED else 50
        require(entry.get("verified") is True and entry.get("denominator") == n
                and entry.get("verification", {}).get("ok") is True
                and entry.get("verification", {}).get("condition") == c
                and entry.get("verification", {}).get("records") == n
                and entry.get("rates", {}).get("denominator") == n
                and 0 <= entry["rates"].get("timeout_infra_rate", 1) < .05,
                "Incomplete pilot coverage or technical failure: " + c, "blocked_pilot_go")
    aggregate = report.get("aggregate", {})
    require(aggregate.get("complete") is True and aggregate.get("denominator") == 600
            and aggregate.get("observed_cases") == 600 and 0 <= aggregate.get("timeout_infra_rate", 1) < .05,
            "Incomplete aggregate pilot", "blocked_pilot_go")
    if fresh:
        from pilot_decision_v3 import decide
        current = decide(canonical_pilot(root), Path(root) / "data/v3/pilot_inputs.jsonl",
                         config=Path(root) / CONFIG, prompts=Path(root) / "prompts", run_tag="pilot_v3")
        require(all(current.get(k) == report.get(k) for k in ("go", "conditions", "aggregate", "estimates")),
                "Pilot evidence changed or cannot be freshly verified", "blocked_pilot_go")
    return report


def approval_gate(root, phase, approval, report_path, authorized=False, *, fresh=True):
    require(authorized is True and approval is not None, "User budget approval file and --authorize-confirmatory required",
            "blocked_budget_approval")
    approved = read_json(approval)
    scopes = {"test", "test_and_latency"} if phase == "test" else {"latency", "test_and_latency"} if phase == "latency" else {"test_and_latency"}
    require(approved.get("approved") is True and approved.get("scope") in scopes
            and approved.get("config_sha256") == digest(Path(root) / CONFIG)
            and approved.get("pilot_report_sha256") == digest(report_path)
            and isinstance(approved.get("approved_by"), str) and approved["approved_by"].strip()
            and approved.get("approved_utc") and epoch(approved["approved_utc"]) <= time.time(),
            "Approval must bind this config SHA, actual pilot report SHA, user and scope", "blocked_budget_approval")
    hours = approved.get("max_gpu_hours")
    require(type(hours) in (int, float) and math.isfinite(hours) and 0 < hours <= 25,
            "Approval must specify max_gpu_hours within the accumulated 25-hour ceiling", "blocked_budget_or_eta")
    if fresh and isinstance(approved.get("operational_amendment"), dict):
        install_private_auditor(root, approved)
    report = pilot_gate(root, report_path, fresh=fresh)
    if "operational_amendment" in approved:
        authorization = runtime_approval_gate(root, job("LATENCY", "latency") if phase == "latency"
                                               else job("G_SINGLE", "test"), approval, report_path)
        require(isinstance(authorization.get("operational_amendment"), dict),
                "Actual runtime must validate the operational amendment", "blocked_budget_approval")
    return approved, report


def initial_budget_gate(root, initial_budget):
    require(initial_budget is not None, "Parent-supplied initial budget JSON required for every GPU phase", "blocked_budget_approval")
    value = read_json(initial_budget)
    cfg = config_gate(root)
    require(value.get("approved") is True and value.get("experiment_id") == EXPERIMENT_ID
            and value.get("scope") == "all_gpu_phases_conditional_technical_go"
            and value.get("max_gpu_hours") == 25 and type(value["max_gpu_hours"]) in (int, float)
            and value.get("max_total_assignments") == 2 and value.get("gpu") == "A100"
            and value.get("implementation_profile") == cfg["implementation_profile"]
            and value.get("models") == cfg["models"] and value.get("config_sha256") == digest(Path(root) / CONFIG)
            and value.get("technical_go_required") is True
            and all(isinstance(value.get(k), str) and value[k].strip() for k in ("approved_by", "approved_utc", "authorization_source"))
            and epoch(value["approved_utc"]) <= time.time(),
            "Initial consent must bind actual parent authorization, experiment, profile/models/config, 25 accumulated A100 hours and two assignments",
            "blocked_budget_approval")
    return value


def budget_ledger(root):
    """Import historical consumption, never derive a fresh budget from consent bytes."""
    root = Path(root).resolve()
    path = root / BUDGET_LEDGER
    require(not any(p.is_symlink() for p in (path, path.with_suffix(".json.tmp"), *path.parents)), "Symlink budget ledger forbidden")
    ledger = read_json(path) if path.exists() else dict(experiment_id=EXPERIMENT_ID, max_gpu_seconds=MAX_GPU_SECONDS, jobs={})
    require(ledger.get("experiment_id") == EXPERIMENT_ID and ledger.get("max_gpu_seconds") == MAX_GPU_SECONDS
            and isinstance(ledger.get("jobs"), dict), "Accumulated experiment ledger identity/cap changed", "blocked_budget_or_eta")
    jobs = ledger["jobs"]
    for old in (root / "results/v3").glob("budget_*.json"):
        if not re.fullmatch(r"budget_[0-9a-f]{64}\.json", old.name):
            continue
        previous = read_json(old)
        require(isinstance(previous.get("jobs"), dict), "Prior budget ledger malformed", "blocked_budget_or_eta")
        for run_id, entry in previous["jobs"].items():
            if run_id not in jobs:
                jobs[run_id] = {**entry, "released": False, "legacy_ledger": str(old)}
    for status in (root / "results/v3").rglob("status.json"):
        # Remote mirrors contain notebook-side status, not allocation leases.
        if any(p in {"remote", "pilot_verified", "artifacts", "checkpoints"} for p in status.relative_to(root / "results/v3").parts):
            continue
        state = read_json(status)
        if not state.get("allocation_attempted") or state.get("job", {}).get("gpu") == "CPU":
            continue
        directory = status.parent
        require(state.get("series") == "v3" and state.get("root") == str(root)
                and state.get("output") == str(directory) and isinstance(state.get("run_id"), str),
                "Prior GPU lease identity uncertain", "blocked_budget_or_eta")
        run_id = state["run_id"]
        if run_id not in jobs:
            start = state.get("allocation_epoch", state.get("start_epoch"))
            require(type(start) in (int, float) and math.isfinite(start), "Prior GPU lease clock missing", "blocked_budget_or_eta")
            historical_limit = state["job"].get("deadline_seconds", LEGACY_JOB_SECONDS)
            jobs[run_id] = dict(key=state["job"]["key"], session=state["job"]["session"], output=str(directory),
                reserved_seconds=historical_limit, reserved_epoch=start, allocation_epoch=start,
                deadline_epoch=state.get("deadline_epoch", start + historical_limit), allocation_attempted=True,
                endpoint=state.get("owned_endpoint"), released=False, observed_seconds=0, imported=True)
    for entry in jobs.values():
        require(isinstance(entry, dict) and type(entry.get("reserved_seconds")) in (int, float)
                and math.isfinite(entry["reserved_seconds"]) and 0 < entry["reserved_seconds"] <= JOB_SECONDS
                and type(entry.get("reserved_epoch")) in (int, float) and math.isfinite(entry["reserved_epoch"])
                and isinstance(entry.get("output"), str) and isinstance(entry.get("session"), str),
                "Budget ledger consumption malformed", "blocked_budget_or_eta")
        require(type(entry.get("released")) is bool, "Budget lease release state malformed", "blocked_budget_or_eta")
        for key in ("actual_seconds", "observed_seconds", "allocation_epoch", "deadline_epoch", "observed_epoch", "observed_end_epoch", "allocation_monotonic", "deadline_monotonic"):
            if key in entry:
                require(type(entry[key]) in (int, float) and math.isfinite(entry[key]) and entry[key] >= 0,
                        "Budget observation malformed", "blocked_budget_or_eta")
        require(entry.get("released") is not True or entry.get("release_verified") is True and "actual_seconds" in entry,
                "Budget refund lacks authoritative release evidence", "blocked_budget_or_eta")
    return ledger


def budget_spent(ledger):
    return sum(e["actual_seconds"] if e.get("released") is True else
               max(e["reserved_seconds"], lease_elapsed(e) if e.get("allocation_attempted", True) else 0)
               for e in ledger["jobs"].values())


def lease_elapsed(entry):
    elapsed = max(0, entry.get("observed_seconds", 0), time.time() - entry.get("allocation_epoch", entry["reserved_epoch"]))
    if type(entry.get("allocation_monotonic")) in (int, float):
        if entry.get("clock_boot_id") == BOOT_ID:
            elapsed = max(elapsed, time.monotonic() - entry["allocation_monotonic"])
        else:
            # Lost clock domain cannot grant fresh spending from an unknown interval.
            elapsed = max(elapsed, MAX_GPU_SECONDS)
    return elapsed


def remaining_confirmatory(root, current_intent=None):
    remaining = []
    for item in [*plan("test"), *plan("latency")]:
        directory = Path(root) / "results/v3" / item["key"]
        if not (directory / "status.json").exists():
            remaining.append(item)
            continue
        state = status_gate(root, directory, item)
        if (current_intent is not None and state == current_intent and state.get("allocation_attempted") is False
                and state.get("status") in {"submission_intent", "startup_acknowledged", "validating_local"}):
            remaining.append(item)
            continue
        if state.get("startup_ack") is True and state.get("status") not in {"completed", "failed"}:
            ledger = budget_ledger(root)
            entry = ledger["jobs"].get(state.get("run_id"))
            require(entry and entry.get("released") is not True and state.get("deadline_epoch", 0) > time.time(),
                    "Active confirmatory job has no current accumulated lease", "blocked_budget_or_eta")
            continue  # Already reserved, coordinator resumes observation rather than budgeting it twice.
        require(state.get("status") == "completed" and state.get("verified") is True
                and state.get("released") is True and state.get("completed_execution") is True,
                "Prior confirmatory lease requires reconciliation; no renewed allowance", "blocked_budget_or_eta")
        operator = Operator(root, directory, state)
        operator.deadline = operator.work_deadline = time.time() + 600
        operator.monotonic_deadline = operator.monotonic_work_deadline = time.monotonic() + 600
        operator.verify()
    return remaining


def pilot_inputs(root, report_path):
    """Resolve all six existing public finals; never rewrite the pinned GO report."""
    report = pilot_gate(root, report_path, fresh=False)
    archives = {}
    for condition, entry in report["conditions"].items():
        verification = entry["verification"]
        path = local_path(verification.get("archive", ""))
        path = path if path.is_absolute() else Path(root) / path
        name = str(verification.get("archive", "")).replace("\\", "/").rsplit("/", 1)[-1]
        require(path.is_file() and name == path.name and name.endswith("_final.zip") and "gold" not in name.lower()
                and re.fullmatch(r"[0-9a-f]{64}", verification.get("archive_sha256", ""))
                and digest(path) == verification["archive_sha256"], "Missing/changed public pilot final: " + condition, "blocked_pilot_go")
        archives[condition] = path.resolve()
    require(len({p.name for p in archives.values()}) == 6, "Duplicate pilot final basenames", "blocked_pilot_go")
    return archives


def runtime_approval_gate(root, item, approval, pilot_report):
    """Run the actual consent validator locally; it reads evidence, never starts GPU code."""
    from jevlab.common import load_prompt, sha256_text
    from jevlab.v3.runner import check_budget_approval
    root = Path(root)
    cfg = config_gate(root)
    data_names = ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "schedule.json",
                   "dataset_manifest.json", "latency_inputs.jsonl", "latency_plan.json", "SHA256SUMS")
    package = root / "src/jevlab"
    code_hashes = {path.relative_to(package).as_posix(): digest(path) for path in
                   sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
    context = SimpleNamespace(root=root, data_dir=root / "data/v3",
        cond="LATENCY" if item["condition"] == "LATENCY" else "G_SINGLE",
        p=dict(SPLIT="test", APPROVAL_FILE=str(Path(approval).resolve()), PILOT_REPORT_FILE=str(Path(pilot_report).resolve()),
               CONFIRMATORY_AUTHORIZED=True), manifest={"created_utc": utc()},
        config=dict(experiment_config_sha256=digest(root / CONFIG), models=cfg["models"], profile=cfg["implementation_profile"],
                    code_sha256=code_hashes,
                    data_sha256={n: digest(root / "data/v3" / n) for n in data_names if (root / "data/v3" / n).is_file()},
                    prompt_sha256={n: sha256_text(load_prompt(root / "prompts" / (n + ".txt"))) for n in
                                   ("generador", "criterio_paso", "criterio_final")}))
    try:
        return check_budget_approval(context)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise Blocked("blocked_budget_approval", "Runtime consent/evidence binding failed: " + type(error).__name__) from None


def phase_calendar(n_items=100, block_size=10, seeds=(17, 29, 43), repetitions=3):
    """Execution order shared by the ETA and fake-runtime calendar checks."""
    for repetition in range(repetitions):
        for block, start in enumerate(range(0, n_items, block_size)):
            for phase in (("B", "H") if (block + repetition) % 2 == 0 else ("H", "B")):
                for seed in seeds:
                    for index in range(start, min(start + block_size, n_items)):
                        yield repetition, block, phase, seed, index


def operational_calendar(root, approved):
    """Use the runtime's exact blind calendar, not an ETA-only residency shortcut."""
    from jevlab.v3.operational import build_calendar, validate_calendar, POLICY_ID, MAX_NEW_JOB_SECONDS, TOTAL_GPU_SECONDS
    policy = approved.get("operational_amendment")
    require(isinstance(policy, dict) and policy.get("id") == POLICY_ID
            and policy.get("max_job_seconds") == JOB_SECONDS == MAX_NEW_JOB_SECONDS
            and policy.get("max_gpu_seconds") == MAX_GPU_SECONDS == TOTAL_GPU_SECONDS,
            "Unknown operational policy or increased limits", "blocked_budget_approval")
    data = Path(root) / "data/v3"
    require(policy.get("latency_source_plan_sha256") == digest(data / "latency_plan.json")
            and policy.get("latency_schedule_sha256") == digest(data / "schedule.json"),
            "Operational source plan/schedule changed", "blocked_budget_approval")
    items = [json.loads(line) for line in (data / "latency_inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    order = read_json(data / "schedule.json")["order"]
    metadata = {row["item_id"]: row for row in read_json(data / "latency_plan.json")["items"]}
    try:
        calendar = build_calendar(items, order, metadata)
        validate_calendar(calendar, items, order, metadata, policy)
    except (ValueError, KeyError, TypeError) as error:
        raise Blocked("blocked_budget_approval", "Operational calendar binding failed: " + type(error).__name__) from None
    return calendar


def phase_bundle_path(root, item, approved=None, dev=False):
    directory = Path(root) / "dist/v3"
    if item["phase"] in {"test", "latency", "analysis"} and isinstance((approved or {}).get("operational_amendment"), dict):
        directory /= "postpilot"
    stem = "jev_llm_v3" + ("_dev" if dev else "")
    return directory / (stem + ("_analysis_bundle.zip" if item["condition"] == "ANALYSIS" else "_bundle.zip"))


def private_auditor_files(root, approved):
    """Exact code selection for newly approved amended jobs, never historical jobs."""
    root = Path(root)
    files = {"tools/verify_run_v3.py": approved.get("new_verifier_sha256"),
             "tools/audit_native_container_v3.py": PRIVATE_AUDITOR_SHA256,
             "tools/audit_inherited_v3.py": INHERITED_AUDITOR_SHA256}
    require(approved.get("approved") is True and isinstance(approved.get("operational_amendment"), dict)
            and re.fullmatch(r"[0-9a-f]{64}", approved.get("new_verifier_sha256", "")),
            "Approved current verifier/V2 auditor SHA mismatch", "blocked_budget_approval")
    for name, sha in files.items():
        path = root / name
        require(not any(p.is_symlink() for p in (path, *path.parents)) and digest(path) == sha,
                "Approved current verifier/V2 auditor SHA mismatch", "blocked_budget_approval")
    return files


def install_private_auditor(root, approved):
    """Bind fresh amended audits in this interpreter, including detached workers."""
    files = private_auditor_files(root, approved)
    expected = dict(AUDITOR_ID="jev-v3-native-container-provenance-audit", AUDITOR_VERSION="2.0.0",
                    AUDITOR_SHA256=files["tools/audit_native_container_v3.py"],
                    VERIFIER_SHA256=files["tools/verify_run_v3.py"],
                    INHERITED_AUDITOR_SHA256=files["tools/audit_inherited_v3.py"])
    module = sys.modules.get("verify_run_v3")
    if not (all(getattr(module, k, None) == v for k, v in expected.items())
            and Path(getattr(module, "__file__", "")).resolve() == (Path(root) / "tools/verify_run_v3.py").resolve()):
        # Load exact local helpers, not stale imports from another root/checker.
        for name in ("audit_inherited_v3", "audit_native_container_v3"):
            spec = importlib.util.spec_from_file_location(name, Path(root) / "tools" / (name + ".py"))
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            require(helper.AUDITOR_SHA256 == files["tools/" + name + ".py"],
                    "Loaded helper identity mismatch", "blocked_budget_approval")
            sys.modules[name] = helper
        require(helper.VERIFIER_SHA256 == expected["VERIFIER_SHA256"]
                and helper.INHERITED_AUDITOR_SHA256 == expected["INHERITED_AUDITOR_SHA256"],
                "Loaded helper checker/inherited identity mismatch", "blocked_budget_approval")
        module = helper.load_verifier()
        require(all(getattr(module, k, None) == v for k, v in expected.items()),
                "Loaded auditor identity mismatch", "blocked_budget_approval")
        sys.modules["verify_run_v3"] = module
    # pilot_decision imports function bindings, so replacing sys.modules alone is insufficient.
    if "pilot_decision_v3" in sys.modules:
        importlib.reload(sys.modules["pilot_decision_v3"])
    return module


def estimate_jobs(root, jobs, approved, report):
    """Serial GPU seconds: preparation once, lifecycle only on actual phase changes."""
    estimates, artifacts = {}, {}
    for item in jobs:
        c = item["condition"]
        if c == "ANALYSIS":
            continue
        arms = ("JFINAL", "B13_GREEDY") if c == "LATENCY" else (c,)
        hot = setup = 0.0
        timings = {}
        for arm in arms:
            entry = report["conditions"][arm]
            amount = entry.get("latency_estimate" if c == "LATENCY" else "estimate", {}).get("estimated_test_walltime_s")
            require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0,
                    "Pilot lacks a measured ETA for " + c, "blocked_budget_or_eta")
            from verify_run_v3 import read_archive
            archive = Path(entry["verification"]["archive"])
            if not archive.is_absolute():
                archive = Path(root) / archive
            key = (archive.resolve(), entry["verification"]["archive_sha256"])
            if key not in artifacts:
                require(digest(archive) == key[1], "Pilot timing archive changed")
                snapshot = read_archive(archive)
                require(snapshot["archive_sha256"] == key[1], "Pilot timing archive changed during read")
                stages = snapshot["manifest"].get("stages", {})
                names = ("environment", "snapshots", "preflight_pre_engines", "engines", "preflight_post_engines", "warmup")
                values = {name: stages.get(name, {}).get("seconds") for name in names}
                require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())
                        and all(stages.get(name, {}).get("ok") is True for name in names),
                        "Pilot lacks measured setup/loading stages", "blocked_budget_or_eta")
                artifacts[key] = values  # Immutable timing evidence, only within this ETA call.
            values = artifacts[key]
            timings[arm] = values
            if c != "LATENCY":
                setup += sum(values.values())
            hot += amount
        if c == "LATENCY":
            phases, previous = Counter(), None
            calendar = (("B" if row["condition"] == "B13_GREEDY" else "H"
                         for row in operational_calendar(root, approved)) if "operational_amendment" in approved
                        else (row[2] for row in phase_calendar()))
            for phase in calendar:
                if phase != previous:
                    phases[phase] += 1
                    previous = phase
            # Environment/native reference once; disjoint G+J and B snapshots once.
            setup = sum(max(t[k] for t in timings.values()) for k in ("environment", "preflight_pre_engines"))
            setup += sum(t["snapshots"] for t in timings.values())
            setup += sum(phases[phase] * sum(timings[arm][k] for k in ("engines", "preflight_post_engines", "warmup"))
                         for phase, arm in (("B", "B13_GREEDY"), ("H", "JFINAL")))
        seconds = 1.5 * (hot + setup) + 900 + CLEANUP_SECONDS
        limit = item["deadline_seconds"] if "operational_amendment" in approved else LEGACY_JOB_SECONDS
        require(seconds <= limit, "Unsharded job " + c + " exceeds approved job deadline with measured setup and 50% margin; sharding unapproved",
                "blocked_budget_or_eta")
        estimates[item["key"]] = seconds
    require(sum(estimates.values()) / 3600 <= approved["max_gpu_hours"],
            "Measured safety-margin ETA exceeds approved GPU-hour budget", "blocked_budget_or_eta")
    return estimates


def notebook_gate(root, item):
    directory = Path(root) / "notebooks/v3"
    if item["phase"] in {"test", "latency", "analysis"}:
        directory /= "postpilot"
    path = directory / (NOTEBOOKS[item["condition"]] + ".ipynb")
    nb = read_json(path)
    import build_notebooks_v3 as builder
    expected = builder.analysis_notebook() if item["condition"] == "ANALYSIS" else builder.condition_notebook(item["condition"])
    for i, cell in enumerate(expected.cells):
        cell.id = f"v3-cell-{i:02d}"
    require(nb == json.loads(builder.nbf.writes(expected)), "Actual notebook is stale/changed; rebuild v3 notebooks before allocation")
    cells = [c for c in nb["cells"] if "parameters" in c.get("metadata", {}).get("tags", [])]
    require(len(cells) == 1, "Notebook must have one parameters cell")
    source = cells[0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    names = set()
    defaults = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    try:
                        defaults[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    require(all(defaults.get(k) == v for k, v in {"CONFIG_FILE": CONFIG, "DATA_SUBDIR": "data/v3", "ROOT": REMOTE_ROOT}.items()),
            "Notebook root/config/data mismatch")
    require(set(item["params"]) <= names, "Notebook lacks requested v3 parameters")
    require(defaults.get("CONDITION", item["condition"]) == item["condition"], "Notebook physical condition mismatch")
    if item["condition"] == "ANALYSIS":
        require("APPROVAL_FILE" in names, "Analysis requires forwarded approval sentinel")
    return path


def smoke_retry_gate(root, prefix):
    require(isinstance(prefix, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", prefix),
            "Unsafe infrastructure smoke attempt prefix", "blocked_ownership")
    ledger, predecessors, infrastructure_failure = budget_ledger(root), {}, False
    previous = [(Path(root) / "results/v3" / item["key"], item, False) for item in plan("smoke")]
    for path in (Path(root) / "results/v3/attempts").glob("*/smoke/*/status.json"):
        if path.parents[2].name == prefix:
            continue  # Same namespace resumes existing jobs; start_operator never overwrites them.
        state = read_json(path)
        item = state.get("job", {})
        require(item.get("phase") == "smoke" and item == job(item["condition"], "smoke", item["session"]),
                "Previous attempt identity changed", "blocked_ownership")
        previous.append((path.parent, item, True))
    for directory, item, namespaced in previous:
        if not (directory / "status.json").exists():
            continue
        state = status_gate(root, directory, item)
        require(state.get("status") in {"failed", "completed"} and state.get("released") is True,
                "Canonical smoke must be reconciled before infrastructure retry", "blocked_ownership")
        predecessors[directory.relative_to(Path(root) / "results/v3").as_posix()] = digest(directory / "status.json")
        require(not namespaced or state.get("status") == "failed", "Successful diagnostic smoke must not be rerun under a new prefix", "blocked_execution")
        if state.get("allocation_attempted"):
            entry = ledger["jobs"].get(state.get("run_id"), {})
            require(entry.get("released") is True and entry.get("release_verified") is True
                    and entry.get("release_evidence") == "backend_absent" and entry.get("session") == item["session"]
                    and entry.get("output") == str(directory.resolve()),
                    "Prior smoke GPU charge/release unproved", "blocked_budget_or_eta")
            if state.get("status") == "failed":
                health = state.get("health", {})
                parameter_failure = False
                notebook = directory / "remote/notebooks" / (NOTEBOOKS[item["condition"]] + ".out." + state["run_id"] + ".ipynb")
                if notebook.is_file() and health.get("execution", {}).get("rc") == 1:
                    raw = read_json(notebook)
                    params = raw.get("metadata", {}).get("papermill", {}).get("parameters", {})
                    cells = [c for c in raw["cells"] if c.get("cell_type") == "code"]
                    errors = [(i, o) for i, c in enumerate(cells) for o in c.get("outputs", []) if o.get("output_type") == "error"]
                    if len(errors) == 1:
                        index, error = errors[0]
                        parameter_failure = (params.get("SPLIT") == "dev" and params.get("RUN_TAG") == "smoke"
                            and type(params.get("SEEDS")) is int and params["SEEDS"] == 17
                            and error.get("ename") == "ValueError"
                            and error.get("evalue") == "Explicit dev smoke requires RUN_TAG=smoke and SEEDS=17"
                            and error["evalue"] in "".join(cells[index]["source"])
                            and any(c.get("execution_count") is None for c in cells[index + 1:])
                            and all(c.get("execution_count") is None and not c.get("outputs") for c in cells[index + 1:])
                            and not any((directory / "artifacts").glob("*_final.zip")))
                require(state.get("remote_startup_ack") is True
                         and health.get("known") is True
                         and health.get("run_id") == state["run_id"]
                         and (state.get("error_type") == "TimeoutExpired"
                              and (state.get("completed_execution") is False and health.get("alive") is True
                              or state.get("collection_error_type") == "TimeoutExpired"
                              and health.get("alive") is False and health.get("execution", {}).get("done") is True
                              and health["execution"].get("rc") == 0)
                              or parameter_failure and state.get("completed_execution") is False
                              and health.get("alive") is False and health.get("execution", {}).get("done") is True),
                         "Only interrupted infrastructure or proven pre-model parameter smoke is retryable; no quality retry", "blocked_execution")
                infrastructure_failure = True
    require(infrastructure_failure, "No released infrastructure-failed dev smoke to retry", "blocked_execution")
    return dict(attempt_prefix=prefix, development_only=True, predecessors=predecessors)


def preflight(root, item, *, approval=None, pilot_report=None, initial_budget=None, authorize_initial=False,
              authorize_confirmatory=False, dev_bundle=False, fresh_pilot=True, current_intent=None, attempt_prefix=None):
    root = Path(root).resolve()
    require(item == job(item["condition"], item["phase"], item["session"])
            and item["params"].get("CONFIRMATORY_AUTHORIZED") is False
            and item["params"].get("APPROVAL_FILE") == "" and item["params"].get("PILOT_REPORT_FILE") == "",
             "Planner controls/parameters must remain canonical", "blocked_budget_approval")
    retry_lineage = None
    if attempt_prefix is not None:
        require(item["phase"] == "smoke" and authorize_initial is True
                and item["session"] == job(item["condition"], "smoke")["session"] + "-" + attempt_prefix,
                "Attempt prefixes are only for explicitly authorized diagnostic dev smoke", "blocked_user_authorization")
        retry_lineage = smoke_retry_gate(root, attempt_prefix)
    if item["phase"] in {"smoke", "pilot"}:
        require(approval is None and pilot_report is None and (authorize_confirmatory is False or authorize_confirmatory is None),
                "Initial phases must not carry confirmatory consent options", "blocked_budget_approval")
    config_gate(root)
    # Even dev GPU launches need actual sealed data/policy/reviews. Building dev
    # notebooks/bundles while unsealed is a separate, strictly local operation.
    dataset_gate(root)
    require(not dev_bundle or item["phase"] == "smoke", "Development bundles are never test/pilot/analysis evidence")
    eta = None
    authorization, pilot_archives, approved = None, {}, None
    if item["phase"] in {"smoke", "pilot"}:
        require(authorize_initial is True, "Initial GPU work requires --authorize-initial", "blocked_user_authorization")
    else:
        approved, report = approval_gate(root, item["phase"], approval, pilot_report, authorize_confirmatory, fresh=fresh_pilot)
        require(isinstance(approved.get("operational_amendment"), dict),
                "New post-pilot jobs require the approved operational amendment", "blocked_budget_approval")
        eta = estimate_jobs(root, [item], approved, report).get(item["key"])
    if item["gpu"] != "CPU":
        initial = initial_budget_gate(root, initial_budget)
        if item["phase"] in {"test", "latency"}:
            require(approved.get("experiment_id") == EXPERIMENT_ID and approved["max_gpu_hours"] <= 25
                    and approved.get("initial_authorization_sha256") == hashlib.sha256(
                        json.dumps(initial, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                    "Post-pilot parent confirmation must bind initial accumulated experiment authorization", "blocked_budget_approval")
            generated = report.get("meta", {}).get("generated_utc")
            require(isinstance(generated, str) and epoch(approved["approved_utc"]) >= epoch(generated),
                    "Parent confirmation must be recorded after the actual immutable pilot decision", "blocked_budget_approval")
        ledger = budget_ledger(root)
        base = root / "results/v3"
        existing = (base / "attempts" / attempt_prefix if attempt_prefix is not None else base) / item["key"] / "status.json"
        completed = existing.exists() and status_gate(root, existing.parent, item).get("status") == "completed"
        require(completed or budget_spent(ledger) < MAX_GPU_SECONDS, "Accumulated GPU budget exhausted", "blocked_budget_or_eta")
        if item["phase"] in {"test", "latency"}:
            estimates = estimate_jobs(root, remaining_confirmatory(root, current_intent), approved, report)
            require(sum(estimates.values()) <= min(MAX_GPU_SECONDS, approved["max_gpu_hours"] * 3600) - budget_spent(ledger),
                    "Full remaining test matrix + LATENCY exceeds accumulated remaining budget", "blocked_budget_or_eta")
    bundle = phase_bundle_path(root, item, approved, dev_bundle)
    man = bundle_tools().inspect_bundle(bundle, root=root, check_local=True)
    require(man.get("series") == "v3" and man.get("code_version") == "0.3.0"
            and man.get("legacy_code_version") == "0.2.1" and man.get("config_sha256") == digest(root / CONFIG)
            and man.get("contains_gold") is (item["condition"] == "ANALYSIS")
            and man.get("development_only") is dev_bundle, "Bundle identity/visibility/version mismatch")
    notebook = notebook_gate(root, item)
    if item["phase"] not in {"smoke", "pilot"}:
        pilot_archives = pilot_inputs(root, pilot_report)
        authorization = runtime_approval_gate(root, item, approval, pilot_report)
        require(man.get("operational_policy_id") == approved["operational_amendment"]["id"]
                and man.get("operational_module_sha256") == digest(root / "src/jevlab/v3/operational.py")
                and {n.removeprefix("src/jevlab/"): h for n, h in man["files"].items()
                     if n.startswith("src/jevlab/") and n.endswith(".py")} == approved["operational_amendment"]["new_code_sha256"],
                "Post-pilot bundle must bind the actual approved runtime inventory/policy", "blocked_budget_approval")
    files = {str(bundle.relative_to(root)): digest(bundle), str(notebook.relative_to(root)): digest(notebook)}
    for name in ("tools/colab/run_v3.py", "tools/colab/launch_v3.sh", "tools/colab/operator_v3.sh",
                 "tools/colab/session_guard.sh", "tools/colab/session_guard.py", "tools/verify_run_v3.py",
                 "tools/pilot_decision_v3.py", "tools/make_bundle_v3.py", "tools/build_notebooks_v3.py"):
        files[name] = digest(root / name)
    if approved is not None:
        files["src/jevlab/v3/operational.py"] = digest(root / "src/jevlab/v3/operational.py")
    if item["phase"] in {"test", "latency"}:
        files.update(private_auditor_files(root, approved))
    for evidence in (approval, pilot_report, initial_budget):
        if evidence is not None:
            files[str(Path(evidence).resolve())] = digest(evidence)
    for path in pilot_archives.values():
        files[str(path)] = digest(path)
    return dict(bundle=str(bundle), notebook=str(notebook), bundle_manifest=man, frozen_files=files, eta_seconds=eta,
                **({"infrastructure_retry": retry_lineage} if retry_lineage is not None else {}),
                initial_budget=str(Path(initial_budget).resolve()) if initial_budget else None,
                 authorization=authorization, pilot_archives={c: str(p) for c, p in pilot_archives.items()},
                 **({"operational_policy_sha256": authorization["operational_policy_sha256"]} if approved is not None else {}))


def host_gate(ack_path, root=ROOT, cleanup=False, job_seconds=JOB_SECONDS):
    require(ack_path is not None and os.name == "posix" and Path("/proc").is_dir(),
             "Known live WSL/Windows host and --host-ack file required", "blocked_host_identity")
    ack = read_json(ack_path)
    distro = os.environ.get("WSL_DISTRO_NAME")
    try:
        probed, issued, expires = (epoch(ack.get(k, "")) for k in ("probed_utc", "issued_utc", "expires_utc"))
    except (ValueError, TypeError, AttributeError):
        raise Blocked("blocked_host_identity", "Host receipt timestamps are invalid") from None
    require(ack.get("schema_version") == 1 and ack.get("acknowledged") is True and distro
            and ack.get("wsl_distro") == distro and ack.get("keep_wsl_alive") is True and ack.get("keep_host_awake") is True
            and ack.get("state") in ({"active", "draining"} if cleanup else {"active"})
            and (cleanup or ack.get("accept_allocations") is True)
            and re.fullmatch(r"[0-9a-f]{32}", ack.get("host_lease_id", ""))
            and all(type(ack.get(k)) is int and ack[k] > 0 for k in
                    ("host_process_id", "caretaker_process_id", "caretaker_pid", "power_request_return"))
            and ack["power_request_return"] <= 2 ** 32 - 1
            and 0 <= time.time() - probed <= 90
            and expires >= time.time() + (0 if cleanup else job_seconds)
            and 0 < expires - issued <= 24 * 3600,
            "Fresh live helper/caretaker receipt and allocation admission required; no renewal or UNKNOWN power", "blocked_host_identity")
    script = Path(root) / "tools/colab/host_v3.ps1"
    require(Path(ack.get("host_script_wsl", "")).resolve() == script.resolve()
            and digest(script) == ack.get("host_script_sha256"), "Host helper source identity changed", "blocked_host_identity")
    probe = f'''$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$p=Get-Process -Id {ack['host_process_id']}; $w=Get-Process -Id {ack['caretaker_process_id']}
$pc=Get-CimInstance Win32_Process -Filter 'ProcessId = {ack['host_process_id']}'
$wc=Get-CimInstance Win32_Process -Filter 'ProcessId = {ack['caretaker_process_id']}'
@{{host_id=(Get-CimInstance Win32_ComputerSystemProduct).UUID; host_name=$p.ProcessName; host_start=$p.StartTime.ToUniversalTime().ToString('o'); host_command=$pc.CommandLine;
caretaker_name=$w.ProcessName; caretaker_start=$w.StartTime.ToUniversalTime().ToString('o'); caretaker_command=$wc.CommandLine}} | ConvertTo-Json -Compress
'''
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                             base64.b64encode(probe.encode("utf-16-le")).decode("ascii")],
                            capture_output=True, text=True, encoding="utf-8", timeout=20)
    require(result.returncode == 0, "Windows CIM/process identity unavailable", "blocked_host_identity")
    try:
        identity = json.loads(result.stdout)
        machine = identity["host_id"].lower()
        require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", machine)
                and machine not in {"00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff"}
                and ack.get("host_id", "").lower() == machine
                and identity["host_name"].lower() in {"powershell", "pwsh"} and identity["caretaker_name"].lower() == "wsl"
                and epoch(identity["host_start"]) == epoch(ack["host_process_started_utc"])
                and epoch(identity["caretaker_start"]) == epoch(ack["caretaker_process_started_utc"])
                and ack["host_script"].lower() in identity["host_command"].lower()
                and ack["host_lease_id"] in identity["caretaker_command"], "Dead/reused/foreign Windows host or caretaker process", "blocked_host_identity")
        note = read_json(Path(ack["caretaker_note_wsl"]))
        stat = Path(f"/proc/{ack['caretaker_pid']}/stat").read_text().rsplit(")", 1)[1].split()
        require(note.get("host_lease_id") == ack["host_lease_id"] and note.get("wsl_distro") == distro
                and note.get("pid") == ack["caretaker_pid"] and str(note.get("starttime")) == ack["caretaker_starttime"]
                and stat[0] not in {"Z", "X"} and stat[19] == ack["caretaker_starttime"],
                "Owned Linux caretaker readiness/liveness unproved", "blocked_host_identity")
        margin = 0 if cleanup else job_seconds + 90
        require(type(note.get("deadline_monotonic")) in (int, float) and math.isfinite(note["deadline_monotonic"])
                and note.get("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                and type(ack.get("caretaker_deadline_monotonic")) in (int, float)
                and abs(note["deadline_monotonic"] - ack["caretaker_deadline_monotonic"]) < .1
                and note["deadline_monotonic"] - time.monotonic() >= margin
                and type(ack.get("host_remaining_seconds")) in (int, float)
                and math.isfinite(ack["host_remaining_seconds"]) and ack["host_remaining_seconds"] >= margin,
                "Monotonic caretaker/host custody cannot cover work; wall-clock changes grant no new allowance", "blocked_host_identity")
    except (KeyError, TypeError, AttributeError, OSError, IndexError, json.JSONDecodeError):
        raise Blocked("blocked_host_identity", "Host/caretaker identity response unavailable or invalid") from None
    return dict(host_id=machine, wsl_distro=distro, host_lease_id=ack["host_lease_id"], accept_allocations=ack.get("accept_allocations") is True,
                acknowledgement_sha256=digest(ack_path), verified_utc=utc())


@contextmanager
def flock(path, *, wait=0):
    import fcntl
    until = time.monotonic() + wait
    with Path(path).open("a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < until, "Existing operator/allocation lock; no relaunch", "blocked_ownership")
                time.sleep(.25)
        yield stream.fileno()


IDENTITY_CODE = '''import json, logging, os, sys
logging.disable(logging.CRITICAL)
try:
    from colab_cli.common import state
    state.config_path = sys.argv[1]
    state.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
    client = state.client
    request = client.session.request
    def bounded(method, url, **kwargs):
        kwargs["timeout"] = (5, 15)
        return request(method, url, **kwargs)
    client.session.request = bounded
    sessions = state.store.list()
    if set(sessions) - {sys.argv[2]}:
        raise ValueError("Unowned local session")
    session = sessions.get(sys.argv[2])
    rows = [{"endpoint": a.endpoint, "accelerator": a.accelerator.value} for a in client.list_assignments()]
    print(json.dumps({"local_endpoint": session.endpoint if session else None, "assignments": rows}))
except Exception:
    sys.exit(1)
'''

RELEASE_CODE = '''import importlib.util, json, logging, os, pathlib, sys
logging.disable(logging.CRITICAL)
try:
    from colab_cli.common import state
    state.config_path = sys.argv[1]
    state.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
    client = state.client
    request = client.session.request
    def bounded(method, url, **kwargs):
        kwargs["timeout"] = (5, 15)
        return request(method, url, **kwargs)
    client.session.request = bounded
    session, endpoint = sys.argv[2:4]
    # Explicit endpoint API: session_guard.main's rollover must not turn a
    # replaced local record into permission to release an unknown assignment.
    class OwnedStore:
        def remove(self, name):
            current = state.store.get(name)
            if current and current.endpoint == endpoint:
                state.store.remove(name)
    spec = importlib.util.spec_from_file_location("v3_guard_release", sys.argv[4])
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    ok = guard.release(client, OwnedStore(), session, endpoint, pathlib.Path(sys.argv[5]))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
'''


def bounded_call(command, *, timeout, log=None, capture=False):
    """Terminate only the local CLI process group; never a remote kernel or guard."""
    stream = Path(log).open("a", encoding="utf-8") if log else None
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE if capture else stream,
                              stderr=subprocess.PIPE if capture else subprocess.STDOUT,
                              text=True, start_new_session=(os.name == "posix")) as process:
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except BaseException:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.communicate(timeout=5)
                raise
            return process.returncode, stdout or ""
    finally:
        if stream:
            stream.close()


class Backend:
    def __init__(self, output, session, python=CLI_PYTHON):
        self.out, self.session, self.python = Path(output), session, python
        self.config = self.out / "sessions.json"
        self.cli = str(Path(python).with_name("colab"))

    def identity(self, timeout=60):
        rc, text = bounded_call([self.python, "-c", IDENTITY_CODE, str(self.config), self.session], timeout=min(60, timeout), capture=True)
        require(rc == 0, "Backend identity uncertain (credentials/details suppressed)", "blocked_backend")
        try:
            data = json.loads(text)
            rows = data["assignments"]
            require(set(data) == {"local_endpoint", "assignments"} and isinstance(rows, list)
                    and all(set(r) == {"endpoint", "accelerator"} and isinstance(r["endpoint"], str)
                            and r["endpoint"] and isinstance(r["accelerator"], str) for r in rows)
                    and len({r["endpoint"] for r in rows}) == len(rows)
                    and (data["local_endpoint"] is None or isinstance(data["local_endpoint"], str)), "Invalid backend identity schema")
            return data
        except (ValueError, KeyError, TypeError):
            raise Blocked("blocked_backend", "Backend response invalid (details suppressed)") from None

    def call(self, args, *, timeout=120, capture=False):
        require(0 < timeout <= 120, "CLI calls must be bounded to 120 seconds")
        require(not any(a == "--config" or a.startswith("--config=") for a in args), "Scoped config cannot be overridden")
        return bounded_call([self.cli, "--config", str(self.config), *args], timeout=timeout,
                            capture=capture, log=None if capture else self.out / "cli.log")


def bootstrap_code(item, evidence, marker, approval_name=None, finals=(), work_seconds=None):
    """One-shot submission. A lost acknowledgement is never permission to repeat it."""
    params = dict(item["params"])
    work_seconds = item["deadline_seconds"] - CLEANUP_SECONDS if work_seconds is None else work_seconds
    require(0 < work_seconds <= item["deadline_seconds"] - CLEANUP_SECONDS, "Bootstrap exceeds phase work allowance")
    require(params.get("CONFIRMATORY_AUTHORIZED") is False and params.get("APPROVAL_FILE") == ""
            and params.get("PILOT_REPORT_FILE") == "", "Planner cannot inject confirmation or consent paths", "blocked_budget_approval")
    confirmatory = item["phase"] in {"test", "latency", "analysis"}
    require(bool(approval_name) == confirmatory, "Confirmatory bootstrap requires bound user approval; initial runs must remain unapproved",
            "blocked_budget_approval")
    pilot_archives = {}
    params["BUNDLE_FILE"] = "/content/" + Path(evidence["bundle"]).name
    if approval_name:
        authorization = evidence.get("authorization") or {}
        paths = evidence.get("pilot_archives", {})
        require(authorization.get("required") is True and set(paths) == set(CONDITIONS)
                and set(authorization.get("pilot_archive_sha256", {})) == set(CONDITIONS)
                and all(re.fullmatch(r"[0-9a-f]{64}", authorization.get(k, "")) for k in ("approval_sha256", "pilot_report_sha256")),
                "Runtime-validated approval and all six actual pilot finals required", "blocked_budget_approval")
        for condition, value in paths.items():
            path = Path(value)
            require(digest(path) == authorization["pilot_archive_sha256"][condition], "Pilot final changed before bootstrap")
            pilot_archives[path.name] = authorization["pilot_archive_sha256"][condition]
        require(len(pilot_archives) == 6, "Duplicate pilot evidence basenames")
        params["APPROVAL_FILE"] = REMOTE_ROOT + "/budget_approval.json"
        params["PILOT_REPORT_FILE"] = REMOTE_ROOT + "/pilot_go.json"
        params["CONFIRMATORY_AUTHORIZED"] = True
    arguments = [v for k, value in params.items() for v in ("-r" if isinstance(value, str) else "-p", k, str(value))]
    notebook = Path(evidence["notebook"])
    notebook_sha = evidence.get("frozen_files", {}).get(str(notebook))
    if notebook_sha is None:
        directory = "notebooks/v3/postpilot/" if item["phase"] in {"test", "latency", "analysis"} else "notebooks/v3/"
        notebook_sha = evidence.get("frozen_files", {}).get(directory + notebook.name)
    require(notebook_sha is not None and digest(notebook) == notebook_sha, "Frozen local notebook hash unavailable/changed")
    payload = dict(marker=marker, bundle=Path(evidence["bundle"]).name, bundle_sha256=digest(evidence["bundle"]),
                   notebook=notebook.name, notebook_sha256=notebook_sha, condition=item["condition"], args=arguments,
                    approval=approval_name, finals=list(finals), pilot_archives=pilot_archives, work_seconds=work_seconds,
                    initial_budget_sha256=digest(evidence["initial_budget"]) if evidence.get("initial_budget") else None,
                    authorization=evidence.get("authorization"))
    return '''import hashlib, json, os, pathlib, subprocess, sys, zipfile
P = ''' + repr(payload) + '''
notebook = pathlib.Path('/content') / P['notebook']
if hashlib.sha256(notebook.read_bytes()).hexdigest() != P['notebook_sha256']:
    raise RuntimeError('Uploaded notebook changed')
root = pathlib.Path('/content/jev_llm_v3')
alias = pathlib.Path('/content/jev_llm')
if root.exists() or root.is_symlink() or alias.exists() or alias.is_symlink():
    raise RuntimeError('Existing workspace: do not alter or restart remote work')
root.mkdir()
(root / 'launch_v3.json').write_text(json.dumps(P['marker']))
# Compatibility alias only on this newly allocated, recorded endpoint. Collection
# never uses legacy helpers, which assume a different results directory.
alias.symlink_to(root, target_is_directory=True)
bundle = pathlib.Path('/content') / P['bundle']
if hashlib.sha256(bundle.read_bytes()).hexdigest() != P['bundle_sha256']:
    raise RuntimeError('Uploaded bundle changed')
with zipfile.ZipFile(bundle) as z:
    names = z.namelist()
    m = json.loads(z.read('BUNDLE_MANIFEST.json'))
    if (len(names) != len(set(names)) or set(names) != set(m['files']) | {'BUNDLE_MANIFEST.json'}
            or m.get('series') != 'v3' or m.get('code_version') != '0.3.0'
            or m.get('legacy_code_version') != '0.2.1'):
        raise RuntimeError('Unknown/unsealed bundle')
    for name, expected in m['files'].items():
        p = pathlib.PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\\\' in name or ':' in name:
            raise RuntimeError('Unsafe bundle member')
        if hashlib.sha256(z.read(name)).hexdigest() != expected:
            raise RuntimeError('Bundle member hash mismatch')
    z.extractall(root)
notebooks = root / 'notebooks'; notebooks.mkdir()
os.replace('/content/' + P['notebook'], notebooks / P['notebook'])
if P['initial_budget_sha256']:
    initial = pathlib.Path('/content/jev_v3_initial_budget.json')
    if hashlib.sha256(initial.read_bytes()).hexdigest() != P['initial_budget_sha256']:
        raise RuntimeError('Uploaded initial parent authorization changed')
    os.replace(initial, root / 'initial_budget.json')
if P['approval']:
    for path, key in ((pathlib.Path('/content') / P['approval'], 'approval_sha256'),
                      (pathlib.Path('/content/jev_v3_pilot_go.json'), 'pilot_report_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != P['authorization'][key]:
            raise RuntimeError('Uploaded user consent or pinned pilot report changed')
    os.replace('/content/' + P['approval'], root / 'budget_approval.json')
    os.replace('/content/jev_v3_pilot_go.json', root / 'pilot_go.json')
if P['pilot_archives']:
    incoming = root / 'incoming'; incoming.mkdir()
    for name, expected in P['pilot_archives'].items():
        path = pathlib.Path('/content') / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Uploaded pilot final changed')
        os.replace(path, incoming / name)
if P['finals']:
    incoming = root / 'analysis_incoming'; incoming.mkdir()
    for name in P['finals']:
        os.replace('/content/' + name, incoming / name)
run_id = P['marker']['run_id']
output = notebooks / (pathlib.Path(P['notebook']).stem + '.out.' + run_id + '.ipynb')
command = [sys.executable, '-m', 'papermill', str(notebooks / P['notebook']), str(output),
           '--log-output', '--request-save-on-cell-execute', '--autosave-cell-every', '60', *P['args']]
driver = root / 'driver_v3.py'
driver.write_text("import json, pathlib, subprocess, sys, time\\nroot = pathlib.Path('/content/jev_llm_v3')\\n"
    + "command = " + repr(command) + "\\n"
    + "deadline = time.monotonic() + " + repr(P['work_seconds']) + "\\n"
    + "try:\\n"
    + "    with (root / 'papermill_v3.log').open('w') as log:\\n"
    + "        rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'papermill'], stdout=log, stderr=subprocess.STDOUT, timeout=min(180, max(.1, deadline - time.monotonic()))).returncode\\n"
    + "        if rc == 0: rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd='/content', timeout=max(.1, deadline - time.monotonic())).returncode\\n"
    + "except subprocess.TimeoutExpired:\\n    rc = 124\\n"
    + "tmp = root / 'job_status_v3.json.tmp'\\n"
    + "tmp.write_text(json.dumps({'done': True, 'rc': rc, 'run_id': " + repr(run_id) + "}))\\n"
    + "tmp.replace(root / 'job_status_v3.json')\\n")
with (root / 'driver_v3.log').open('w') as stream:
    proc = subprocess.Popen([sys.executable, str(driver)], stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd='/content')
birth = pathlib.Path('/proc/' + str(proc.pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
(root / 'worker_v3.json').write_text(json.dumps({'pid': proc.pid, 'starttime': birth, 'run_id': run_id}))
print('V3_JSON::' + json.dumps({'launched': True, 'run_id': run_id}))
'''


def observation_code(marker, collect=False):
    return '''import hashlib, json, pathlib, zipfile
root = pathlib.Path('/content/jev_llm_v3')
expected = ''' + repr(marker) + '''
def blocked(state, message):
    print('V3_JSON::' + json.dumps({'blocked': state}), flush=True)
    raise RuntimeError(message)
if not (root / 'launch_v3.json').is_file() or json.loads((root / 'launch_v3.json').read_text()) != expected:
    blocked('blocked_ownership', 'Remote root/output ownership unknown')
result = {'known': False, 'alive': None, 'run_id': expected['run_id']}
worker = root / 'worker_v3.json'
if worker.exists():
    record = json.loads(worker.read_text())
    if record['run_id'] != expected['run_id']:
        blocked('blocked_ownership', 'Remote worker identity changed')
    try:
        stat = pathlib.Path('/proc/' + str(record['pid']) + '/stat').read_text().rsplit(')', 1)[1].split()
        alive = stat[0] not in ('Z', 'X') and stat[19] == record['starttime']
    except FileNotFoundError:
        alive = False
    result.update(known=True, alive=alive)
status = root / 'job_status_v3.json'
if status.is_file():
    status_content = status.read_bytes()
    value = json.loads(status_content)
    if value.get('run_id') != expected['run_id']:
        blocked('blocked_ownership', 'Remote status identity changed')
    result['execution'] = value
''' + ('''
paths = list((root / 'results/v3').rglob('*')) + list((root / 'notebooks').glob('*.out.*.ipynb'))
paths += [root / n for n in ('papermill_v3.log', 'driver_v3.log', 'job_status_v3.json', 'worker_v3.json')]
paths += list(root.glob('analysis_*.zip'))
paths = sorted(set(p for p in paths if p.is_file() and not p.is_symlink() and not p.name.endswith('.tmp')))
execution = result.get('execution', {})
if execution.get('done') is True and type(execution.get('rc')) is int and execution['rc'] == 0:
    duplicates = set()
    for path in paths:
        if path.name.endswith('_final.zip'):
            try:
                with zipfile.ZipFile(path) as final:
                    if final.testzip() is None:
                        duplicates.add(path.with_name(path.name.removesuffix('_final.zip') + '_checkpoint.zip'))
            except (OSError, zipfile.BadZipFile):
                pass
    paths = [p for p in paths if p not in duplicates]
snapshot = pathlib.Path('/content/jev_v3_collect_' + expected['run_id'] + '.zip')
hashes = {}
with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            blocked('blocked_integrity', 'Result path escaped owned root')
        name = path.relative_to(root).as_posix()
        content = status_content if path == status else path.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        z.writestr(name, content)
    z.writestr('COLLECTION_MANIFEST.json', json.dumps({'marker': expected, 'files': hashes}))
result['snapshot'] = {'path': str(snapshot), 'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()}
''' if collect else "") + "\nprint('V3_JSON::' + json.dumps(result))\n"


def preserve_snapshot(path, output, marker, expected_hash):
    """Recursive v3 results + raw notebook outputs; keep each intact checkpoint."""
    output = Path(output)
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlink collection output forbidden")
    require(digest(path) == expected_hash, "Downloaded snapshot hash mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expanded = sum(i.file_size for i in archive.infolist())
        require(expanded <= MAX_COLLECTION_BYTES and all(not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16)
                for i in archive.infolist()), "Collection ZIP expansion/symlink limit exceeded")
        require(len(names) == len(set(names)) and archive.testzip() is None, "Invalid collection ZIP")
        man = json.loads(archive.read("COLLECTION_MANIFEST.json"))
        require(man.get("marker") == marker and set(names) == set(man["files"]) | {"COLLECTION_MANIFEST.json"}, "Collection ownership/inventory mismatch")
        contents = {}
        for name, expected in man["files"].items():
            p = PurePosixPath(name)
            require(not p.is_absolute() and all(part not in {"", ".", ".."} for part in name.split("/")) and "\\" not in name and ":" not in name,
                    "Unsafe collected path")
            require(name.startswith(("results/v3/", "notebooks/")) or name in {"papermill_v3.log", "driver_v3.log", "job_status_v3.json", "worker_v3.json"}
                    or re.fullmatch(r"analysis_[A-Za-z0-9_-]+\.zip", name), "Unexpected collected path")
            content = archive.read(name)
            require(hashlib.sha256(content).hexdigest() == expected, "Collected member hash mismatch")
            if p.suffix == ".zip":
                import io
                with zipfile.ZipFile(io.BytesIO(content)) as nested:
                    expanded += sum(i.file_size for i in nested.infolist())
                    require(expanded <= MAX_COLLECTION_BYTES, "Nested collection ZIP expansion limit exceeded")
                    require(nested.testzip() is None, "Partial result ZIP; retain prior checkpoint")
            contents[name] = content
        status = json.loads(contents["job_status_v3.json"]) if "job_status_v3.json" in contents else None
        finalized = status is not None and status.get("done") is True
        if status is not None:
            require(status.get("run_id") == marker["run_id"], "Collected execution identity changed")
        writes, checkpoints = {}, []
        for name, content in contents.items():
            expected = man["files"][name]
            p = PurePosixPath(name)
            target = output / "remote" / p
            raw_notebook = p.suffix == ".ipynb" and ".out." in p.name
            immutable = p.name.endswith("_final.zip") or raw_notebook or p.name.startswith("analysis_")
            if raw_notebook and not finalized:
                target = output / "raw_notebooks" / expected / p.name
            targets = [target]
            if p.name.endswith("_checkpoint.zip"):
                checkpoint = output / "checkpoints" / (p.stem + "_" + expected[:16] + ".zip")
                targets.append(checkpoint)
                checkpoints.append(dict(path=str(checkpoint), sha256=expected, utc=utc()))
            if immutable and (not raw_notebook or finalized):
                targets.append(output / "artifacts" / p.name)
            for target in targets:
                require(target.resolve().is_relative_to(output.resolve()) and not any(
                    x.is_symlink() for x in (target, target.with_name(target.name + ".pending"), *target.parents)),
                    "Collection target escaped owned output or contains symlink")
                require(target not in writes or writes[target][0] == content, "Conflicting collection destination")
                require(not target.exists() or target.is_file(), "Non-file collection destination")
                if target.exists() and (immutable or target.parent == output / "checkpoints"):
                    require(target.is_file() and digest(target) == expected, "Immutable raw/final artifact changed; no overwrite")
                writes[target] = (content, immutable or target.parent == output / "checkpoints")
        index = output / "checkpoint_index.jsonl"
        require(not index.is_symlink(), "Symlink checkpoint index forbidden")
        snapshot = output / "collections" / (expected_hash + ".zip")
        commit = output / "collection_commit.json"
        for target in (snapshot, snapshot.with_name(snapshot.name + ".pending"), commit, commit.with_suffix(".json.tmp")):
            require(target.resolve().is_relative_to(output.resolve()) and not any(x.is_symlink() for x in (target, *target.parents)),
                    "Unsafe collection transaction destination")
        require(not snapshot.exists() or digest(snapshot) == expected_hash, "Committed collection changed")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            temporary = snapshot.with_name(snapshot.name + ".pending")
            shutil.copyfile(path, temporary)
            require(digest(temporary) == expected_hash, "Collection changed while staging transaction")
            temporary.replace(snapshot)
            snapshot.chmod(0o444)
        # The complete validated ZIP is the transaction. Mirrors/index are derived
        # views; a failed mirror write is recoverable from this atomic commit.
        atomic_json(commit, dict(marker=marker, snapshot=snapshot.relative_to(output).as_posix(), sha256=expected_hash, utc=utc()))
        for target, (content, immutable) in writes.items():
            if target.exists() and target.read_bytes() == content:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".pending")
            tmp.write_bytes(content)
            tmp.replace(target)
            if immutable:
                target.chmod(0o444)
        for record in checkpoints:
            append(index, record)
    return man


def audit_development(root, artifacts, item):
    """Requested smoke slots only, never a strict pilot/test integrity certificate."""
    from verify_run_v3 import BLOCKING, canonical_hash, check_notebook, jsonl, read_archive, timestamp
    root, artifacts = Path(root), Path(artifacts)
    c = item["condition"]
    archives = list(artifacts.glob(c + "_smoke_*_final.zip"))
    notebooks = list(artifacts.glob(NOTEBOOKS[c] + ".out.*.ipynb"))
    require(len(archives) == len(notebooks) == 1, "Development audit needs one raw notebook and one final")
    data = read_archive(archives[0])
    man, cfg = data["manifest"], data["manifest"]["config"]
    params = cfg["params"]
    frozen = read_json(root / CONFIG)
    require(cfg.get("protocol_version") == "3" and cfg.get("code_version") == "0.3.0"
            and cfg.get("experiment_config_sha256") == digest(root / CONFIG)
            and cfg.get("profile") == frozen["implementation_profile"] and cfg.get("models") == frozen["models"]
            and man.get("condition") == c and man.get("development_only") is True
            and man.get("config_hash") == canonical_hash(cfg), "Unknown/non-development smoke provenance")
    require(all(params.get(k) == v for k, v in item["params"].items() if k in params), "Development parameters changed")
    require(params.get("SEEDS") == "17" and params.get("SPLIT") == "dev"
            and params.get("DATA_SUBDIR") == "data/v3" and params.get("CONFIG_FILE") == CONFIG,
            "Smoke must remain seed17/dev/v3")
    inputs = root / "data/v3/dev_inputs.jsonl"
    if not inputs.is_file():
        inputs = root / "data/v3/staging/dev_reuse_inputs.jsonl"
    if not inputs.is_file():
        inputs = root / "data/dev_inputs.jsonl"
    rows = jsonl(inputs.read_text(encoding="utf-8"))
    require(len(rows) == len({r["id"] for r in rows}) == 20 and all(set(r) == {"id", "problem", "language"} for r in rows),
            "Smoke inputs must reuse the original20, not claimed test items")
    require(cfg.get("data_sha256", {}).get("dev_inputs.jsonl") == digest(inputs), "Smoke dev input hash changed")
    requested_ids = [r["id"] for r in rows[:2]] if c == "LATENCY" else [sorted(r["id"] for r in rows)[0]]
    predictions = data["latency_metrics" if c == "LATENCY" else "predictions"]
    if c == "LATENCY":
        expected = {(pid, arm, 17, 0) for pid in requested_ids for arm in ("JFINAL", "B13_GREEDY")}
        keys = [(r["item_id"], r["condition"], r["seed"], r["repetition"]) for r in predictions]
        require(params.get("LAT_N_ITEMS") == 2 and params.get("LAT_REPETITIONS") == 1
                and params.get("LAT_CONDITIONS") == "JFINAL,B13_GREEDY", "Development latency is four measurements only")
    else:
        expected = {(pid, c, 17) for pid in requested_ids}
        keys = [(r["problem_id"], r["condition"], r["seed"]) for r in predictions]
        require(man.get("requested_coverage", {}).get("n_problems") == 1, "Quality smoke is N1 only")
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, "Development requested physical coverage mismatch")
    require(len(data["predictions"]) == len(data["metrics"]) == len(expected), "Development companion ledger coverage mismatch")
    row_key = ((lambda r: (r.get("item_id"), r.get("condition"), r.get("seed"), r.get("repetition")))
               if c == "LATENCY" else (lambda r: (r.get("problem_id"), r.get("condition"), r.get("seed"))))
    require(all(Counter(row_key(r) for r in data[name]) == Counter(expected) for name in ("predictions", "metrics")),
            "Development companion identities differ from requested slots")
    require(man.get("finished_utc") and man.get("stages") and all(s.get("ok") is True for s in man["stages"].values()),
            "Development execution stages incomplete")
    preflights = man.get("preflight", {})
    checks = BLOCKING | ({"selector_native_reference", "selector_equivalence"} if c in {"JFINAL", "LATENCY"} else set())
    require(checks <= set(preflights) and all(preflights[k].get("ok") is True and preflights[k].get("skipped") is not True for k in checks),
            "Development mandatory preflight failed/missing")
    uuid_value = man.get("gpu_uuid")
    require(uuid_value and len(man.get("environment", {}).get("gpus", [])) == 1
            and man["environment"]["gpus"][0].get("uuid") == uuid_value, "Development physical GPU missing")
    for row in predictions:
        require(row.get("config_hash") == man["config_hash"] and row.get("gpu_uuid") == uuid_value
                and isinstance(row.get("public_output"), str), "Development row identity mismatch")
        t = row.get("T_total")
        require(type(t) in (int, float) and math.isfinite(t) and t > 0
                and 0 < timestamp(row["t_end_utc"]) - timestamp(row["t_start_utc"]) + .1 >= t,
                "Development clocks invalid")
        require(row.get("effective_seed") == (17 if row["condition"] in SAMPLED else None),
                "Development effective seed mismatch")
    intervals = sorted((timestamp(r["t_start_utc"]), timestamp(r["t_end_utc"])) for r in predictions)
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "Development executions overlap or clone clocks")
    nb_hash = check_notebook(notebooks[0], man)
    return dict(ok=True, development_only=True, confirmatory_verified=False, pilot_go=False,
                scope="Requested diagnostic slots only; no full20/500 coverage or funding claim",
                runs={c: dict(ok=True, condition=c, records=len(predictions), archive=str(archives[0]),
                              archive_sha256=digest(archives[0]), notebook_sha256=nb_hash, config_hash=man["config_hash"])})


class Operator:
    def __init__(self, root, output, state, backend=None):
        self.root, self.out, self.state = Path(root), Path(output), state
        self.item = state["job"]
        self.backend = backend or Backend(self.out, self.item["session"])
        self.guard = self.out / "guard"
        self.owner = state.get("owned_endpoint")
        self.ownership_record = None
        self.marker = state.get("remote_marker")
        self.deadline = state["deadline_epoch"]
        self.work_deadline = self.deadline - CLEANUP_SECONDS
        self.monotonic_deadline = state.get("deadline_monotonic", time.monotonic() + max(0, self.deadline - time.time()))
        self.monotonic_work_deadline = state.get("work_deadline_monotonic", self.monotonic_deadline - CLEANUP_SECONDS)
        self.recovery = None

    def save(self, status, **fields):
        self.state.update(status=status, updated_utc=utc(), **fields)
        atomic_json(self.out / "status.json", self.state)
        atomic_json(self.out / "execution_handover.json", self.state)
        append(self.out / "events.jsonl", dict(utc=utc(), status=status, **fields))

    def remaining(self, maximum=120, cleanup=False):
        left = min((self.deadline if cleanup else self.work_deadline) - time.time(),
                   (self.monotonic_deadline if cleanup else self.monotonic_work_deadline) - time.monotonic())
        require(left > 0, "Job deadline expired", "blocked_deadline")
        return min(maximum, left)

    def identity(self, *, cleanup=False):
        return self.backend.identity(timeout=self.remaining(60, cleanup))

    def remote(self, code, name, *, cleanup=False, maximum=120):
        script = self.out / (name + "_remote.py")
        script.write_text(code, encoding="utf-8")
        rc, text = self.backend.call(["exec", "-s", self.item["session"], "--timeout", "30", "-f", str(script)],
                                     timeout=self.remaining(maximum, cleanup), capture=True)
        values = [line.split("::", 1)[1] for line in text.splitlines() if line.startswith("V3_JSON::")]
        result = json.loads(values[0]) if len(values) == 1 else None
        if isinstance(result, dict) and result.get("blocked") in {"blocked_ownership", "blocked_integrity"}:
            raise Blocked(result["blocked"], "Remote observation ownership/integrity conflict; no retry")
        require(rc == 0, "Remote transport uncertain; no remote restart", "blocked_transport")
        require(len(values) == 1, "Remote acknowledgement missing/ambiguous", "blocked_transport")
        return result

    def record_owner(self, data):
        endpoint = data.get("local_endpoint")
        require(endpoint and endpoint not in self.state["before_endpoints"], "Fresh owned endpoint unproved", "blocked_ownership")
        require(self.owner is None or self.owner == endpoint, "Owned endpoint changed", "blocked_ownership")
        self.owner = endpoint
        identity = dict(session=self.item["session"], endpoint=endpoint, output=str(self.out),
                        root=str(self.root), series="v3", run_id=self.state["run_id"])
        self.ownership_record = identity
        self.state["owned_endpoint"] = endpoint
        atomic_json(self.out / "ownership.json", identity)
        self.save("owned", owned_endpoint=endpoint)

    def check_owner(self, data, *, present=True):
        require(self.owner and data["local_endpoint"] == self.owner, "Scoped session replaced/missing; unknowns untouched", "blocked_ownership")
        record = read_json(self.out / "ownership.json") if (self.out / "ownership.json").exists() else self.ownership_record
        require(record == dict(session=self.item["session"], endpoint=self.owner, output=str(self.out),
                               root=str(self.root), series="v3", run_id=self.state["run_id"]), "Ownership output/root mismatch")
        matches = [r for r in data["assignments"] if r["endpoint"] == self.owner]
        require(not present or len(matches) == 1, "Owned assignment absent/uncertain", "blocked_backend")
        if matches and present:
            require(matches[0]["accelerator"] == ("NONE" if self.item["gpu"] == "CPU" else "A100"), "Wrong allocated accelerator")
        require(not present or len(data["assignments"]) <= 2, "Global two-assignment cap exceeded; unknowns untouched", "blocked_capacity")

    def validate_frozen(self):
        evidence = self.state["evidence"]
        for name, expected in {**evidence["frozen_files"], **evidence.get("bundle_manifest", {}).get("source_sha256", {})}.items():
            require(digest(self.root / name) == expected, "Frozen source/config/data/evidence changed before allocation/submission")

    def reserve_budget(self):
        if self.item["gpu"] == "CPU":
            return
        host = host_gate(self.state["options"].get("host_ack"), self.root, job_seconds=self.item["deadline_seconds"])
        require(host["host_lease_id"] == self.state.get("host_identity", {}).get("host_lease_id"),
                "Host lease changed before reservation; no adoption for new work", "blocked_host_identity")
        initial_budget_gate(self.root, self.state["options"].get("initial_budget"))
        budget = self.root / BUDGET_LEDGER
        eta = self.state["evidence"].get("eta_seconds")
        if self.item["phase"] in {"test", "latency"}:
            require(type(eta) in (int, float) and math.isfinite(eta) and CLEANUP_SECONDS < eta <= JOB_SECONDS,
                    "Measured job allowance unavailable", "blocked_budget_or_eta")
        with flock("/tmp/jev-v3-budget.lock", wait=self.remaining()):
            ledger = budget_ledger(self.root)
            self.reconcile_budget(ledger)
            require(self.state["run_id"] not in ledger["jobs"], "Prior budget reservation; no retry", "blocked_budget_or_eta")
            require(sum(e.get("released") is not True for e in ledger["jobs"].values()) < 2,
                    "Two live GPU budget leases already reserved", "blocked_capacity")
            ceiling = MAX_GPU_SECONDS
            if self.item["phase"] in {"test", "latency"}:
                approved = read_json(self.state["options"]["approval"])
                require(approved.get("experiment_id") == EXPERIMENT_ID and type(approved.get("max_gpu_hours")) in (int, float)
                        and 0 < approved["max_gpu_hours"] <= 25, "Post-pilot experiment budget binding changed", "blocked_budget_approval")
                ceiling = min(ceiling, approved["max_gpu_hours"] * 3600)
                report = read_json(self.state["options"]["pilot_report"])
                estimates = estimate_jobs(self.root, remaining_confirmatory(self.root, self.state), approved, report)
                require(sum(estimates.values()) <= ceiling - budget_spent(ledger),
                        "Full remaining matrix + LATENCY no longer fits reconciled budget", "blocked_budget_or_eta")
            allowance = min(self.item["deadline_seconds"], ceiling - budget_spent(ledger), self.deadline - time.time(), self.monotonic_deadline - time.monotonic())
            require(allowance > CLEANUP_SECONDS and (eta is None or eta <= allowance),
                    "Accumulated budget consumed/reserved; bounded tail cannot cover job/cleanup", "blocked_budget_or_eta")
            ledger["jobs"][self.state["run_id"]] = dict(key=self.item["key"], session=self.item["session"], output=str(self.out),
                host_lease_id=host["host_lease_id"],
                reserved_seconds=allowance, reserved_epoch=time.time(), deadline_epoch=time.time() + allowance,
                deadline_monotonic=time.monotonic() + allowance, clock_boot_id=BOOT_ID,
                allocation_attempted=False, observed_seconds=0, released=False,
                authorization_sha256=digest(self.state["options"]["initial_budget"]))
            atomic_json(budget, ledger)
        self.state["budget_ledger"] = str(budget)
        self.state["budget_reserved_seconds"] = allowance
        self.deadline = min(self.deadline, time.time() + allowance)
        self.work_deadline = self.deadline - CLEANUP_SECONDS
        self.monotonic_deadline = min(self.monotonic_deadline, time.monotonic() + allowance)
        self.monotonic_work_deadline = self.monotonic_deadline - CLEANUP_SECONDS
        self.state["deadline_epoch"] = self.deadline
        self.state.update(deadline_monotonic=self.monotonic_deadline, work_deadline_monotonic=self.monotonic_work_deadline, clock_boot_id=BOOT_ID)
        self.save("budget_reserved")

    def reconcile_budget(self, ledger):
        """Only authoritative absence refunds a prior lease; live work is never renewed."""
        for run_id, entry in ledger["jobs"].items():
            if entry.get("released") is True:
                continue
            directory = Path(entry["output"])
            require(directory.resolve().is_relative_to((self.root / "results/v3").resolve())
                    and not any(p.is_symlink() for p in (directory, *directory.parents)),
                    "Prior lease escaped owned results", "blocked_ownership")
            state = read_json(directory / "status.json")
            require(state.get("run_id") == run_id and state.get("output") == str(directory)
                    and state.get("root") == str(self.root) and state.get("series") == "v3"
                    and state.get("job", {}).get("session") == entry["session"],
                    "Prior lease/status identity changed", "blocked_ownership")
            attempted = state.get("allocation_attempted") is True or entry.get("allocation_attempted") is True
            if not attempted:
                # A reservation held by another running operator is still live.
                require(state.get("status") == "failed", "Prior reservation/startup uncertain; reconcile without renewal", "blocked_budget_or_eta")
                entry.update(released=True, release_verified=True, actual_seconds=0, release_evidence="durable_no_allocation")
                continue
            endpoint = state.get("owned_endpoint", entry.get("endpoint"))
            require(endpoint and read_json(directory / "ownership.json") == dict(session=entry["session"], endpoint=endpoint,
                    output=str(directory), root=str(self.root), series="v3", run_id=run_id),
                    "Prior allocated lease ownership uncertain; no new spending", "blocked_ownership")
            data = Backend(directory, entry["session"]).identity()
            start = entry.get("allocation_epoch", state.get("allocation_epoch", entry["reserved_epoch"]))
            previous_observed = entry.get("observed_seconds", 0)
            observed = max(previous_observed, lease_elapsed(entry))
            entry.update(endpoint=endpoint, allocation_attempted=True, allocation_epoch=start, observed_seconds=observed)
            if endpoint not in {r["endpoint"] for r in data["assignments"]}:
                # Previously recorded elapsed time is usable only after absence is independently proved.
                actual = max(entry.get("actual_seconds", 0), entry.get("observed_seconds", 0))
                if state.get("released") is True and type(state.get("observed_gpu_seconds_upper_bound")) in (int, float):
                    actual = max(entry.get("actual_seconds", 0), previous_observed, state["observed_gpu_seconds_upper_bound"])
                elif state.get("released") is True and state.get("status") in {"completed", "failed"} and state.get("updated_utc"):
                    actual = max(entry.get("actual_seconds", 0), previous_observed, epoch(state["updated_utc"]) - start, 0)
                    if "allocation_monotonic" in entry:
                        actual = max(actual, observed)  # Missing end bound never refunds the monotonic consumption floor.
                entry.update(released=True, release_verified=True, actual_seconds=actual,
                             observed_end_epoch=time.time(), release_evidence="backend_absent")
            else:
                require(data["local_endpoint"] == endpoint, "Prior live lease identity uncertain", "blocked_ownership")
                if (time.time() >= entry.get("deadline_epoch", start + entry["reserved_seconds"]) or
                        entry.get("clock_boot_id") == BOOT_ID and time.monotonic() >= entry.get("deadline_monotonic", float("inf"))):
                    # Cleanup only: this extends no work deadline or GPU reservation.
                    try:
                        with flock("/tmp/jev-v3-operator-" + entry["session"] + ".lock"):
                            prior = Operator(self.root, directory, state)
                            prior.deadline = time.time() + CLEANUP_SECONDS
                            prior.work_deadline = time.time()
                            prior.monotonic_deadline = time.monotonic() + CLEANUP_SECONDS
                            prior.monotonic_work_deadline = time.monotonic()
                            require(prior.release(), "Prior expired owned lease release unconfirmed", "blocked_budget_or_eta")
                    except Blocked as error:
                        if error.state == "blocked_ownership":
                            continue  # A live operator retains control; never wait on it under the budget lock.
                        raise
                    actual = max(observed, lease_elapsed(entry))
                    entry.update(released=True, release_verified=True, actual_seconds=actual, observed_seconds=actual,
                                 observed_end_epoch=time.time(), release_evidence="backend_absent")
        atomic_json(self.root / BUDGET_LEDGER, ledger)

    def observe_budget(self, *, allocating=False):
        if not self.state.get("budget_ledger"):
            return
        with flock("/tmp/jev-v3-budget.lock", wait=self.remaining()):
            ledger = budget_ledger(self.root)
            entry = ledger["jobs"][self.state["run_id"]]
            if entry.get("released") is True:
                return  # Backend-verified closure is terminal, including its observed end clock.
            if allocating:
                entry.update(allocation_attempted=True, allocation_epoch=time.time(), allocation_monotonic=time.monotonic(), clock_boot_id=BOOT_ID)
                self.state["allocation_epoch"] = entry["allocation_epoch"]
                self.state.update(allocation_monotonic=entry["allocation_monotonic"], clock_boot_id=BOOT_ID)
            entry.update(endpoint=self.owner, observed_epoch=time.time(), observed_seconds=max(entry.get("observed_seconds", 0),
                         lease_elapsed(entry) if entry.get("allocation_attempted") else 0))
            atomic_json(self.root / BUDGET_LEDGER, ledger)
            require(budget_spent(ledger) <= MAX_GPU_SECONDS, "Observed accumulated GPU time exceeds cap", "blocked_budget_or_eta")

    def finish_budget(self, released):
        if not self.state.get("budget_ledger"):
            return
        path = Path(self.state["budget_ledger"])
        with flock("/tmp/jev-v3-budget.lock", wait=max(0, min(10, self.deadline - time.time()))):
            ledger = budget_ledger(self.root)
            entry = ledger["jobs"][self.state["run_id"]]
            require(entry["session"] == self.item["session"] and entry["output"] == str(self.out), "Budget ownership mismatch")
            if entry.get("released") is True:
                self.state["observed_gpu_seconds_upper_bound"] = entry["actual_seconds"]
                return
            attempted = self.state.get("allocation_attempted") or entry.get("allocation_attempted")
            absence = False
            if released and attempted and self.owner:
                absence = self.owner not in {r["endpoint"] for r in self.identity(cleanup=True)["assignments"]}
            verified_release = bool(released and (absence or not attempted))
            actual = lease_elapsed(entry) if attempted else 0.0
            entry.update(released=verified_release, release_verified=verified_release, actual_seconds=actual,
                         observed_seconds=actual, observed_end_epoch=time.time(),
                         release_evidence="backend_absent" if absence else "durable_no_allocation" if not attempted else None,
                          endpoint=self.owner, billed_cu=None, monetary_cost=None)
            atomic_json(path, ledger)
            self.state["observed_gpu_seconds_upper_bound"] = actual
            require(not released or verified_release, "Budget refund requires verified server release", "blocked_budget_or_eta")

    def allocate(self):
        self.validate_frozen()
        with flock("/tmp/jev-colab-allocation.lock", wait=min(120, self.remaining())):
            host = host_gate(self.state["options"].get("host_ack"), self.root, job_seconds=self.item["deadline_seconds"])
            require(host["host_lease_id"] == self.state.get("host_identity", {}).get("host_lease_id"),
                    "Host lease changed before allocation", "blocked_host_identity")
            self.reserve_budget()
            data = self.identity()
            require(data["local_endpoint"] is None, "Existing session: do not rerun/restart", "blocked_ownership")
            require(len(data["assignments"]) < 2, "Global two-assignment cap; unknown assignments untouched", "blocked_capacity")
            self.validate_frozen()
            self.save("allocation_intent", before_endpoints=[r["endpoint"] for r in data["assignments"]], allocation_attempted=True)
            self.observe_budget(allocating=True)
            args = ["new", "-s", self.item["session"]]
            if self.item["gpu"] != "CPU":
                args += ["--gpu", "A100"]
            rc, _ = self.backend.call(args, timeout=self.remaining())
            after = self.identity()
            if after["local_endpoint"]:
                self.record_owner(after)
            require(rc == 0, "Allocation uncertain/failed; never retry", "blocked_backend")
            self.check_owner(after)
            require(len(after["assignments"]) <= 2, "Global assignment cap exceeded")

    def guard_call(self, action, *, cleanup=False):
        env_python = self.backend.python
        # session_guard.sh inherits COLAB_PYTHON, not the notebook environment.
        command = ["env", "COLAB_PYTHON=" + env_python, "setsid", "--wait", "bash", str(self.root / "tools/colab/session_guard.sh"),
                   action, self.item["session"], str(self.backend.config), str(self.guard)]
        rc, _ = bounded_call(command, timeout=self.remaining(180 if cleanup else 30, cleanup), log=self.out / "guard_calls.log")
        require(rc == 0, "Independent guard " + action + " unconfirmed", "blocked_guard")

    def guard_running(self):
        try:
            pid = int((self.guard / "session_guard.pid").read_text())
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            return stat[0] not in {"Z", "X"} and str(self.root / "tools/colab/session_guard.py") in args and self.item["session"] in args
        except (OSError, ValueError):
            return False

    def guard_healthy(self):
        try:
            if not self.guard_running() or read_json(self.guard / "session_guard_identity.json") != {"session": self.item["session"], "endpoint": self.owner}:
                return False
            records = [json.loads(line) for line in (self.guard / "session_lifecycle.jsonl").read_text().splitlines() if line]
            for record in reversed(records):
                if record["event"] in {"guard_started", "guard_stopped", "assignment_absent"}:
                    return False
                if record["event"] == "heartbeat" and record.get("assignment_present") is True and record.get("socket_connected") is True:
                    return 0 <= time.time() - epoch(record["utc"]) < 65
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return False

    def recover_guard(self):
        if self.guard_running():
            return  # Live guard reconnects itself; never restart remote active work.
        identity = self.guard / "session_guard_identity.json"
        if identity.exists() and read_json(identity) == {"session": self.item["session"], "endpoint": self.owner}:
            self.check_owner(self.identity())
            self.guard_call("start")

    def submit(self):
        evidence = self.state["evidence"]
        self.validate_frozen()
        self.check_owner(self.identity())
        self.guard_call("start")
        uploads = [Path(evidence["bundle"]), Path(evidence["notebook"])]
        finals = []
        approval_name = None
        if evidence.get("initial_budget"):
            rc, _ = self.backend.call(["upload", "-s", self.item["session"], evidence["initial_budget"],
                                       "/content/jev_v3_initial_budget.json"], timeout=self.remaining())
            require(rc == 0, "Initial parent authorization upload failed")
        if self.state["options"].get("approval"):
            approval_name = "jev_v3_budget_approval.json"
            rc, _ = self.backend.call(["upload", "-s", self.item["session"], self.state["options"]["approval"], "/content/" + approval_name], timeout=self.remaining())
            require(rc == 0, "Approval sentinel upload failed")
            rc, _ = self.backend.call(["upload", "-s", self.item["session"], self.state["options"]["pilot_report"], "/content/jev_v3_pilot_go.json"], timeout=self.remaining())
            require(rc == 0, "Pinned technical pilot GO upload failed")
            uploads += [Path(p) for p in evidence["pilot_archives"].values()]
        if self.item["phase"] == "analysis":
            analysis = analysis_inputs(self.root, self.state["options"]["approval"])
            uploads += analysis
            finals = [p.name for p in analysis]
        for path in uploads:
            rc, _ = self.backend.call(["upload", "-s", self.item["session"], str(path), "/content/" + path.name], timeout=self.remaining())
            require(rc == 0, "Upload failed; no resubmission")
        self.marker = dict(series="v3", session=self.item["session"], endpoint=self.owner, run_id=self.state["run_id"],
                           output=str(self.out), config_sha256=digest(self.root / CONFIG))
        self.save("remote_submission_intent", remote_marker=self.marker)
        result = self.remote(bootstrap_code(self.item, evidence, self.marker, approval_name, finals,
                                            work_seconds=self.remaining(JOB_SECONDS)), "bootstrap")
        require(result == {"launched": True, "run_id": self.state["run_id"]}, "Missing durable remote startup acknowledgement")
        self.save("running", remote_startup_ack=True)

    def collect(self, *, cleanup=False):
        allowance = min(120, self.remaining(JOB_SECONDS, cleanup) - (180 if cleanup else 0))
        require(allowance > 0, "Collection allowance exhausted; retain time for owned release", "blocked_deadline")
        until = time.monotonic() + allowance
        self.state["collection_operation"] = "remote_snapshot"
        result = self.remote(observation_code(self.marker, collect=True), "collect", cleanup=cleanup, maximum=min(60, allowance))
        snapshot = result["snapshot"]
        require(snapshot["path"] == "/content/jev_v3_collect_" + self.state["run_id"] + ".zip", "Unexpected remote snapshot path")
        temp = self.out / "collection.pending.zip"
        self.state["collection_operation"] = "download"
        timeout = min(self.remaining(60, cleanup), until - time.monotonic())
        require(timeout > 0, "Collection transport allowance exhausted", "blocked_transport")
        rc, _ = self.backend.call(["download", "-s", self.item["session"], snapshot["path"], str(temp)],
                                 timeout=timeout)
        require(rc == 0, "Collection download failed; preserve prior snapshot", "blocked_transport")
        self.state["collection_operation"] = "local_integrity"
        preserve_snapshot(temp, self.out, self.marker, snapshot["sha256"])
        temp.unlink()
        return result

    def collect_with_recovery(self, *, final=False, cleanup=False):
        for attempt in range(1, 4 if final else 2):
            if final:
                self.state["final_collection_attempted"] = True
            try:
                if final and attempt > 1:
                    require(self.remaining(JOB_SECONDS, cleanup) > (300 if cleanup else 120),
                            "Final collection observation allowance exhausted; preserve release reserve", "blocked_deadline")
                    self.state["collection_operation"] = "identity"
                    self.check_owner(self.identity(cleanup=cleanup))
                    self.state["collection_operation"] = "health"
                    observed = self.remote(observation_code(self.marker), "health", cleanup=cleanup, maximum=60)
                    require(observed.get("run_id") == self.state["run_id"] and observed.get("known") is True,
                            "Remote status unknown during final collection", "blocked_transport")
                    self.state["collection_operation"] = "guard"
                    require(self.guard_healthy(), "Independent guard not live during final collection", "blocked_guard")
                self.collect(cleanup=cleanup)
                self.save("collection_recovered", collection_degraded=False)
                return True
            except (Blocked, subprocess.TimeoutExpired) as error:
                if isinstance(error, Blocked) and error.state != "blocked_transport":
                    raise
                self.save("collection_degraded", collection_degraded=True, collection_stage="final" if final else "periodic",
                          collection_attempt=attempt, collection_operation=self.state.get("collection_operation", "collect"),
                          collection_error_type=type(error).__name__, collection_error_state=getattr(error, "state", None),
                          collection_timeout_seconds=getattr(error, "timeout", None))
                if not final:
                    return False  # Health/guard observation resumes immediately, not notebook execution.
                if attempt == 3:
                    raise
                time.sleep(min(30, self.remaining(30, cleanup)))

    def verify(self):
        item = self.item
        released_reaudit = (self.state.get("status") == "completed" and self.state.get("verified") is True
                            and self.state.get("released") is True and self.state.get("completed_execution") is True)
        output = self.out / "verification.json"
        if item["phase"] == "analysis":
            from verify_run_v3 import check_notebook
            nb = list((self.out / "artifacts").glob("07_ANALYSIS.out.*.ipynb"))
            require(len(nb) == 1, "Missing unique raw analysis notebook")
            check_notebook(nb[0])
            archive = self.out / "artifacts/analysis_confirmatory_v3.zip"
            with zipfile.ZipFile(archive) as z:
                require(z.testzip() is None and {"analysis.json", "report.md"} <= set(z.namelist()), "Incomplete analysis ZIP")
                meta = json.loads(z.read("analysis.json"))["meta"]
                require(meta.get("protocol_version") == "3" and meta.get("code_version") == "0.3.0"
                        and meta.get("n_physical_cases") == 6000 and meta.get("n_test_problems") == 500
                        and meta.get("run_tag") == "confirmatory_v3" and meta.get("historical_compatibility") is False,
                        "Unknown/incomplete analysis coverage")
            atomic_json(output, dict(ok=True, archive_sha256=digest(archive), notebook_sha256=digest(nb[0])))
            return
        if item["phase"] == "smoke":
            atomic_json(output, audit_development(self.root, self.out / "artifacts", item))
            return
        checker = self.root / "tools/verify_run_v3.py"
        options = self.state["options"]
        if item["phase"] in {"test", "latency"} and options.get("approval"):
            approved = read_json(options["approval"])
            if "operational_amendment" in approved:
                files = private_auditor_files(self.root, approved)
                frozen = self.state["evidence"]["frozen_files"]
                require(all(frozen.get(name) == sha for name, sha in files.items()),
                        "Private auditor not bound in frozen job evidence")
                if released_reaudit:
                    # Released evidence binds science and consent, not subsequently repaired launch tooling.
                    require(frozen.get(str(Path(options["approval"]).resolve())) == digest(options["approval"]),
                            "Frozen approval changed after release")
                else:
                    self.validate_frozen()
                runtime_approval_gate(self.root, item, options["approval"], options["pilot_report"])
                checker = self.root / "tools/audit_native_container_v3.py"
        args = [sys.executable, str(checker), str(self.out / "artifacts"),
                "--conditions", item["condition"], "--inputs", str(self.root / "data/v3" / (item["params"]["SPLIT"] + "_inputs.jsonl")),
                "--config", str(self.root / CONFIG), "--prompts", str(self.root / "prompts"),
                "--expected-count", str(item["count"]), "--split", item["params"]["SPLIT"],
                "--run-tag", item["params"]["RUN_TAG"], "--output", str(output)]
        if self.state["options"].get("approval"):
            args += ["--approval", self.state["options"]["approval"]]
        if checker.name == "audit_native_container_v3.py":
            args += ["--pilot-report", options["pilot_report"]]
        release_reserve = (180 if self.state.get("final_collection_attempted") and self.owner
                           and item["gpu"] != "CPU" and self.state.get("released") is not True else 0)
        timeout = min(600 if released_reaudit else 120, self.remaining(JOB_SECONDS, True) - release_reserve)
        require(timeout > 0, "Local verification allowance exhausted; preserve owned release reserve", "blocked_deadline")
        rc, _ = bounded_call(args, timeout=timeout, log=self.out / "verification.log")
        require(rc == 0 and read_json(output).get("ok") is True, "Current v3 integrity CLI failed; no quality retry")
        require(read_json(output)["runs"][item["condition"]]["records"] == item["expected_rows"], "Verifier physical coverage mismatch")

    def release(self):
        if not self.owner:
            return not self.state.get("allocation_attempted", False)
        data = self.identity(cleanup=True)
        if self.owner not in {r["endpoint"] for r in data["assignments"]}:
            return True  # Authoritative absence; never touch a replacement record.
        self.check_owner(data, present=False)
        path = self.guard / "session_guard_identity.json"
        if path.exists():
            require(read_json(path) == {"session": self.item["session"], "endpoint": self.owner}, "Guard ownership changed; do not release replacement")
        # Temporary release logs remain writable if result persistence fills the
        # workspace volume. The helper receives ONLY the recorded endpoint.
        with tempfile.TemporaryDirectory(prefix="jev-v3-release-") as directory:
            try:
                bounded_call([self.backend.python, "-c", RELEASE_CODE, str(self.backend.config), self.item["session"],
                              self.owner, str(self.root / "tools/colab/session_guard.py"), directory],
                             timeout=self.remaining(180, True), capture=True)
            except subprocess.SubprocessError:
                pass  # A timeout/error is not proof of release; query the server.
        after = self.identity(cleanup=True)
        require(self.owner not in {r["endpoint"] for r in after["assignments"]}, "Server-side owned release unconfirmed")
        try:
            (self.guard / "session_guard.stop").touch()
        except OSError:
            pass  # The independent watcher also exits on authoritative absence.
        return True

    def run(self, *, resume=False):
        verified = completed = released = False
        try:
            self.save("observation_resumed" if resume else "startup_acknowledged", operator_pid=os.getpid(), startup_ack=True)
            if resume:
                require(self.state.get("clock_boot_id") == BOOT_ID and "deadline_monotonic" in self.state,
                        "Original monotonic clock domain unavailable; cleanup only, never renew work", "blocked_ownership")
                require(self.owner and self.marker and self.state.get("remote_startup_ack") is True,
                        "No recorded remote submission to observe; never allocate/resubmit", "blocked_ownership")
                self.validate_frozen()
                commit = self.out / "collection_commit.json"
                if commit.exists():
                    transaction = read_json(commit)
                    snapshot = self.out / transaction["snapshot"]
                    require(snapshot.resolve().is_relative_to((self.out / "collections").resolve()) and not snapshot.is_symlink(),
                            "Committed collection escaped owned output")
                    preserve_snapshot(snapshot, self.out, self.marker, transaction["sha256"])
                self.check_owner(self.identity())
            else:
                self.allocate()
                self.submit()
            controls, last_collect = set(self.state.get("controls_minutes", [])), 0.0
            while time.time() < self.work_deadline and time.monotonic() < self.monotonic_work_deadline:
                tick = time.time()
                self.observe_budget()
                try:
                    self.check_owner(self.identity())
                    observed = self.remote(observation_code(self.marker), "health", maximum=60)
                    require(observed.get("run_id") == self.state["run_id"] and observed.get("known") is True, "Remote status unknown", "blocked_transport")
                    healthy = self.guard_healthy()
                    if not healthy:
                        self.recover_guard()
                    require(healthy, "Independent guard heartbeat not live", "blocked_guard")
                    self.recovery = None
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    if isinstance(error, Blocked) and error.state in {"blocked_ownership", "blocked_integrity", "blocked_capacity"}:
                        raise
                    self.recovery = self.recovery or tick
                    self.save("guard_or_transport_degraded", recovery_started_epoch=self.recovery, error_type=type(error).__name__)
                    require(time.time() - self.recovery < RECOVERY_SECONDS, "Health unknown for 180 seconds; no restart", "blocked_guard")
                    time.sleep(min(5, max(0, min(self.work_deadline - time.time(), self.monotonic_work_deadline - time.monotonic()))))
                    continue
                elapsed = tick - self.state["start_epoch"]
                for minute in CONTROLS:
                    if minute not in controls and elapsed >= minute * 60:
                        controls.add(minute)
                        append(self.out / "controls.jsonl", dict(minute=minute, utc=utc(), elapsed_seconds=elapsed, health=observed))
                self.save("running", health=observed, elapsed_seconds=elapsed, controls_minutes=sorted(controls))
                if observed["alive"] is False:
                    execution = observed.get("execution", {})
                    require(execution.get("done") is True and execution.get("rc") == 0, "Notebook stopped/incomplete/nonzero; no quality retry")
                    completed = True
                    self.collect_with_recovery(final=True, cleanup=True)
                    self.verify()
                    verified = True
                    break
                if tick - last_collect >= 120:
                    collected = self.collect_with_recovery()
                    last_collect = time.time()
                    if not collected:
                        continue
                time.sleep(max(0, min(POLL_SECONDS - (time.time() - tick), self.work_deadline - time.time(), self.monotonic_work_deadline - time.monotonic())))
            require(completed, "Four-hour deadline reached; preserve partial results", "blocked_deadline")
        except BaseException as error:
            try:
                self.save(getattr(error, "state", "failed"), error_type=type(error).__name__, error_message=diagnosis(error)["error_message"])
            except OSError:
                self.state["persistence_failed"] = True
        finally:
            try:
                self.save("preserving_and_releasing", completed_execution=completed, verified=verified)
            except OSError:
                self.state["persistence_failed"] = True
            try:
                if not self.owner and self.state.get("allocation_attempted"):
                    data = self.identity(cleanup=True)
                    if data["local_endpoint"]:
                        self.record_owner(data)
                if self.owner and self.marker and not verified and not self.state.get("final_collection_attempted"):
                    try:
                        self.collect(cleanup=True)
                    except (ValueError, OSError, subprocess.SubprocessError):
                        self.state["preservation_unconfirmed"] = True
                released = self.release()
            except BaseException as error:
                self.state.update(release_error_type=type(error).__name__, release_unconfirmed=True)
            try:
                self.finish_budget(released)
            except (ValueError, OSError):
                self.state["persistence_failed"] = True
            final = dict(status="completed" if completed and verified and released and not self.state.get("persistence_failed") else "failed", verified=verified,
                         completed_execution=completed, released=released)
            self.state.update(final)
            try:
                self.save(final.pop("status"), **final)
            except OSError:
                self.state["persistence_failed"] = True
        return 0 if completed and verified and released and not self.state.get("persistence_failed") else 1


def analysis_inputs(root, approval):
    from verify_run_v3 import verify_run
    root = Path(root)
    finals, notebooks = [], []
    for item in [*plan("test"), *plan("latency")]:
        directory = root / "results/v3" / item["key"] / "artifacts"
        report = verify_run(directory, conditions=[item["condition"]], inputs=root / "data/v3/test_inputs.jsonl",
                            config=root / CONFIG, prompts=root / "prompts", expected_count=500, split="test",
                            run_tag="confirmatory_v3", approval=approval)
        require(report.get("ok") is True, "Analysis requires seven current-v3 verified physical finals")
        finals.append(Path(report["runs"][item["condition"]]["archive"]))
        candidates = list(directory.glob(NOTEBOOKS[item["condition"]] + ".out.*.ipynb"))
        require(len(candidates) == 1, "Analysis requires corresponding raw executed notebooks")
        notebooks += candidates
    require(len({p.name for p in finals + notebooks}) == 14, "Duplicate analysis input identities")
    return finals + notebooks


def canonical_pilot(root):
    """Artifact-only view: recursive mirrors must not become duplicate pilot runs."""
    from verify_run_v3 import verify_run
    root = Path(root)
    destination = root / "results/v3/pilot_verified"
    require(not any(p.is_symlink() for p in (destination, *destination.parents)), "Symlink canonical pilot destination forbidden")
    expected_paths = set()
    sources = []
    for item in plan("pilot"):
        directory = root / "results/v3" / item["key"]
        state = status_gate(root, directory, item)
        require(state.get("status") == "completed" and state.get("verified") is True
                and state.get("released") is True and state.get("completed_execution") is True,
                "Pilot operator completion/release unproved", "blocked_pilot_go")
        report = verify_run(directory / "artifacts", conditions=[item["condition"]], inputs=root / "data/v3/pilot_inputs.jsonl",
                            config=root / CONFIG, prompts=root / "prompts", expected_count=50, split="pilot", run_tag="pilot_v3")
        require(report.get("ok") is True, "Pilot fresh physical verification failed", "blocked_pilot_go")
        stored = read_json(directory / "verification.json")
        require(stored.get("ok") is True and all(stored["runs"][item["condition"]].get(k) == report["runs"][item["condition"]].get(k)
                for k in ("archive_sha256", "notebook_sha256", "config_hash", "records", "condition")), "Pilot stored evidence changed")
        files = [Path(report["runs"][item["condition"]]["archive"])]
        files += list((directory / "artifacts").glob(NOTEBOOKS[item["condition"]] + ".out.*.ipynb"))
        require(len(files) == 2, "Pilot raw notebook missing/ambiguous")
        files += [directory / n for n in ("verification.json", "status.json", "execution_handover.json")]
        for source in files:
            require(source.resolve().is_relative_to(directory.resolve()) and not any(p.is_symlink() for p in (source, *source.parents)),
                    "Canonical pilot source escaped owned output")
            relative = Path(item["condition"]) / ("artifacts" if source.suffix in {".zip", ".ipynb"} else "") / source.name
            expected_paths.add(relative.as_posix())
            sources.append((source, destination / relative))
    if destination.exists():
        require(not any(p.is_symlink() or p.is_file() and p.relative_to(destination).as_posix() not in expected_paths
                        for p in destination.rglob("*")), "Unexpected canonical pilot evidence")
    for source, target in sources:
        require(target.resolve().is_relative_to(destination.resolve()) and not any(
                p.is_symlink() for p in (target, target.with_name(target.name + ".pending"), *target.parents)),
                "Canonical pilot target escaped or contains symlink")
        if target.exists():
            require(digest(target) == digest(source), "Canonical pilot evidence changed")
    for source, target in sources:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".pending")
            shutil.copyfile(source, tmp)
            require(digest(tmp) == digest(source), "Pilot source changed during collection")
            tmp.replace(target)
            target.chmod(0o444)
    return destination


def status_gate(root, output, item):
    output = Path(output).resolve()
    state = read_json(output / "status.json")
    require(state.get("series") == "v3" and state.get("root") == str(Path(root).resolve())
            and state.get("output") == str(output) and state.get("job") == item
            and state.get("cli_config") == str(output / "sessions.json"), "Status root/output/job identity mismatch")
    return state


def start_operator(root, output, item, options):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / "results/v3") and output != root / "results/v3", "Output must be a new isolated results/v3 job directory")
    if options.get("attempt_prefix") is not None:
        require(output == root / "results/v3/attempts" / options["attempt_prefix"] / item["key"],
                "Infrastructure retry output must use its isolated namespace", "blocked_ownership")
    evidence = preflight(root, item, **{k: options.get(k) for k in
                         ("approval", "pilot_report", "initial_budget", "authorize_initial", "authorize_confirmatory", "dev_bundle", "attempt_prefix")})
    if item["phase"] == "analysis":
        analysis_inputs(root, options["approval"])
    host = host_gate(options.get("host_ack"), root, job_seconds=item["deadline_seconds"])
    require(Path(CLI_PYTHON).is_file() and os.access(CLI_PYTHON, os.X_OK), "CLI Python environment unavailable", "blocked_host_identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with flock("/tmp/jev-v3-operator-" + item["session"] + ".lock") as fd:
        with flock("/tmp/jev-colab-allocation.lock", wait=120):
            current_host = host_gate(options.get("host_ack"), root, job_seconds=item["deadline_seconds"])
            require(current_host["host_lease_id"] == host["host_lease_id"], "Host admission changed during startup", "blocked_host_identity")
            require(not output.exists() or not any(output.iterdir()), "Prior submission/output exists; never retry or restart", "blocked_ownership")
            output.mkdir(exist_ok=True)
            state = dict(series="v3", root=str(root), output=str(output), cli_config=str(output / "sessions.json"),
                         job=item, options=options, host_identity=current_host, evidence=evidence, run_id=uuid.uuid4().hex,
                         start_epoch=time.time(), status="submission_intent", startup_ack=False, verified=False,
                         released=False, allocation_attempted=False, operator_lock_fd=fd, clock_boot_id=BOOT_ID,
                         operator_lock=str(Path("/tmp/jev-v3-operator-" + item["session"] + ".lock")))
            state["deadline_epoch"] = state["start_epoch"] + item["deadline_seconds"]
            state["deadline_monotonic"] = time.monotonic() + item["deadline_seconds"]
            state["work_deadline_monotonic"] = state["deadline_monotonic"] - CLEANUP_SECONDS
            atomic_json(output / "status.json", state)
            atomic_json(output / "execution_handover.json", state)
        with (output / "operator.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(root / "tools/colab/run_v3.py"), "--root", str(root), "--_worker", str(output)],
                                     stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, pass_fds=(fd,))
        until = time.time() + 10
        while time.time() < until:
            record = status_gate(root, output, item)
            if record.get("startup_ack") is True and record.get("operator_pid") == child.pid:
                return record
            require(child.poll() is None, "Detached worker exited before durable acknowledgement; do not retry", "blocked_startup")
            time.sleep(.1)
        raise Blocked("blocked_startup", "Detached worker startup unknown; reconcile durable status, never resubmit")


def reconcile_budget_only(root, host_ack):
    """Explicit online cleanup/identity operation, never reachable from --plan."""
    root = Path(root).resolve()
    host = host_gate(host_ack, root, True)
    output = root / "results/v3"
    output.mkdir(parents=True, exist_ok=True)
    with flock("/tmp/jev-colab-allocation.lock", wait=120):
        with flock("/tmp/jev-v3-budget.lock", wait=120):
            ledger = budget_ledger(root)
            atomic_json(root / BUDGET_LEDGER, ledger)
            state = dict(job=job("ANALYSIS", "analysis"), deadline_epoch=time.time() + CLEANUP_SECONDS)
            Operator(root, output, state).reconcile_budget(ledger)
            pending = 0
            for path in output.rglob("status.json"):
                if any(p in {"remote", "pilot_verified", "artifacts", "checkpoints"} for p in path.relative_to(output).parts):
                    continue
                current = read_json(path)
                item = current.get("job", {})
                require(current.get("root") == str(root) and current.get("output") == str(path.parent)
                        and current.get("series") == "v3" and item == job(item["condition"], item["phase"], item["session"]),
                        "Owned operator identity uncertain during host drain", "blocked_ownership")
                try:
                    with flock("/tmp/jev-v3-operator-" + item["session"] + ".lock"):
                        if current.get("status") not in {"completed", "failed"}:
                            entry = ledger["jobs"].get(current.get("run_id"))
                            if (entry and entry.get("released") is True and entry.get("release_verified") is True
                                    and entry.get("release_evidence") == "backend_absent"):
                                Operator(root, path.parent, current).save("failed", released=True, verified=False, completed_execution=False,
                                    release_evidence="backend_absent", observed_gpu_seconds_upper_bound=entry["actual_seconds"],
                                    reconciled_without_work_renewal=True)
                            else:
                                pending += 1
                        elif item["gpu"] == "CPU" and current.get("allocation_attempted") and not current.get("owned_endpoint"):
                            pending += 1  # Unknown endpoint is not authoritative absence; never touch foreign assignments.
                        elif item["gpu"] == "CPU" and current.get("owned_endpoint"):
                            data = Backend(path.parent, item["session"]).identity()
                            if current["owned_endpoint"] in {r["endpoint"] for r in data["assignments"]}:
                                if time.time() >= current.get("deadline_epoch", float("inf")):
                                    prior = Operator(root, path.parent, current)
                                    prior.deadline = time.time() + CLEANUP_SECONDS
                                    prior.work_deadline = time.time()
                                    prior.monotonic_deadline = time.monotonic() + CLEANUP_SECONDS
                                    prior.monotonic_work_deadline = time.monotonic()
                                    require(prior.release(), "Expired owned CPU release unconfirmed", "blocked_ownership")
                                else:
                                    pending += 1
                except Blocked as error:
                    if error.state != "blocked_ownership":
                        raise
                    pending += 1
            live = sum(e.get("released") is not True for e in ledger["jobs"].values())
            return dict(schema_version=1, experiment_id=EXPERIMENT_ID, host_lease_id=host.get("host_lease_id"),
                        ledger=str(root / BUDGET_LEDGER), ledger_sha256=digest(root / BUDGET_LEDGER), verified_utc=utc(),
                        charged_or_reserved_seconds=budget_spent(ledger), max_gpu_seconds=MAX_GPU_SECONDS,
                        live_leases=live, pending_operators=pending,
                        safe_to_stop_host=live == pending == 0 and host.get("accept_allocations") is False, allocation_permitted=False)


class Coordinator:
    def __init__(self, root=ROOT, phase="prepared", **options):
        self.root = Path(root).resolve()
        self.out = self.root / "results/v3"
        self.phase, self.options, self.jobs = phase, options, plan(phase)
        self.attempt_prefix = options.get("attempt_prefix")
        if self.attempt_prefix is not None:
            require(phase in {"smoke", "prepared", "full"} and options.get("authorize_initial") is True,
                    "Infrastructure attempt must include explicitly authorized dev smoke", "blocked_user_authorization")
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", self.attempt_prefix), "Unsafe attempt prefix", "blocked_ownership")
            self.jobs = [job(j["condition"], j["phase"], j["session"] + "-" + self.attempt_prefix)
                         if j["phase"] == "smoke" else j for j in self.jobs]
        self.path = self.out / ("coordinator_" + phase + ("_" + self.attempt_prefix if self.attempt_prefix else "") + "_state.json")
        self.state = dict(series="v3", root=str(self.root), phase=phase, jobs=self.jobs, submissions={}, verified_jobs={}, max_total_assignments=2)

    def job_directory(self, item):
        base = self.out / "attempts" / self.attempt_prefix if self.attempt_prefix and item["phase"] == "smoke" else self.out
        return base / item["key"]

    def save(self, outcome, **fields):
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.update(outcome=outcome, updated_utc=utc(), **fields)
        atomic_json(self.path, self.state)

    def run(self):
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            with flock(self.out / ("coordinator_" + self.phase + ".lock")):
                if self.path.exists():
                    saved = read_json(self.path)
                    require(saved.get("series") == "v3" and saved.get("root") == str(self.root)
                            and saved.get("jobs") == self.jobs, "Coordinator identity changed", "blocked_ownership")
                    self.state = saved
                self.startup_gate()
                # Durable CPU-only progress precedes slow full certification and all API work.
                self.save("validating_local", validation_job=None, validation_completed=[], validation_started_utc=utc())
                initial_jobs = [j for j in self.jobs if j["phase"] in {"smoke", "pilot"}] if self.phase == "full" else self.jobs
                evidence = {}
                for item in initial_jobs:
                    self.save("validating_local", validation_job=item["key"])
                    evidence[item["key"]] = preflight(self.root, item, **self.job_options(item))
                    self.state["validation_completed"].append(item["key"])
                    self.save("validating_local", validation_job=item["key"], validation_job_done_utc=utc())
                if self.phase in {"test", "latency"}:
                    approved, report = approval_gate(self.root, self.phase, self.options.get("approval"), self.options.get("pilot_report"),
                                                     self.options.get("authorize_confirmatory"))
                    estimates = estimate_jobs(self.root, remaining_confirmatory(self.root), approved, report)
                else:
                    estimates = {}
                host = host_gate(self.options.get("host_ack"), self.root,
                                 job_seconds=max(j["deadline_seconds"] for j in self.jobs))
                return self.execute(evidence, estimates, host)
        except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            fields = dict(error_type=type(error).__name__, error_message=str(error)[:500] if isinstance(error, Blocked)
                          else diagnosis(error)["error_message"])
            if getattr(error, "state", None) == "blocked_ownership" and self.path.exists():
                self.state.update(outcome="blocked_ownership", **fields)  # Foreign/live coordinator disk state is untouched.
            else:
                self.save(getattr(error, "state", "blocked_execution"), **fields)
            return 1

    def startup_gate(self):
        """Phase-specific local checks before any fresh pilot verification."""

    def job_options(self, item):
        options = {k: self.options.get(k) for k in
                    ("approval", "pilot_report", "initial_budget", "authorize_initial", "authorize_confirmatory", "dev_bundle")}
        options["attempt_prefix"] = self.attempt_prefix if item["phase"] == "smoke" else None
        if self.phase == "full" and item["phase"] in {"smoke", "pilot"}:
            options.update(approval=None, pilot_report=None, authorize_confirmatory=False)
        return options

    def decide_pilot(self):
        from pilot_decision_v3 import decide, write_report
        report = decide(canonical_pilot(self.root), self.root / "data/v3/pilot_inputs.jsonl", config=self.root / CONFIG,
                        prompts=self.root / "prompts", run_tag="pilot_v3")
        report_id = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        path = self.out / "pilot" / ("decision_" + report_id + ".json")
        if path.exists():
            stored = read_json(path)
            require({k: v for k, v in stored.items() if k not in {"paths", "meta"}} == report, "Pinned pilot decision changed")
        else:
            require(not path.with_suffix(".md").exists(), "Pinned pilot markdown already exists")
            write_report(report, path)
        fields = dict(technical_go=report["go"], pilot_report=str(path), pilot_report_sha256=digest(path), budget_ready=False)
        if report["go"]:
            try:
                estimates = estimate_jobs(self.root, remaining_confirmatory(self.root), {"max_gpu_hours": 25}, report)
                fields.update(remaining_estimates=estimates, remaining_gpu_seconds=MAX_GPU_SECONDS - budget_spent(budget_ledger(self.root)))
                fields["budget_ready"] = sum(estimates.values()) <= fields["remaining_gpu_seconds"]
            except Blocked as error:
                fields["budget_error"] = str(error)
        self.save("pilot_go" if report["go"] else "pilot_no_go", **fields)
        return report

    def execute(self, evidence, estimates, host):
        try:
            if self.state.get("frozen_evidence"):
                require(all(evidence.get(k, v) == v for k, v in self.state["frozen_evidence"].items()),
                        "Frozen coordinator sources/evidence changed; no further submission")
            evidence = {**self.state.get("frozen_evidence", {}), **evidence}
            batches = []
            for phase in ("smoke", "pilot", "test", "latency", "analysis"):
                stage = [item for item in self.jobs if item["phase"] == phase]
                batches.extend(stage[i:i + 2] for i in range(0, len(stage), 2))
            batch_plan = [[item["key"] for item in batch] for batch in batches]
            require(self.state.get("batch_plan", batch_plan) == batch_plan, "Recorded source-order batches changed")
            self.save("ready", frozen_evidence=evidence, estimates=estimates, host_identity=host, batch_plan=batch_plan)
            for batch in batches:
                if self.phase == "full" and batch[0]["key"] == "test/G_SINGLE":
                    report = self.decide_pilot()
                    require(report["go"], "Technical pilot NO-GO; no full promotion", "blocked_pilot_go")
                    require(self.state["budget_ready"], "Technical GO but full remaining matrix + LAT budget not ready", "blocked_budget_or_eta")
                    require(self.options.get("pilot_report") and digest(self.options["pilot_report"]) == self.state["pilot_report_sha256"],
                            "Parent must supply actual post-pilot confirmation bound to immutable report; no autogenerated consent", "blocked_budget_approval")
                    approved, actual_report = approval_gate(self.root, "full", self.options.get("approval"), self.options.get("pilot_report"),
                                                           self.options.get("authorize_confirmatory"))
                    estimates = estimate_jobs(self.root, remaining_confirmatory(self.root), approved, actual_report)
                for item in self.jobs:
                    if item["phase"] == batch[0]["phase"] and item["key"] not in evidence:
                        evidence[item["key"]] = preflight(self.root, item, **self.job_options(item))
                self.save("ready", frozen_evidence=evidence, estimates=estimates)
                active = []
                for item in batch:
                    directory = self.job_directory(item)
                    if (directory / "status.json").exists():
                        state = status_gate(self.root, directory, item)
                        require(state.get("startup_ack") is True, "Prior startup ambiguous; no resubmission", "blocked_startup")
                    else:
                        for previous in active[:]:
                            current = status_gate(self.root, self.job_directory(previous), previous)
                            while current["status"] not in {"completed", "failed"} and not (
                                    current.get("owned_endpoint") and current.get("budget_reserved_seconds", 0) > 0):
                                require(time.time() < current["deadline_epoch"] and time.monotonic() < current.get("deadline_monotonic", float("inf")),
                                        "Prior allocation ownership/reservation uncertain; no paired launch", "blocked_startup")
                                time.sleep(POLL_SECONDS)
                                current = status_gate(self.root, self.job_directory(previous), previous)
                            if current["status"] in {"completed", "failed"}:
                                self.wait_operator(previous)
                                active.remove(previous)
                        probe = self.out / "capacity_probe"
                        probe.mkdir(exist_ok=True)
                        capacity = Backend(probe, "jev-v3-capacity-probe").identity()
                        require(capacity.get("local_endpoint") is None, "Foreign capacity-probe session; untouched", "blocked_ownership")
                        while len(capacity["assignments"]) >= 2 and active:
                            self.wait_operator(active.pop(0))  # A foreign slot forces serial execution, not release of that slot.
                            capacity = Backend(probe, "jev-v3-capacity-probe").identity()
                        require(len(capacity["assignments"]) < 2, "Global assignment cap; no submission and unknowns untouched", "blocked_capacity")
                        try:
                            prepared = preflight(self.root, item, **self.job_options(item))
                        except Blocked as error:
                            if error.state != "blocked_budget_or_eta" or not active:
                                raise
                            for previous in active:
                                self.wait_operator(previous)
                            active.clear()
                            prepared = preflight(self.root, item, **self.job_options(item))
                        require(prepared == evidence[item["key"]], "Frozen phase evidence changed before submission")
                        require(item["key"] not in self.state["submissions"], "Prior submission intent is ambiguous; no restart", "blocked_startup")
                        self.state["submissions"][item["key"]] = dict(status="intent", utc=utc())
                        self.save("submission_intent")
                        state = start_operator(self.root, directory, item, {**self.options, **self.job_options(item)})
                        require(state.get("startup_ack") is True, "Paired worker startup unproved", "blocked_startup")
                    active.append(item)
                for item in active:
                    self.wait_operator(item)
            if self.phase in {"prepared", "pilot"}:
                report = self.decide_pilot()
                return 0 if report["go"] else 1
            self.save("completed")
            return 0
        except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            self.save(getattr(error, "state", "blocked_execution"), error_type=type(error).__name__,
                      error_message=str(error)[:500] if isinstance(error, Blocked) else diagnosis(error)["error_message"])
            return 1

    def wait_operator(self, item):
        directory = self.job_directory(item)
        state = status_gate(self.root, directory, item)
        until = state["deadline_epoch"] + 30
        while state["status"] not in {"completed", "failed"}:
            require(time.time() < until and time.monotonic() < state.get("deadline_monotonic", float("inf")) + 30,
                    "Owned worker deadline reached; reconcile only, never inline restart", "blocked_deadline")
            time.sleep(min(POLL_SECONDS, until - time.time()))
            state = status_gate(self.root, directory, item)
        require(state.get("status") == "completed" and state.get("verified") is True
                and state.get("released") is True and state.get("completed_execution") is True,
                "Operator failed or release unconfirmed; stage barrier closed", "blocked_execution")
        data = Backend(directory, item["session"]).identity()
        require(state.get("owned_endpoint") and state["owned_endpoint"] not in {r["endpoint"] for r in data["assignments"]}, "Owned endpoint still active")
        operator = Operator(self.root, directory, state)
        operator.deadline = operator.work_deadline = time.time() + 120
        operator.monotonic_deadline = operator.monotonic_work_deadline = time.monotonic() + 120
        operator.verify()
        self.state["verified_jobs"][item["key"]] = read_json(directory / "verification.json")
        self.save("job_verified_and_released")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Offline readiness and schedule; no API/auth")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--operator-start", action="store_true")
    mode.add_argument("--operator-status", action="store_true")
    mode.add_argument("--reconcile-budget", action="store_true", help="Online owned-only lease reconciliation/cleanup; no allocations or renewals")
    mode.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("prepared", "full", "smoke", "pilot", "test", "latency", "analysis"), default="prepared")
    parser.add_argument("--session")
    parser.add_argument("--condition", choices=tuple(NOTEBOOKS), default="JFINAL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dev-bundle", action="store_true")
    parser.add_argument("--attempt-prefix", help="Isolated retry of released infrastructure-failed dev smoke only; never quality/pilot/test reruns")
    parser.add_argument("--authorize-initial", action="store_true", help="User authorizes smoke/technical pilot only")
    parser.add_argument("--authorize-confirmatory", "--authorized", action="store_true", help="User supplies actual separate config/pilot-pinned budget approval")
    for name in ("approval", "pilot-report", "initial-budget", "host-ack"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    options = {k: str(getattr(args, k).resolve()) if getattr(args, k) is not None else None for k in ("approval", "pilot_report", "initial_budget", "host_ack")}
    options.update(authorize_initial=args.authorize_initial, authorize_confirmatory=args.authorize_confirmatory,
                   dev_bundle=args.dev_bundle, attempt_prefix=args.attempt_prefix)
    if args.reconcile_budget:
        try:
            print(json.dumps(reconcile_budget_only(root, options["host_ack"]), indent=2))
            return 0
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(json.dumps(dict(status=getattr(error, "state", "blocked_budget_or_eta"), error_type=type(error).__name__)))
            return 1
    if args._worker:
        out = args._worker.resolve()
        state = read_json(out / "status.json")
        require(state.get("root") == str(root) and state.get("output") == str(out) and state.get("series") == "v3"
                and state.get("status") == "submission_intent" and state.get("startup_ack") is False,
                "Worker identity/startup intent mismatch")
        status_gate(root, out, state["job"])
        lock_path = "/tmp/jev-v3-operator-" + state["job"]["session"] + ".lock"
        require(state.get("operator_lock") == lock_path and type(state.get("operator_lock_fd")) is int,
                "Worker lacks inherited per-session lock")
        inherited, original = os.fstat(state["operator_lock_fd"]), Path(lock_path).stat()
        require((inherited.st_dev, inherited.st_ino) == (original.st_dev, original.st_ino), "Worker lock identity changed")
        worker = Operator(root, out, state)
        worker.save("validating_local", startup_ack=True, operator_pid=os.getpid(), validation_job=state["job"]["key"])
        try:
            evidence = preflight(root, state["job"], **{k: state["options"].get(k) for k in
                                ("approval", "pilot_report", "initial_budget", "authorize_initial", "authorize_confirmatory", "dev_bundle", "attempt_prefix")}, current_intent=state)
            require(evidence == state["evidence"], "Prepared evidence changed before durable startup; no allocation")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            require(state.get("allocation_attempted") is False and not state.get("owned_endpoint"), "Startup failure ownership uncertain")
            worker.save("failed", released=True, completed_execution=False, verified=False,
                        release_evidence="durable_no_allocation", error_type=type(error).__name__,
                        error_message=diagnosis(CoordinatorError(str(error)) if isinstance(error, Blocked)
                                                else error)["error_message"])
            return 1
        def interrupt(signum, frame):
            raise InterruptedError("Operator interrupted; preserve and release owned endpoint")
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        worker.save("validated_local", validation_job_done_utc=utc())
        return worker.run()
    if args.operator_start or args.operator_status:
        require(args.session and args.output and args.phase not in {"prepared", "full"}, "Operator requires session, output and explicit phase")
        item = job(args.condition, args.phase, args.session)
        if args.operator_status:
            print(json.dumps(status_gate(root, args.output, item), indent=2))
            return 0
        try:
            record = start_operator(root, args.output, item, options)
            print(json.dumps(dict(status=record["status"], output=record["output"], startup_ack=True)))
            return 0
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(json.dumps(dict(status=getattr(error, "state", "blocked_execution"), error_type=type(error).__name__,
                                  message=str(error)[:500] if isinstance(error, Blocked) else diagnosis(error)["error_message"])))
            return 1
    if args.run:
        coordinator = Coordinator(root, args.phase, **options)
        rc = coordinator.run()
        print(json.dumps(coordinator.state, indent=2))
        return rc
    readiness = []
    planner = Coordinator(root, args.phase, **options)
    for item in planner.jobs:
        try:
            evidence = preflight(root, item, **planner.job_options(item))
            readiness.append(dict(key=item["key"], status="ready_local", bundle=evidence["bundle"]))
        except (ValueError, OSError, KeyError, TypeError) as error:
            readiness.append(dict(key=item["key"], status=getattr(error, "state", "blocked_integrity"),
                                  message=str(error)[:500] if isinstance(error, Blocked) else diagnosis(error)["error_message"]))
    print(json.dumps(dict(offline=True, allocation_permitted=False, phase=args.phase, jobs=planner.jobs, readiness=readiness,
                         obligations="Actual data/policy/review seal, known CIM/WSL host acknowledgement, live host, durable startup; no quality retries/restarts",
                         confirmatory="Explicit --authorize-confirmatory + approval JSON config_sha256, pilot_report_sha256, scope, approved/by/utc, max_gpu_hours; never autogenerated",
                          host_ack="acknowledged, host_id (actual CIM UUID), wsl_distro, keep_wsl_alive, keep_host_awake, expires_utc (>=4h)",
                          initial_budget="Parent supplied --initial-budget: approved/by/utc/authorization_source, experiment_id=" + EXPERIMENT_ID
                              + ", scope=all_gpu_phases_conditional_technical_go, max_gpu_hours=25, max_total_assignments=2, gpu=A100, technical_go_required=true, config_sha256, implementation_profile, models",
                          accumulated_budget=dict(ledger=BUDGET_LEDGER, experiment_id=EXPERIMENT_ID, max_gpu_hours=25, phases=["smoke", "pilot", "test", "latency"]),
                         policy=dict(max_global_assignments=2, allocation_lock="/tmp/jev-colab-allocation.lock", job_deadline_seconds=JOB_SECONDS,
                                     poll_seconds=POLL_SECONDS, recovery_seconds=RECOVERY_SECONDS, controls_minutes=CONTROLS, ws_ping_seconds=15, backend_check_seconds=20)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
