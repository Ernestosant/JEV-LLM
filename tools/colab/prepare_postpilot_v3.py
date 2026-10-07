"""CPU-only, write-once preparation of the user-authorized 25h amendment.

Run under the same WSL root as the historical receipts, after source hashes settle.
This does not build bundles, re-audit with the original verifier, or contact Colab.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as r
from postpilot_v3 import PostPilotCoordinator
from jevlab.common import config_hash
from jevlab.v3.operational import build_calendar, validate_code_transition, POLICY_ID

PILOT_SHA256 = "032a4231146e30c47ef8a64c2fefdb6f0ceace4f0a8e27d8a2fde427af0b925a"
ASSETS_SHA256 = "c1945ea8b135a53a80225826afa01f033469abd7a63bc8020a83edd9733c938d"
AUTHORIZATION = ("User explicitly authorized Replanificar dentro de 25h: balanced B/H/B "
                 "latency reload calendar and 6h individual new jobs; preserve 25h total, "
                 "scientific config/data/models/statistics, 6000 test and 1800 latency rows. "
                 "No additional allowance, sharding, or reduced observations.")
STAGES = ("environment", "snapshots", "preflight_pre_engines", "engines",
          "preflight_post_engines", "warmup")


def safe_path(path):
    path = Path(path)
    r.require(not any(p.is_symlink() for p in (path, *path.parents)),
              "Symlink preparation path forbidden: " + str(path))
    return path


def manifest_snapshot(path, expected_sha):
    """Read only the public manifest of an already hash-verified historical ZIP."""
    path = safe_path(path)
    r.require(path.stat().st_size <= r.MAX_COLLECTION_BYTES and r.digest(path) == expected_sha,
              "Historical timing archive SHA/size changed")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        r.require(len(names) <= 10000 and len(names) == len({n.casefold().rstrip('/') for n in names})
                  and all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                          and '\\' not in n and ':' not in n and 'gold' not in n.lower() for n in names),
                  "Unsafe/duplicate/nonpublic historical archive")
        manifests = [n for n in names if n == "manifest.json" or n.endswith("/manifest.json")]
        r.require(len(manifests) == 1 and archive.getinfo(manifests[0]).file_size <= 4 * 1024 * 1024,
                  "Historical archive requires one bounded manifest")
        with archive.open(manifests[0]) as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        r.require(len(raw) <= 4 * 1024 * 1024, "Oversized historical manifest")
        manifest = json.loads(raw)
    r.require(r.digest(path) == expected_sha, "Historical archive changed during read")
    return manifest


def historical_estimates(root, jobs, approved, report, manifests):
    """Exact estimate_jobs formula, without its current original-verifier reader.

    Historical hashes/coverage are validated separately, not re-certified here.
    Keep the original 50% margin, 900s startup and 600s cleanup unchanged.
    """
    estimates, evidence = {}, {}
    for item in jobs:
        c = item["condition"]
        arms = ("JFINAL", "B13_GREEDY") if c == "LATENCY" else (c,)
        timings, hot = {}, 0.0
        for arm in arms:
            entry = report["conditions"][arm]
            amount = entry.get("latency_estimate" if c == "LATENCY" else "estimate", {}).get("estimated_test_walltime_s")
            stages = manifests[arm].get("stages", {})
            values = {n: stages.get(n, {}).get("seconds") for n in STAGES}
            r.require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0
                      and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())
                      and all(stages.get(n, {}).get("ok") is True for n in STAGES),
                      "Missing measured historical hot/setup timings: " + arm, "blocked_budget_or_eta")
            hot += amount
            timings[arm] = values
        loads = {}
        if c == "LATENCY":
            phases, previous = Counter(), None
            for row in r.operational_calendar(root, approved):
                phase = "B" if row["condition"] == "B13_GREEDY" else "H"
                if phase != previous:
                    phases[phase] += 1
                    previous = phase
            r.require(phases == {"B": 2, "H": 1}, "Unexpected latency reload commitments")
            loads = dict(phases)
            setup = sum(max(t[n] for t in timings.values()) for n in ("environment", "preflight_pre_engines"))
            setup += sum(t["snapshots"] for t in timings.values())
            setup += sum(phases[p] * sum(timings[a][n] for n in ("engines", "preflight_post_engines", "warmup"))
                         for p, a in (("B", "B13_GREEDY"), ("H", "JFINAL")))
        else:
            setup = sum(sum(t.values()) for t in timings.values())
        seconds = 1.5 * (hot + setup) + 900 + r.CLEANUP_SECONDS
        r.require(seconds <= item["deadline_seconds"] == 21600,
                  "Measured ETA exceeds unsharded 6h deadline: " + c, "blocked_budget_or_eta")
        estimates[item["key"]] = seconds
        evidence[item["key"]] = dict(hot_seconds=hot, setup_seconds=setup, stages=timings, load_counts=loads)
    return estimates, evidence


def prepare(root, authorize_replan=False):
    r.require(authorize_replan is True, "Explicit --authorize-replan-within-25h required", "blocked_user_authorization")
    root = safe_path(Path(root).absolute()).resolve()
    out = safe_path(root / "results/v3/operational_amendment")
    approval_path, record_path = out / "approval.json", out / "preregistration_eta.json"
    for path in (approval_path, record_path):
        safe_path(path)
        r.require(not path.exists(), "Immutable preparation output already exists: " + str(path))
    history_path = root / "results/v3/coordinator_full_infra03_recovery_state.json"
    receipt_path = root / "results/v3/recovery/infra03/recovery_receipt.json"
    assets_path = out / "pre_amendment_assets.zip"
    initial_path = root / "results/v3/initial_budget.json"
    history = r.read_json(safe_path(history_path))
    r.require(history.get("root") == str(root),
              "Use the original WSL root; historical receipt identities must not be translated or rewritten")
    report_path = safe_path(Path(history["pilot_report"]))
    r.require(report_path.resolve().is_relative_to(root / "results/v3/pilot")
              and r.digest(report_path) == history.get("pilot_report_sha256") == PILOT_SHA256
              and r.digest(safe_path(assets_path)) == ASSETS_SHA256, "Actual pilot/preserved asset pin changed")
    initial = r.initial_budget_gate(root, safe_path(initial_path))
    # Runtime gates below bind the public inputs; bundle/preflight owns the full seal audit.
    report = r.pilot_gate(root, report_path, fresh=False)
    archives = r.pilot_inputs(root, report_path)
    manifests = {c: manifest_snapshot(p, report["conditions"][c]["verification"]["archive_sha256"])
                 for c, p in archives.items()}
    old = manifests[r.CONDITIONS[0]]["config"]["code_sha256"]
    r.require(all(m["config"]["code_sha256"] == old for m in manifests.values()),
              "Historical pilot runtime inventories differ")
    package = root / "src/jevlab"
    new = {p.relative_to(package).as_posix(): r.digest(safe_path(p)) for p in
           sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
    data = root / "data/v3"
    items = [json.loads(line) for line in safe_path(data / "latency_inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    order = r.read_json(safe_path(data / "schedule.json"))["order"]
    metadata = {row["item_id"]: row for row in r.read_json(safe_path(data / "latency_plan.json"))["items"]}
    calendar = build_calendar(items, order, metadata)
    now = r.utc()
    r.require(r.epoch(now) >= r.epoch(report["meta"]["generated_utc"]), "Approval cannot predate actual pilot")
    policy = dict(id=POLICY_ID, approved=True, approved_by="user", approved_utc=now,
                  config_sha256=r.digest(root / r.CONFIG), pilot_report_sha256=PILOT_SHA256,
                  old_code_sha256=old, new_code_sha256=new, max_job_seconds=21600, max_gpu_seconds=90000,
                  expected_test_rows=6000, expected_latency_rows=1800, seeds=[17, 29, 43], repetitions=3,
                  load_counts={"B": 2, "H": 1}, latency_source_plan_sha256=r.digest(data / "latency_plan.json"),
                  latency_schedule_sha256=r.digest(data / "schedule.json"), calendar_sha256=config_hash(calendar),
                  authorization_source=AUTHORIZATION)
    policy.update({"cohort_" + c.lower(): list(dict.fromkeys(row["item_id"] for row in calendar if row["cohort"] == c))
                   for c in ("A", "C")})
    approved = dict(approved=True, scope="test_and_latency", experiment_id=r.EXPERIMENT_ID, max_gpu_hours=25,
                    config_sha256=policy["config_sha256"], pilot_report_sha256=PILOT_SHA256,
                    initial_authorization_sha256=hashlib.sha256(json.dumps(initial, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode()).hexdigest(), approved_by="user", approved_utc=now,
                    no_additional_allowance_after_pilot=True, authorization_source=AUTHORIZATION,
                    operational_amendment=policy, new_verifier_sha256=r.digest(safe_path(root / "tools/verify_run_v3.py")),
                    recovery_state_sha256=r.digest(history_path), historical_recovery_receipt_sha256=r.digest(safe_path(receipt_path)),
                    preserved_assets_sha256=ASSETS_SHA256)
    validate_code_transition(old, new, approved, policy["config_sha256"], PILOT_SHA256)
    ledger = r.budget_ledger(root)
    r.require(all(e.get("released") is True and e.get("release_verified") is True for e in ledger["jobs"].values()),
              "All prior GPU leases must be authoritatively released", "blocked_budget_or_eta")
    jobs = [*r.plan("test"), *r.plan("latency")]
    # Preparation is preregistration, never continuation/reverification of test outputs.
    r.require(not any((root / "results/v3" / j["key"] / "status.json").exists() for j in jobs),
              "Confirmatory jobs already exist; preparation must precede new execution")
    r.require(sum(j["expected_rows"] for j in jobs if j["phase"] == "test") == 6000
              and jobs[-1]["expected_rows"] == len(calendar) == 1800 and r.MAX_GPU_SECONDS == 90000,
              "Scientific observation counts/total changed")
    estimates, timing_evidence = historical_estimates(root, jobs, approved, report, manifests)
    spent = r.budget_spent(ledger)
    remaining = sum(estimates.values())
    r.require(spent + remaining <= 90000, "Measured remaining ETA exceeds original accumulated 25h budget", "blocked_budget_or_eta")
    # Bind the full source/data tree and historical proofs; detect changes before publication.
    inputs = [history_path, receipt_path, assets_path, initial_path, report_path, root / r.CONFIG,
              root / r.BUDGET_LEDGER, *archives.values(), *package.glob("*.py"), *(package / "v3").glob("*.py"),
              *(root / "tools").glob("*.py"), *(root / "tools/colab").glob("*.py"),
              *(root / "data/v3").glob("*"), *(root / "prompts").glob("*.txt")]
    source_pins = {str(p): r.digest(safe_path(p)) for p in inputs if p.is_file()}
    payload = (json.dumps(approved, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.TemporaryDirectory(prefix="jev-postpilot-cpu-") as temp:
        staged = Path(temp) / "approval.json"
        staged.write_bytes(payload)
        # Avoid Coordinator.__init__: it can create coordinator state. Only its read-only history gate is used.
        checker = PostPilotCoordinator.__new__(PostPilotCoordinator)
        checker.root, checker.out = root, root / "results/v3"
        checker.recovery_path, checker.receipt_path, checker.assets_path = history_path, receipt_path, assets_path
        checker.options = dict(approval=str(staged), initial_budget=str(initial_path), pilot_report=str(report_path))
        checker.validate_history()
        for item in (jobs[0], jobs[-1]):
            r.runtime_approval_gate(root, item, staged, report_path)
    r.require(source_pins == {p: r.digest(safe_path(p)) for p in source_pins}
              and r.budget_ledger(root) == ledger, "Preparation inputs changed during validation")
    record = dict(schema_version=1, cpu_only=True, allocation_permitted=False, generated_utc=now,
                  approval=str(approval_path), approval_sha256=hashlib.sha256(payload).hexdigest(),
                  experiment_id=r.EXPERIMENT_ID, max_gpu_seconds=90000, spent_gpu_seconds=spent,
                  remaining_budget_seconds=90000 - spent, remaining_estimates=estimates,
                  remaining_eta_seconds=remaining, projected_total_gpu_seconds=spent + remaining,
                  formula="1.5 * (measured_hot + measured_setup) + 900 + 600",
                  timing_evidence=timing_evidence, calendar=calendar, frozen_files=source_pins,
                  historical_evidence="Pinned actual GO and recorded historical verification; no original-verifier re-audit")
    out.mkdir(parents=True, exist_ok=True)
    # Publish approval last: an interrupted record write must never authorize execution.
    for path, content in ((record_path, (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()),
                          (approval_path, payload)):
        safe_path(path)
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(approval=str(approval_path), preregistration_eta=str(record_path), approval_sha256=record["approval_sha256"],
                remaining_eta_seconds=remaining, projected_total_gpu_seconds=spent + remaining, cpu_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--authorize-replan-within-25h", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.root, args.authorize_replan_within_25h)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps(dict(outcome=getattr(error, "state", "blocked_integrity"), error=str(error), cpu_only=True)))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
