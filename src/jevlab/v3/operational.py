"""Pure, approval-bound operational ordering; no inference or scoring changes."""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter

from ..common import config_hash

POLICY_ID = "jev-v3-balanced-residencies-25h-20261006"
MAX_NEW_JOB_SECONDS = 21600
TOTAL_GPU_SECONDS = 90000
MAX_GLOBAL_JOBS = 2
ALLOWED_CODE_DELTA = frozenset({"v3/latency_study.py", "v3/runner.py", "v3/operational.py"})
residency_phases = ["B", "H", "B"]
DOMAINS = {"arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"}


def build_calendar(items, schedule_order, metadata):
    """Split alternating frozen schedule positions within each public stratum.

    items are blind input dicts; metadata maps item_id to frozen plan records.
    Repetition is zero-based, block is the original schedule-filtered index // 10,
    and residency is zero-based (0=B/A, 1=H/all, 2=B/C).
    """
    ids = [it["id"] for it in items]
    if (len(ids) != 100 or len(set(ids)) != 100 or set(metadata) != set(ids)
            or len(schedule_order) != len(set(schedule_order))
            or not set(ids) <= set(schedule_order)):
        raise ValueError("calendar requires exactly 100 unique frozen IDs and schedule coverage")
    ordered = [pid for pid in schedule_order if pid in metadata]
    strata = Counter((metadata[pid].get("domain"), metadata[pid].get("sampling_tier")) for pid in ordered)
    expected = {(domain, tier): n for domain in DOMAINS for tier, n in
                (("easy", 6), ("medium", 8), ("hard", 6))}
    if strata != expected:
        raise ValueError("calendar requires frozen per-domain source tiers 6/8/6")
    seen, cohorts = Counter(), {}
    for pid in ordered:
        stratum = (metadata[pid]["domain"], metadata[pid]["sampling_tier"])
        cohorts[pid] = "A" if seen[stratum] % 2 == 0 else "C"
        seen[stratum] += 1
    blocks = {pid: i // 10 for i, pid in enumerate(ordered)}
    return [{"item_id": pid, "condition": "B13_GREEDY" if phase == "B" else "JFINAL",
             "seed": seed, "repetition": repetition, "block": blocks[pid],
             "cohort": cohorts[pid], "residency": residency}
            for residency, phase in enumerate(residency_phases)
            for repetition in range(3) for seed in (17, 29, 43) for pid in ordered
            if phase == "H" or cohorts[pid] == ("A" if residency == 0 else "C")]


def validate_calendar(calendar, items, schedule_order, metadata, policy):
    """Reject altered ordering, coverage, strata, cohorts or approved digest."""
    expected = build_calendar(items, schedule_order, metadata)
    cohorts = {c: list(dict.fromkeys(row["item_id"] for row in expected if row["cohort"] == c))
               for c in ("A", "C")}
    if (calendar != expected or config_hash(calendar) != policy.get("calendar_sha256")
            or policy.get("cohort_a") != cohorts["A"] or policy.get("cohort_c") != cohorts["C"]):
        raise ValueError("operational calendar/cohort approval mismatch")


def validate_code_transition(old, new, approval, config_sha, report_sha):
    """Return embedded policy, or None for unchanged code without amendment.

    old/new and policy old_code_sha256/new_code_sha256 are full path->SHA256
    inventories, NOT aggregate hashes. approval is the complete approval JSON.
    """
    policy = approval.get("operational_amendment")
    if policy is None:
        if old != new:
            raise ValueError("pilot code inventory mismatch without operational approval")
        return None
    if not isinstance(policy, dict):
        raise ValueError("invalid operational amendment")
    fixed = {"id": POLICY_ID, "approved": True, "approved_by": "user",
             "config_sha256": config_sha, "pilot_report_sha256": report_sha,
             "old_code_sha256": old, "new_code_sha256": new,
             "max_job_seconds": MAX_NEW_JOB_SECONDS, "max_gpu_seconds": TOTAL_GPU_SECONDS,
             "expected_test_rows": 6000, "expected_latency_rows": 1800,
             "seeds": [17, 29, 43], "repetitions": 3, "load_counts": {"B": 2, "H": 1}}
    if any(policy.get(k) != v for k, v in fixed.items()) or policy.get("approved") is not True:
        raise ValueError("operational approval bindings/limits mismatch")
    for inventory in (old, new):
        if (not isinstance(inventory, dict) or not inventory
                or any(not isinstance(k, str) or not isinstance(v, str)
                       or not re.fullmatch(r"[0-9a-f]{64}", v) for k, v in inventory.items())):
            raise ValueError("unknown code SHA inventory")
    delta = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
    if (delta != ALLOWED_CODE_DELTA or set(new) - set(old) != {"v3/operational.py"}
            or set(old) - set(new)):
        raise ValueError("operational code delta must be exactly the three approved runtime files")
    for name in ("latency_source_plan_sha256", "latency_schedule_sha256", "calendar_sha256"):
        if not isinstance(policy.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", policy[name]):
            raise ValueError(f"missing/invalid operational binding: {name}")
    for name in ("cohort_a", "cohort_c"):
        cohort = policy.get(name)
        if (not isinstance(cohort, list) or len(cohort) != 50
                or any(not isinstance(pid, str) or not pid for pid in cohort) or len(set(cohort)) != 50):
            raise ValueError("operational cohorts must contain 50 unique IDs each")
    if set(policy["cohort_a"]) & set(policy["cohort_c"]):
        raise ValueError("operational cohorts overlap")
    if not isinstance(policy.get("authorization_source"), str) or not policy["authorization_source"].strip():
        raise ValueError("operational approval requires authorization_source")
    approved = dt.datetime.fromisoformat(str(policy.get("approved_utc", "")).replace("Z", "+00:00"))
    if approved.tzinfo is None or approved > dt.datetime.now(dt.timezone.utc):
        raise ValueError("operational approval must be dated before execution")
    return policy
