"""CPU-only operational calendar and exact inventory transition checks."""

from collections import Counter
from copy import deepcopy
from itertools import groupby

import pytest

from jevlab.common import config_hash, read_json, read_jsonl
from jevlab.v3.operational import (
    ALLOWED_CODE_DELTA, DOMAINS, MAX_NEW_JOB_SECONDS, POLICY_ID, TOTAL_GPU_SECONDS,
    build_calendar, residency_phases, validate_calendar, validate_code_transition,
)
from test_latency_v3 import protocol  # Reuse existing CPU-only lifecycle fixture.


@pytest.fixture
def frozen():
    items, metadata = [], {}
    for domain in sorted(DOMAINS):
        for tier, count in (("easy", 6), ("medium", 8), ("hard", 6)):
            for i in range(count):
                pid = f"{domain}-{tier}-{i}"
                items.append({"id": pid, "problem": "1+1", "language": "en"})
                metadata[pid] = {"item_id": pid, "domain": domain, "sampling_tier": tier,
                                 "difficulty": "hard"}
    order = ["other-test-id", *[it["id"] for it in reversed(items)]]
    return items, order, metadata


def approved(calendar):
    old = {"common.py": "a" * 64, "v3/algorithms.py": "b" * 64,
           "v3/runner.py": "c" * 64, "v3/latency_study.py": "d" * 64}
    new = {**old, **{path: "e" * 64 for path in ALLOWED_CODE_DELTA}}
    policy = {"id": POLICY_ID, "approved": True, "approved_by": "user",
              "approved_utc": "2026-01-01T00:00:00+00:00", "config_sha256": "f" * 64,
              "pilot_report_sha256": "1" * 64, "old_code_sha256": old, "new_code_sha256": new,
              "latency_source_plan_sha256": "2" * 64, "latency_schedule_sha256": "3" * 64,
              "calendar_sha256": config_hash(calendar),
              "cohort_a": list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "A")),
              "cohort_c": list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "C")),
              "max_job_seconds": MAX_NEW_JOB_SECONDS, "max_gpu_seconds": TOTAL_GPU_SECONDS,
              "expected_test_rows": 6000, "expected_latency_rows": 1800,
              "seeds": [17, 29, 43], "repetitions": 3, "load_counts": {"B": 2, "H": 1},
              "authorization_source": "explicit user pre-test operational replan"}
    return old, new, {"operational_amendment": policy}


def test_calendar_exact_pairs_strata_order_and_original_blocks(frozen):
    items, order, metadata = frozen
    calendar = build_calendar(*frozen)
    assert build_calendar(list(reversed(items)), order, metadata) == calendar
    assert len(calendar) == 1800
    keys = {(r["item_id"], r["condition"], r["seed"], r["repetition"]) for r in calendar}
    assert len(keys) == 1800
    assert Counter(r["residency"] for r in calendar) == {0: 450, 1: 900, 2: 450}
    phases = ["B" if phase == "B13_GREEDY" else "H" for phase, _ in
              groupby(r["condition"] for r in calendar)]
    assert phases == residency_phases == ["B", "H", "B"]
    assert Counter(phases) == {"B": 2, "H": 1}
    ordered = [pid for pid in order if pid in metadata]
    assert all(r["block"] == ordered.index(r["item_id"]) // 10 for r in calendar)
    first = {pid: next(r["condition"] for r in calendar if r["item_id"] == pid) for pid in ordered}
    assert Counter(first.values()) == {"B13_GREEDY": 50, "JFINAL": 50}
    for cohort in ("A", "C"):
        ids = {r["item_id"] for r in calendar if r["cohort"] == cohort}
        assert len(ids) == 50
        for domain in DOMAINS:
            assert Counter(metadata[pid]["sampling_tier"] for pid in ids
                           if metadata[pid]["domain"] == domain) == {"easy": 3, "medium": 4, "hard": 3}
    pairs = Counter((r["item_id"], r["seed"], r["repetition"]) for r in calendar)
    assert len(pairs) == 900 and set(pairs.values()) == {2}
    _, _, approval = approved(calendar)
    validate_calendar(calendar, *frozen, approval["operational_amendment"])


@pytest.mark.parametrize("damage", ["cohort", "digest", "calendar", "tier", "schedule"])
def test_calendar_tampering_rejected(frozen, damage):
    items, order, metadata = deepcopy(frozen)
    calendar = build_calendar(items, order, metadata)
    _, _, approval = approved(calendar)
    policy = approval["operational_amendment"]
    if damage == "cohort":
        policy["cohort_a"][0], policy["cohort_c"][0] = policy["cohort_c"][0], policy["cohort_a"][0]
    elif damage == "digest":
        policy["calendar_sha256"] = "0" * 64
    elif damage == "calendar":
        calendar[0]["block"] = 99
        policy["calendar_sha256"] = config_hash(calendar)
    elif damage == "tier":
        metadata[items[0]["id"]]["sampling_tier"] = "hard"
    else:
        order[1], order[2] = order[2], order[1]
    with pytest.raises(ValueError):
        validate_calendar(calendar, items, order, metadata, policy)


def test_strict_old_equality_and_exact_approved_transition(frozen):
    old, new, approval = approved(build_calendar(*frozen))
    assert validate_code_transition(old, old, {}, "f" * 64, "1" * 64) is None
    with pytest.raises(ValueError):
        validate_code_transition(old, new, {}, "f" * 64, "1" * 64)
    assert validate_code_transition(old, new, approval, "f" * 64, "1" * 64) == approval["operational_amendment"]


@pytest.mark.parametrize("damage", ["unknown", "extra", "remove", "unapproved", "config", "report",
                                   "old", "new", "budget", "future", "missing_helper"])
def test_transition_rejects_unknown_or_broader_changes(frozen, damage):
    old, new, approval = approved(build_calendar(*frozen))
    policy = approval["operational_amendment"]
    config, report = "f" * 64, "1" * 64
    if damage == "unknown":
        new["v3/runner.py"] = "unknown"
    elif damage == "extra":
        new["v3/algorithms.py"] = "9" * 64
    elif damage == "remove":
        del new["common.py"]
    elif damage == "unapproved":
        policy["approved"] = False
    elif damage == "config":
        config = "9" * 64
    elif damage == "report":
        report = "9" * 64
    elif damage in ("old", "new"):
        policy[f"{damage}_code_sha256"] = {}
    elif damage == "budget":
        policy["max_gpu_seconds"] += 1
    elif damage == "future":
        policy["approved_utc"] = "2100-01-01T00:00:00Z"
    else:
        del new["v3/operational.py"]
    with pytest.raises(ValueError):
        validate_code_transition(old, new, approval, config, report)


def test_approved_runtime_uses_three_residencies_with_existing_lifecycle(protocol):
    Study, state, data = protocol
    study = Study()
    items = study.latency_items()
    order = read_json(data / "schedule.json")["order"]
    metadata = {r["item_id"]: r for r in read_json(data / "latency_plan.json")["items"]}
    calendar = build_calendar(items, order, metadata)
    _, _, approval = approved(calendar)
    study.manifest["authorization"] = approval
    study.run()
    rows = read_jsonl(study.f["predictions"])
    assert len(rows) == 1800 and len({r["resume_key"] for r in rows}) == 1800
    assert state.loads == 3
    assert [roles for action, roles in state.lifecycle if action == "load"] == [("B",), ("G", "J"), ("B",)]
    assert [{k: row[k] for k in calendar[0]} for row in rows] == calendar
    archive_plan = read_json(study.dir / "latency_plan.json")
    assert archive_plan["source_plan_sha256"] == study.dataset_seals["latency_plan.json"]
    assert archive_plan["calendar_sha256"] == config_hash(calendar)
    study.run()  # Completed checkpoints do not reload or rerun any case.
    assert state.loads == 3 and len(read_jsonl(study.f["predictions"])) == 1800
