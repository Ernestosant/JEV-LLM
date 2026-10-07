"""Synthetic offline amendment evidence; no engines, real inputs, or gold."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import uuid

import pytest

from jevlab.v3.operational import ALLOWED_CODE_DELTA, DOMAINS, build_calendar
from test_operational_v3 import approved as calendar_approval
from test_verify_v3 import (
    evidence, full_evidence, snapshot, complete_fixture, write_archive, bind_approval,
    options, verify, latency_snapshot, seal_public_fixture,
)


@pytest.fixture
def calendar_evidence():
    items, metadata = [], {}
    for domain in sorted(DOMAINS):
        for tier, count in (("easy", 6), ("medium", 8), ("hard", 6)):
            for index in range(count):
                pid = f"{domain}-{tier}-{index}"
                items.append({"id": pid, "problem": "Synthetic public problem", "language": "en"})
                metadata[pid] = {"item_id": pid, "problem": items[-1]["problem"], "domain": domain,
                                 "difficulty": "hard", "sampling_tier": tier}
    order = [r["id"] for r in reversed(items)]
    schedule_raw = json.dumps({"order": order}).encode()
    source = {"items": list(metadata.values()), "intrinsic_difficulty_counts": {"easy": 0, "medium": 0, "hard": 100}}
    source_sha = verify.sha256(json.dumps(source).encode())
    calendar = build_calendar(items, order, metadata)
    _, _, approval = calendar_approval(calendar)
    policy = approval["operational_amendment"]
    policy.update(latency_source_plan_sha256=source_sha, latency_schedule_sha256=verify.sha256(schedule_raw))
    operational = {"operational_amendment": policy, "operational_policy_sha256": verify.canonical_hash(policy),
                   "calendar_sha256": verify.canonical_hash(calendar), "calendar": calendar,
                   "residency_phases": ["B", "H", "B"]}
    plan = {"items": source["items"], "quota_axis": verify.QUOTA_AXIS, "source_plan_sha256": source_sha,
            "intrinsic_difficulty_counts": source["intrinsic_difficulty_counts"],
            "blocks": [order[i:i + 10] for i in range(0, 100, 10)], "seeds": [17, 29, 43],
            "repetitions": 3, "conditions": ["JFINAL", "B13_GREEDY"], "generation": 0, "split": "test",
            **operational}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{**descriptor, "case_index": index + 1, "phase_epoch": descriptor["residency"] + 1,
             "execution_id": str(uuid.uuid4()), "gpu_uuid": "synthetic-same-gpu",
             "t_start_utc": (start + timedelta(seconds=index * 5)).isoformat(),
             "t_end_utc": (start + timedelta(seconds=index * 5 + 4)).isoformat()}
            for index, descriptor in enumerate(calendar)]
    data = {"manifest": {"operational_metadata": operational, "config": {"params": {}}},
            "latency_plan": plan, "latency_metrics": deepcopy(rows), "predictions": deepcopy(rows),
            "metrics": deepcopy(rows), "swaps": [{"to": phase, "ok": True} for phase in ("B", "H", "B")]}
    return data, items, source, source_sha, schedule_raw, policy


def test_exact_actual_calendar(calendar_evidence):
    verify.check_operational_calendar(*calendar_evidence)


@pytest.mark.parametrize("fault", [None, "omitted", "inside_case", "identity", "outside", "private_epoch"])
def test_phase_warmup_uses_runtime_perf_clocks(calendar_evidence, fault):
    data = calendar_evidence[0]
    data["manifest"]["config"]["params"]["WARMUP_N"] = 1
    data["warmup"] = []
    for epoch, swap in enumerate(data["swaps"], 1):
        swap.update(phase_id=str(uuid.uuid4()), phase_epoch=epoch, gpu_uuid="synthetic-same-gpu",
                    resident_roles=["B"] if epoch != 2 else ["G", "J"], cleanup_confirmed=True,
                    t_start_perf=epoch * 10., t_end_perf=epoch * 10. + 5.)
        data["warmup"].append({**swap, "outside_T_total": True, "generation": 900,
                               "t_start_perf": epoch * 10. + 1., "t_end_perf": epoch * 10. + 2.})
    if fault == "omitted":
        data["warmup"].pop()
    elif fault == "inside_case":
        data["warmup"][0]["outside_T_total"] = False
    elif fault == "identity":
        data["warmup"][0]["phase_id"] = str(uuid.uuid4())
    elif fault == "outside":
        data["warmup"][0]["t_end_perf"] = 99.
    elif fault == "private_epoch":
        data["warmup"][0]["phase_epoch"] = 4
    if fault is None:
        verify.check_operational_calendar(*calendar_evidence)
    else:
        with pytest.raises(ValueError):
            verify.check_operational_calendar(*calendar_evidence)


@pytest.mark.parametrize("fault", ["cohort", "calendar_sha", "policy_sha", "source_sha", "schedule_sha",
    "calendar", "order", "block", "case_index", "epoch", "residency", "companion", "clock", "phase_count",
    "phase_order", "failed_phase", "metadata", "items", "blocks", "private", "null", "reseed", "repeat", "omit"])
def test_calendar_tampering_fails(calendar_evidence, fault):
    data, items, source, source_sha, schedule_raw, policy = calendar_evidence
    plan = data["latency_plan"]
    rows = data["latency_metrics"]
    if fault == "cohort":
        policy["cohort_a"][0], policy["cohort_c"][0] = policy["cohort_c"][0], policy["cohort_a"][0]
    elif fault in ("calendar_sha", "policy_sha", "source_sha"):
        plan[{"calendar_sha": "calendar_sha256", "policy_sha": "operational_policy_sha256",
              "source_sha": "source_plan_sha256"}[fault]] = "0" * 64
    elif fault == "schedule_sha":
        policy["latency_schedule_sha256"] = "0" * 64
    elif fault == "calendar":
        plan["calendar"] = list(reversed(plan["calendar"]))
    elif fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault in ("block", "case_index", "epoch", "residency", "reseed", "repeat"):
        rows[0][{"epoch": "phase_epoch", "reseed": "seed", "repeat": "repetition"}.get(fault, fault)] = 99
    elif fault == "companion":
        data["metrics"][0]["cohort"] = "C"
    elif fault == "clock":
        rows[1]["t_start_utc"] = rows[0]["t_start_utc"]
    elif fault == "phase_count":
        data["swaps"].append({"to": "H", "ok": True})
    elif fault == "phase_order":
        data["swaps"][0]["to"] = "H"
    elif fault == "failed_phase":
        data["swaps"][0]["ok"] = False
    elif fault == "metadata":
        data["manifest"]["operational_metadata"] = {}
    elif fault == "items":
        plan["items"] = deepcopy(plan["items"])
        plan["items"][0]["difficulty"] = "easy"
    elif fault == "blocks":
        plan["blocks"] = list(reversed(plan["blocks"]))
    elif fault == "private":
        plan["private_label"] = "forbidden"
    elif fault == "null":
        plan["calendar"] = None
    else:
        rows.pop()
    with pytest.raises(ValueError):
        verify.check_operational_calendar(data, items, source, source_sha, schedule_raw, policy)


@pytest.fixture
def transition(calendar_evidence):
    package = verify.ROOT / "src" / "jevlab"
    new = {p.relative_to(package).as_posix(): verify.sha256(p.read_bytes())
           for p in sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
    old = {k: "a" * 64 if k in ALLOWED_CODE_DELTA else v for k, v in new.items() if k != "v3/operational.py"}
    policy = deepcopy(calendar_evidence[-1])
    policy.update(old_code_sha256=old, new_code_sha256=new)
    approval = {"operational_amendment": policy}
    manifest = {"created_utc": "2026-01-02T00:00:00Z", "authorization": deepcopy(approval)}
    return old, new, approval, manifest, policy["config_sha256"], policy["pilot_report_sha256"]


def test_exact_transition_and_unchanged_default(transition):
    assert verify.check_operational_transition(*transition) == transition[2]["operational_amendment"]
    old = transition[0]
    assert verify.check_operational_transition(old, old, {}, {}, "f" * 64, "1" * 64) is None
    with pytest.raises(ValueError, match="without operational approval"):
        verify.check_operational_transition(old, transition[1], {}, {}, "f" * 64, "1" * 64)


def test_top_level_checker_binding_preserves_runtime_policy(transition):
    old, new, approval, manifest, config, report = transition
    policy = deepcopy(approval["operational_amendment"])
    approval["new_verifier_sha256"] = verify.sha256(verify.safe_read(verify.__file__))
    assert verify.check_operational_transition(*transition) == policy
    assert approval["operational_amendment"] == policy
    assert {name for name in set(old) | set(new) if old.get(name) != new.get(name)} == ALLOWED_CODE_DELTA
    approval["operational_amendment"]["new_verifier_sha256"] = approval["new_verifier_sha256"]
    with pytest.raises(ValueError, match="policy fields"):
        verify.check_operational_transition(*transition)


@pytest.mark.parametrize("digest", [None, "", "0" * 64, "A" * 64, 123, {}])
def test_top_level_checker_binding_rejects_invalid_sha(transition, digest):
    transition[2]["new_verifier_sha256"] = digest
    with pytest.raises(ValueError, match="current verifier SHA mismatch"):
        verify.check_operational_transition(*transition)


def test_checker_binding_uses_executing_file_not_root(transition, tmp_path, monkeypatch):
    transition[2]["new_verifier_sha256"] = verify.sha256(verify.safe_read(verify.__file__))
    checker = tmp_path / "verify_run_v3.py"
    checker.write_bytes(b"changed checker")
    monkeypatch.setattr(verify, "__file__", str(checker))
    with pytest.raises(ValueError, match="current verifier SHA mismatch"):
        verify.check_operational_transition(*transition)


@pytest.mark.parametrize("fault", ["old", "new", "undeclared", "actual_source", "null", "private", "manifest",
                                  "unapproved", "after_construction", "nonutc", "config", "report", "float"])
def test_transition_fail_closed(transition, fault):
    old, new, approval, manifest, config, report = deepcopy(transition)
    policy = approval["operational_amendment"]
    if fault in ("old", "new"):
        policy[f"{fault}_code_sha256"]["v3/runner.py"] = "9" * 64
    elif fault == "undeclared":
        new["common.py"] = "9" * 64
    elif fault == "actual_source":
        new["v3/runner.py"] = "9" * 64
        policy["new_code_sha256"] = deepcopy(new)
        manifest["authorization"] = deepcopy(approval)
    elif fault == "null":
        approval["operational_amendment"] = None
    elif fault == "private":
        policy["private_label"] = "forbidden"
    elif fault == "manifest":
        manifest["authorization"] = {}
    elif fault == "unapproved":
        policy["approved"] = False
    elif fault == "after_construction":
        policy["approved_utc"] = "2026-01-03T00:00:00Z"
    elif fault == "nonutc":
        policy["approved_utc"] = "2026-01-01T00:00:00+01:00"
    elif fault == "config":
        config = "9" * 64
    elif fault == "report":
        report = "9" * 64
    else:
        policy["max_gpu_seconds"] = 90000.0
    with pytest.raises(ValueError):
        verify.check_operational_transition(old, new, approval, manifest, config, report)


def test_approval_reaudits_actual_old_pilots_without_rewriting_receipts(evidence, transition):
    full = full_evidence(evidence)
    root, inputs, config, prompts, _ = full
    old, new, approval, _, _, _ = transition
    report_path, sentinel = root / "pilot_go.json", root / "budget_approval.json"
    source_path = root / "latency_plan.json"
    source = json.loads(source_path.read_bytes())
    for row in source["items"]:
        row["domain"] = sorted(DOMAINS)[int(row["domain"].removeprefix("domain"))]
    source_path.write_text(json.dumps(source))
    schedule_path = root / "schedule.json"
    order = [r["id"] for r in verify.jsonl(inputs.read_text())]
    schedule_path.write_text(json.dumps({"order": order}))
    seal_public_fixture(root, (inputs, root / "pilot_inputs.jsonl", root / "dataset_manifest.json",
                              root / "REVIEW_CONTRACT.md", root / "latency_inputs.jsonl", source_path, schedule_path))
    subset = verify.jsonl((root / "latency_inputs.jsonl").read_text())
    calendar = build_calendar(subset, order, {r["item_id"]: r for r in source["items"]})
    report = json.loads(report_path.read_bytes())
    pilot_evidence = (root, root / "pilot_inputs.jsonl", *full[2:])
    for arm, entry in report["conditions"].items():
        data = snapshot(pilot_evidence, arm)
        data["manifest"]["config"]["code_sha256"] = old
        data["manifest"]["config"]["data_sha256"]["schedule.json"] = verify.sha256(schedule_path.read_bytes())
        complete_fixture(data, pilot_evidence)
        data["manifest"]["run_name"] = f"{arm}_mock_{data['manifest']['config_hash'][:8]}"
        archive = write_archive(root / "old_pilot" / arm, data)
        entry["verification"].update(archive=str(archive), archive_sha256=verify.sha256(archive.read_bytes()),
            notebook_sha256=verify.sha256(next(archive.parent.glob("*.ipynb")).read_bytes()),
            config_hash=data["manifest"]["config_hash"])
    report_path.write_text(json.dumps(report))
    budget = json.loads(sentinel.read_bytes())
    policy = approval["operational_amendment"]
    policy.update(config_sha256=verify.sha256(config.read_bytes()), pilot_report_sha256=verify.sha256(report_path.read_bytes()),
                  approved_utc="2025-12-31T00:00:00Z", calendar_sha256=verify.canonical_hash(calendar),
                  latency_source_plan_sha256=verify.sha256(source_path.read_bytes()),
                  latency_schedule_sha256=verify.sha256(schedule_path.read_bytes()),
                  cohort_a=list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "A")),
                  cohort_c=list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "C")))
    budget.update(pilot_report_sha256=policy["pilot_report_sha256"], operational_amendment=policy,
                  new_verifier_sha256=verify.sha256(verify.safe_read(verify.__file__)))
    sentinel.write_text(json.dumps(budget))
    execution = snapshot(full, "B13_GREEDY", "test")
    execution["manifest"]["config"]["code_sha256"] = new
    execution["manifest"]["config"]["data_sha256"]["schedule.json"] = verify.sha256(schedule_path.read_bytes())
    complete_fixture(execution, full)
    bind_approval(execution, sentinel)
    execution["manifest"]["authorization"]["operational_amendment"] = deepcopy(policy)
    before = {arm: verify.safe_read(entry["verification"]["archive"]) for arm, entry in report["conditions"].items()}
    result = verify.verify_records(execution, **options(full, "B13_GREEDY", "test"), approval=sentinel)
    assert result["ok"] and result["records"] == 500
    assert result["authorization"]["operational_amendment"] == policy
    latency = latency_snapshot(full)
    latency["manifest"]["config"]["code_sha256"] = new
    latency["manifest"]["config"]["data_sha256"]["schedule.json"] = verify.sha256(schedule_path.read_bytes())
    indexed = {(r["item_id"], r["condition"], r["seed"], r["repetition"]): r for r in latency["latency_metrics"]}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    measured = []
    for index, descriptor in enumerate(calendar):
        row = indexed[tuple(descriptor[k] for k in ("item_id", "condition", "seed", "repetition"))]
        row.update(descriptor, case_index=index + 1,
                   domain=next(r["domain"] for r in source["items"] if r["item_id"] == row["item_id"]),
                   t_start_perf=index * 5., t_end_perf=index * 5. + 4.,
                   t_start_utc=(start + timedelta(seconds=index * 5)).isoformat(),
                   t_end_utc=(start + timedelta(seconds=index * 5 + 4)).isoformat())
        measured.append(row)
    latency.update(latency_metrics=measured, swaps=[])
    digest = verify.canonical_hash(latency["manifest"]["config"])
    for row in measured:
        row["resume_key"] = verify.sha256(f"{digest}|{row['item_id']}|{row['condition']}|{row['seed']}|{row['repetition']}".encode())
    complete_fixture(latency, full)
    operational = {"operational_amendment": policy, "operational_policy_sha256": verify.canonical_hash(policy),
                   "calendar_sha256": verify.canonical_hash(calendar), "calendar": calendar,
                   "residency_phases": ["B", "H", "B"]}
    latency["manifest"]["authorization"]["operational_amendment"] = deepcopy(policy)
    latency["manifest"]["operational_metadata"] = deepcopy(operational)
    latency["latency_plan"] = {"items": source["items"], "quota_axis": verify.QUOTA_AXIS,
        "source_plan_sha256": policy["latency_source_plan_sha256"],
        "intrinsic_difficulty_counts": source["intrinsic_difficulty_counts"],
        "blocks": [order[i:i + 10] for i in range(0, 100, 10)], "seeds": [17, 29, 43],
        "repetitions": 3, "conditions": ["JFINAL", "B13_GREEDY"], "generation": 0, "split": "test", **operational}
    result = verify.verify_records(latency, **options(full, "LATENCY", "test"), approval=sentinel)
    assert result["ok"] and result["records"] == 1800
    assert [r["to"] for r in latency["swaps"]] == ["B", "H", "B"]
    for fault in ("cohort", "calendar"):
        damaged = deepcopy(budget)
        amendment = damaged["operational_amendment"]
        if fault == "cohort":
            amendment["cohort_a"][0], amendment["cohort_c"][0] = amendment["cohort_c"][0], amendment["cohort_a"][0]
        else:
            amendment["calendar_sha256"] = "0" * 64
        sentinel.write_text(json.dumps(damaged))
        bind_approval(execution, sentinel)
        execution["manifest"]["authorization"]["operational_amendment"] = deepcopy(amendment)
        with pytest.raises(ValueError, match="calendar/cohort"):
            verify.verify_records(execution, **options(full, "B13_GREEDY", "test"), approval=sentinel)
    assert before == {arm: verify.safe_read(entry["verification"]["archive"]) for arm, entry in report["conditions"].items()}
    budget.pop("operational_amendment")
    sentinel.write_text(json.dumps(budget))
    bind_approval(execution, sentinel)
    with pytest.raises(ValueError, match="without operational approval"):
        verify.verify_records(execution, **options(full, "B13_GREEDY", "test"), approval=sentinel)
