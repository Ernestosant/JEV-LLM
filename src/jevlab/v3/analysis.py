"""CPU-only v3 analysis; gold is read only by an explicit offline analysis call.

Clusters are problems, not seeds or repetitions. The two co-primary percentile
intervals are approximate 97.5% Bonferroni intervals; 95% intervals are descriptive.
No legacy-run discovery, permissive answer repair, or inference is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import read_json, read_jsonl, sha256_file, utc_now, write_json
from ..evaluate import grade_prediction
from ..parsing import extract_final

SEEDS = (17, 29, 43)
ORDER = ("G_SINGLE", "JFINAL", "B13", "G_GREEDY", "B13_GREEDY", "Q9_GREEDY")
SAMPLED = frozenset(ORDER[:3])
SECONDARY = (("JFINAL", "G_SINGLE"), ("JFINAL", "VOTE4SHARED"),
             ("G_GREEDY", "Q9_GREEDY"), ("JFINAL", "Q9_GREEDY"), ("G_SINGLE", "B13"))
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 271828
GRADE_FIELDS = ("format_status", "answer_raw", "answer_normalized", "valid_format", "correct")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cluster_bootstrap(values, domains, n_boot=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Resample paired problem effects within domain, never individual seed/rep rows."""
    values, domains = np.asarray(values, dtype=float), np.asarray(domains)
    require(values.ndim == 1 and len(values) == len(domains) > 0 and np.isfinite(values).all(),
            "Finite complete problem effects/domains required")
    require(isinstance(n_boot, int) and n_boot > 0, "Positive bootstrap resample count required")
    strata = [np.flatnonzero(domains == domain) for domain in sorted(set(domains))]
    rng = np.random.default_rng(seed)
    # Vectorize within strata without materializing B x N x arms x seeds arrays.
    totals = np.zeros(n_boot)
    for indices in strata:
        sampled = rng.choice(indices, size=(n_boot, len(indices)), replace=True)
        totals += values[sampled].sum(axis=1)
    return totals / len(values)


def effect_summary(values, domains, n_boot=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, *, exponential=False):
    values = np.asarray(values, dtype=float)
    samples = cluster_bootstrap(values, domains, n_boot, seed)
    transform = np.exp if exponential else lambda x: x
    samples = transform(samples)
    return {"estimate": float(transform(values.mean())),
            "ci95_descriptive": np.percentile(samples, [2.5, 97.5]).tolist(),
            "ci97_5_bonferroni": np.percentile(samples, [1.25, 98.75]).tolist(),
            "n_problems": len(values), "n_clusters": len(values), "n_boot": n_boot, "seed": seed,
            "strata": dict(sorted(Counter(map(str, domains)).items())),
            "bootstrap_method": "approximate paired domain-stratified problem-cluster percentile"}


def per_problem_accuracy(df, condition):
    data = df.loc[df.condition == condition]
    require(not data.empty, "Missing condition: " + condition)
    expected_seeds = SEEDS if condition in SAMPLED or condition in {"VOTE4SHARED", "FIXED_BRANCH0_SHARED"} else (17,)
    require(not data.duplicated(["problem_id", "seed"]).any(), "Duplicate accuracy case")
    require(data.correct.isin([False, True, 0, 1]).all(), "Accuracy inputs must be binary case grades")
    require(all(isinstance(d, str) and d.strip() for d in data.domain), "Every problem requires a nonempty domain")
    for _, group in data.groupby("problem_id"):
        require(set(group.seed) == set(expected_seeds) and len(group) == len(expected_seeds),
                "Incomplete condition seeds: " + condition)
        require(group.domain.nunique() == 1, "Inconsistent problem domain")
    return data.groupby("problem_id", sort=True).agg(acc=("correct", "mean"), domain=("domain", "first"))


def paired_accuracy_v3(df, method="JFINAL", baseline="B13_GREEDY", n_boot=BOOTSTRAP_RESAMPLES):
    a, b = per_problem_accuracy(df, method), per_problem_accuracy(df, baseline)
    require(a.index.equals(b.index) and a.domain.equals(b.domain), "Complete identical paired IDs/domains required")
    differences = 100 * (a.acc.to_numpy() - b.acc.to_numpy())
    result = effect_summary(differences, a.domain.to_numpy(), n_boot)
    result.update(method=method, baseline=baseline, acc_diff_pp=result.pop("estimate"),
                  method_accuracy_pct=float(100 * a.acc.mean()), baseline_accuracy_pct=float(100 * b.acc.mean()))
    return result


def centered_bootstrap_pvalue(values, domains, n_boot=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Approximate two-sided null tail using bootstrap effects centered at observed mean.

    Global centering retains domain differences under the null of zero overall
    mean. The plus-one correction avoids reporting a finite Monte Carlo tail as zero.
    """
    values = np.asarray(values, dtype=float)
    null = cluster_bootstrap(values - values.mean(), domains, n_boot, seed)
    tail = int(np.count_nonzero(np.abs(null) >= abs(values.mean()) - 1e-12))
    return (tail + 1) / (n_boot + 1)


def holm_adjusted_pvalues(pvalues):
    require(len(pvalues) == 5 and all(math.isfinite(p) and 0 <= p <= 1 for p in pvalues),
            "Holm requires the fixed five secondary p-values")
    order = sorted(range(5), key=pvalues.__getitem__)
    adjusted, previous = [0.0] * 5, 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (5 - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def secondary_comparisons(df, n_boot=BOOTSTRAP_RESAMPLES):
    rows = []
    for method, baseline in SECONDARY:
        result = paired_accuracy_v3(df, method, baseline, n_boot)
        a, b = per_problem_accuracy(df, method), per_problem_accuracy(df, baseline)
        result["p_two_sided_approximate"] = centered_bootstrap_pvalue(
            a.acc.to_numpy() - b.acc.to_numpy(), a.domain.to_numpy(), n_boot)
        result.pop("ci97_5_bonferroni")
        result["p_method"] = "approximate centered paired domain-stratified problem-cluster bootstrap"
        rows.append(result)
    for result, pvalue in zip(rows, holm_adjusted_pvalues([r["p_two_sided_approximate"] for r in rows])):
        result.update(holm_adjusted_p=pvalue, holm_reject_two_sided_05=pvalue <= .05,
                      more_accurate_holm=pvalue <= .05 and result["acc_diff_pp"] > 0)
    return rows


def paired_latency_v3(records, metadata, n_boot=BOOTSTRAP_RESAMPLES):
    """Median nine actual times per arm/problem, then geometric paired TB/TJ."""
    data = pd.DataFrame(records)
    require(not data.empty, "Missing co-primary latency records")
    required = {"problem_id", "condition", "seed", "repetition", "T_total"}
    require(required <= set(data), "Missing latency columns")
    require(set(data.condition) == {"JFINAL", "B13_GREEDY"}, "Latency requires exactly two fixed arms")
    require(not data.duplicated(["problem_id", "condition", "seed", "repetition"]).any(),
            "Duplicate latency case")
    ids = sorted(set(data.problem_id))
    require(set(ids) == set(metadata), "Latency plan/record ID mismatch")
    domains, log_ratios, medians = [], [], []
    expected = {(seed, repetition) for seed in SEEDS for repetition in range(3)}
    for pid in ids:
        times = {}
        for arm in ("JFINAL", "B13_GREEDY"):
            group = data.loc[(data.problem_id == pid) & (data.condition == arm)]
            require(len(group) == 9 and set(zip(group.seed, group.repetition)) == expected,
                    "Latency requires all three seeds and three actual repetitions per arm")
            values = group.T_total.to_numpy(dtype=float)
            require(np.isfinite(values).all() and (values > 0).all(), "Positive finite latency required for every case")
            times[arm] = float(np.median(values))
        domains.append(metadata[pid]["domain"])
        log_ratios.append(math.log(times["B13_GREEDY"] / times["JFINAL"]))
        medians.append({"problem_id": pid, "domain": domains[-1], "sampling_tier": metadata[pid].get("sampling_tier"),
                        "difficulty": metadata[pid].get("difficulty"), "T_JFINAL_median9": times["JFINAL"],
                        "T_B13_GREEDY_median9": times["B13_GREEDY"], "paired_speedup": math.exp(log_ratios[-1])})
    result = effect_summary(log_ratios, domains, n_boot, exponential=True)
    result.update(speedup=result.pop("estimate"), n_records=len(data), per_problem=medians,
                  estimand="exp(mean(log(median9(T_B13_GREEDY) / median9(T_JFINAL))))",
                  status_counts={arm: dict(Counter(data.loc[data.condition == arm, "status"])) for arm in ("JFINAL", "B13_GREEDY")},
                  practical_point_10pct_reduction=False)
    result["practical_point_10pct_reduction"] = result["speedup"] >= 1 / .9
    return result


def independent_grade_v3(pred, gold):
    """Independent strict line/number parser: no canonical parsing/grade helper calls."""
    lines = [line.strip()[6:] for line in (pred.get("public_output") or "").split("\n")
             if line.strip()[:6] == "FINAL:"]
    raw, value = lines[-1] if lines else None, None
    status = "no_final" if not lines else "multiple_final" if len(lines) > 1 else "invalid_format"
    if len(lines) == 1:
        number = raw.translate(str.maketrans({chr(c): "-" for c in (0x2212, 0x2013, 0x2014, 0xfe63, 0xff0d)})).strip()
        integer = r"[+-]?\d+"
        if re.fullmatch(integer + r"|[+-]?(?:\d+\.\d*|\.\d+)|" + integer + r"\s*/\s*" + integer, number):
            try:
                value = Fraction(*map(int, number.split("/"))) if "/" in number else Fraction(number)
            except (ValueError, ZeroDivisionError):
                pass
        if value is not None:
            status = "ok"
    return {"format_status": status, "answer_raw": raw,
            "answer_normalized": (str(value.numerator) if value.denominator == 1 else
                                  f"{value.numerator}/{value.denominator}") if value is not None else None,
            "valid_format": status == "ok", "correct": bool(pred.get("status") == "final" and value is not None
                and value == Fraction(gold["gold_numerator"], gold["gold_denominator"]))}


def audit_predictions(predictions, gold):
    rows, grades = [], []
    for pred in predictions:
        canonical = grade_prediction(pred, gold[pred["problem_id"]])
        independent = independent_grade_v3(pred, gold[pred["problem_id"]])
        disagreements = [field for field in GRADE_FIELDS if canonical[field] != independent[field]]
        rows.append({"problem_id": pred["problem_id"], "seed": pred["seed"], "condition": pred["condition"],
                     "disagreements": disagreements, "canonical": {f: canonical[f] for f in GRADE_FIELDS},
                     "independent": independent})
        grades.append(canonical)
    return {"n_predictions": len(rows), "n_disagreements": sum(bool(r["disagreements"]) for r in rows),
            "fields_checked": list(GRADE_FIELDS), "rows": rows}, grades


def shared_control_choices(proposal, candidates):
    """Gold-free fixed branch and eligible plurality from the SAME JFINAL pool."""
    require(proposal.get("candidate_set_id"), "Missing candidate_set_id")
    require(all(c.get("candidate_set_id") == proposal["candidate_set_id"] for c in candidates),
            "Cannot mix shared candidate sets")
    branches = {c["branch"]: c for c in candidates}
    require(len(branches) == len(candidates) and set(branches) <= {0, 1, 2, 3}, "Duplicate/invalid candidate branches")
    complete = proposal["status"] == "complete"
    require(not complete or set(branches) == {0, 1, 2, 3}, "Complete proposal needs all four branches")
    if not complete:
        require(proposal["status"] in {"partial", "no_candidates"}, "Unknown candidate-set status")
        require((not candidates) == (proposal["status"] == "no_candidates"), "No-candidate status must match the persisted empty pool")
        return {"FIXED_BRANCH0_SHARED": None, "VOTE4SHARED": None}
    votes = defaultdict(list)
    for branch, candidate in sorted(branches.items()):
        status, _, value = extract_final(candidate["public_output"])
        if candidate["status"] == "final" and status == "ok":
            votes[value].append(branch)
    winner = min(votes.values(), key=lambda group: (-len(group), min(group)))[0] if votes else 0
    return {"FIXED_BRANCH0_SHARED": branches[0], "VOTE4SHARED": branches[winner]}


def shared_controls(predictions, proposals, candidates, gold):
    """Failures retain four planned slots and all controls count them incorrect."""
    predictions = [p for p in predictions if p["condition"] == "JFINAL"]
    proposal_by_id = {p["candidate_set_id"]: p for p in proposals}
    require(len(proposal_by_id) == len(proposals), "Duplicate candidate set")
    require(len(predictions) == len(proposals), "Every JFINAL case must persist a proposal, including failures")
    require(len({p["candidate_set_id"] for p in predictions}) == len(predictions),
            "Every JFINAL case requires a unique candidate set")
    by_set = defaultdict(list)
    for candidate in candidates:
        by_set[candidate["candidate_set_id"]].append(candidate)
    require(set(by_set) <= set(proposal_by_id), "Orphan candidate sets")
    require({p["candidate_set_id"] for p in predictions} == set(proposal_by_id), "Prediction/proposal linkage mismatch")
    control_grades, diagnostics = [], []
    for pred in predictions:
        proposal = proposal_by_id[pred["candidate_set_id"]]
        require(all(proposal.get(k) == pred.get(k) for k in ("problem_id", "seed", "condition")),
                "Candidate-set case identity mismatch")
        pool = by_set[proposal["candidate_set_id"]]
        require(all(c.get("problem_id") == pred["problem_id"] and c.get("seed") == pred["seed"] and
                    c.get("condition") == "JFINAL" for c in pool), "Candidate case identity mismatch")
        choices = shared_control_choices(proposal, pool)
        for condition, choice in choices.items():
            control = {**pred, "condition": condition, "public_output": choice["public_output"] if choice else "",
                       "status": (choice["status"] if choice else proposal["status"]) if pred["status"] == "final" else pred["status"]}
            control_grades.append({**grade_prediction(control, gold[pred["problem_id"]]),
                                   "candidate_set_id": proposal["candidate_set_id"],
                                   "chosen_branch": choice["branch"] if choice else None})
        complete = proposal["status"] == "complete"
        correct = [int(grade_prediction({**pred, **c}, gold[pred["problem_id"]])["correct"]) for c in pool] if complete else []
        diagnostics.append({"problem_id": pred["problem_id"], "seed": pred["seed"],
                            "candidate_set_id": proposal["candidate_set_id"], "pool_status": proposal["status"],
                            "n_logged_candidates": len(pool), "planned_candidate_slots": 4,
                            "uniform_expected": sum(correct) / 4, "oracle_at4": max(correct, default=0),
                            "chosen_correct": int(grade_prediction(pred, gold[pred["problem_id"]])["correct"])})
    available = [r for r in diagnostics if r["oracle_at4"]]
    summary = {"n_cases": len(diagnostics), "planned_candidate_slots": 4 * len(diagnostics),
               "logged_candidate_slots": len(candidates), "pool_status_counts": dict(Counter(r["pool_status"] for r in diagnostics)),
               "uniform_expected_accuracy_pct_diagnostic": 100 * np.mean([r["uniform_expected"] for r in diagnostics]),
               "oracle_at4_pct_diagnostic": 100 * np.mean([r["oracle_at4"] for r in diagnostics]),
               "correct_choice_given_any_correct_pct": 100 * np.mean([r["chosen_correct"] for r in available]) if available else None,
               "scope": "Gold-free shared selection; uniform/oracle use gold only offline as diagnostics. No control latency.",
               "rows": diagnostics}
    return control_grades, summary


def co_primary_conclusions(quality, latency):
    quality_ok = quality["ci97_5_bonferroni"][0] > 0
    latency_ok = latency["ci97_5_bonferroni"][0] > 1
    return {"quality_superiority": quality_ok, "latency_superiority": latency_ok,
            "joint_superiority": quality_ok and latency_ok,
            "quality_practical_point_5pp": quality["acc_diff_pp"] >= 5,
            "latency_practical_point_10pct_reduction": latency["speedup"] >= 1 / .9,
            "rule": "Quality lower97.5 > 0; latency lower97.5 > 1; joint requires both. Practical thresholds are point estimates."}


def summary_table(df, metrics, n_boot=BOOTSTRAP_RESAMPLES):
    rows = []
    metric_by_case = {(m["problem_id"], m["condition"], m["seed"]): m for m in metrics}
    require(len(metric_by_case) == len(metrics), "Duplicate descriptive metric case")
    for condition in (*ORDER, "FIXED_BRANCH0_SHARED", "VOTE4SHARED"):
        cases = df.loc[df.condition == condition]
        problems = per_problem_accuracy(df, condition)
        ci = np.percentile(cluster_bootstrap(100 * problems.acc.to_numpy(), problems.domain.to_numpy(), n_boot), [2.5, 97.5]).tolist()
        times = [metric_by_case[(r.problem_id, condition, r.seed)]["T_total"] for r in cases.itertuples()] if condition in ORDER else []
        rows.append({"condition": condition, "n_cases": len(cases), "n_problems": len(problems),
                     "accuracy_pct": float(100 * problems.acc.mean()), "accuracy_ci95_descriptive": ci,
                     "status_counts": dict(Counter(cases.status)), "incorrect_cases": int((~cases.correct.astype(bool)).sum()),
                     "latency_p50_s_descriptive": float(np.median(times)) if times else None,
                     "latency_p95_s_descriptive": float(np.percentile(times, 95)) if times else None,
                     "latency_scope": "different runs; descriptive only" if times else "not measured; shared offline control"})
    return rows


def _integrity_module():
    # Tools stay standalone; import this v3-only validator without importing runtime engines.
    path = Path(__file__).resolve().parents[3] / "tools" / "verify_run_v3.py"
    spec = importlib.util.spec_from_file_location("jevlab_v3_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runs(results_root, *, inputs, config, run_tag, prompts, approval, pilot_report=None):
    """Only unique verified final ZIPs and completed notebooks, never loose checkpoints."""
    verifier = _integrity_module()
    root = Path(results_root)
    require(root.is_dir(), "Results directory unavailable")
    snapshots, provenance = {}, {}
    archives = sorted(root.rglob("*_final.zip"))
    require(len(archives) == 7, "v3 requires exactly six physical finals and one LATENCY final; no mixing/retries")
    common = []
    for path in archives:
        snapshot = verifier.read_archive(path)
        condition = snapshot["manifest"]["condition"]
        require(condition in (*ORDER, "LATENCY") and condition not in snapshots, "Duplicate/historical/unexpected run")
        verification = verifier.verify_run(path.parent, conditions=[condition], inputs=inputs, config=config,
                                           expected_count=500, split="test", run_tag=run_tag,
                                           prompts=prompts, approval=approval, pilot_report=pilot_report)
        require(verification["ok"], "Execution integrity failed: " + str(verification["runs"][condition].get("error")))
        evidence = verification["runs"][condition]
        require(evidence["archive_sha256"] == snapshot["archive_sha256"], "Archive changed while verifying")
        cfg = snapshot["manifest"]["config"]
        common.append({k: cfg[k] for k in ("models", "profile", "jevk5_runtime", "protocol_version", "code_version",
                                          "experiment_config_sha256", "prompt_sha256", "code_sha256", "data_sha256")})
        snapshots[condition] = snapshot
        provenance[condition] = {**evidence, "archive": str(path)}
    require(set(snapshots) == set(ORDER) | {"LATENCY"}, "Incomplete v3 matrix")
    require(all(signature == common[0] for signature in common), "Shared model/config/prompt provenance differs")
    return {"snapshots": snapshots, "provenance": provenance}


def validate_gold_v3(gold_path, inputs, config):
    """Validate the sealed test only; this is deliberately not used by integrity/pilot."""
    gold_path, inputs = Path(gold_path), Path(inputs)
    require(gold_path.name == "test_gold.jsonl" and inputs.name == "test_inputs.jsonl"
            and gold_path.parent.resolve() == inputs.parent.resolve(), "Only sibling frozen v3 test inputs/gold accepted")
    frozen = read_json(config)
    require(frozen.get("protocol_version") == "3" and frozen.get("code_version") == "0.3.0", "Requires frozen v3 code0.3.0")
    seals = {}
    for line in gold_path.with_name("SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            require(name not in seals, "Duplicate dataset seal")
            seals[name] = digest
    manifest_path = gold_path.with_name("dataset_manifest.json")
    for path in (gold_path, inputs, manifest_path):
        require(seals.get(path.name) == sha256_file(path), "Dataset seal mismatch: " + path.name)
    metadata = read_json(manifest_path)
    verifier = _integrity_module()
    public_provenance = verifier.check_public_provenance(metadata, seals, inputs.parent)
    require(metadata.get("dataset_version") == "v3" and metadata.get("sealed") is True,
            "Requires sealed v3 dataset, not historical compatibility")
    require(metadata.get("actual_n", {}).get("test") == 500 and metadata.get("actual_n", {}).get("pilot") == 50,
            "Requires exactly500test+50pilot")
    review = metadata.get("review", {})
    require(review.get("agent_reviewed_all") is True and review.get("human_reviewed") is False,
            "Requires agent review completed, human review false")
    require(isinstance(review.get("policy_id"), str) and review["policy_id"].strip()
            and re.fullmatch(r"[0-9a-f]{64}", review.get("policy_sha256", "")), "Missing sealed review policy identity")
    policy_path = gold_path.with_name("REVIEW_CONTRACT.md")
    require(seals.get(policy_path.name) == sha256_file(policy_path) == review["policy_sha256"],
            "Review policy seal/hash mismatch")
    bound_policy = frozen.get("dataset_review_policy")
    if bound_policy:
        require(bound_policy.get("id") == review["policy_id"] and bound_policy.get("sha256") == review["policy_sha256"],
                "Frozen review policy mismatch")
    gold_rows, input_rows = read_jsonl(gold_path), read_jsonl(inputs)
    gold = {g["id"]: g for g in gold_rows}
    ids = [r["id"] for r in input_rows]
    require(len(gold_rows) == len(gold) == len(ids) == len(set(ids)) == 500 and set(gold) == set(ids),
            "Exactly500 unique matching test IDs required")
    # The public commitment binds private final review decisions, not source hints.
    review_path = gold_path.with_name("review_manifest.json")
    require(seals.get(review_path.name) == sha256_file(review_path) == metadata.get("analysis_manifest_sha256"),
            "Final review manifest commitment mismatch")
    final_review = read_json(review_path)
    require(final_review.get("dataset_version") == "v3" and final_review.get("status") == "FROZEN_ANALYSIS_ONLY"
            and final_review.get("quota_axis") == verifier.QUOTA_AXIS
            and all(final_review.get("review", {}).get(k) == review[k]
                    for k in ("policy_id", "policy_sha256", "agent_reviewed_all", "human_reviewed")),
            "Final review policy binding mismatch")
    assignments = final_review.get("selected_candidate_assignments", {}).get("test", [])
    assigned = {r["id"]: r for r in assignments}
    require(len(assignments) == len(assigned) == 500 and set(assigned) == set(gold),
            "Exactly500 final review assignments required")
    require(len({r.get("candidate_id") for r in assignments}) == 500
            and len({r.get("source_id") for r in assignments}) == 500,
            "Unique final candidate/source identities required")
    pool_path = gold_path.parent / "staging" / "candidate_pool.jsonl"
    require(seals.get("staging/candidate_pool.jsonl") == sha256_file(pool_path), "Source candidate pool seal mismatch")
    pool_rows = read_jsonl(pool_path)
    pool = {r["id"]: r for r in pool_rows}
    require(len(pool) == len(pool_rows), "Duplicate source candidate identity")
    pilot_path = inputs.with_name("pilot_inputs.jsonl")
    require(seals.get(pilot_path.name) == sha256_file(pilot_path) == metadata.get("input_sha256", {}).get(pilot_path.name),
            "Pilot input seal/commitment mismatch")
    pilot_inputs = read_jsonl(pilot_path)
    pilot_assignments = final_review.get("selected_candidate_assignments", {}).get("pilot", [])
    pilot_ids = [r["id"] for r in pilot_inputs]
    require(len(pilot_inputs) == len(set(pilot_ids)) == len(pilot_assignments) == 50
            and {r["id"] for r in pilot_assignments} == set(pilot_ids) and not set(pilot_ids) & set(gold),
            "Exactly50 unused pilot IDs and final assignments required")
    all_assignments = assignments + pilot_assignments
    require(len({r.get("candidate_id") for r in all_assignments}) == len({r.get("source_id") for r in all_assignments}) == 550
            and final_review.get("selected_twice_approved_candidates") == 550,
            "Exactly550 distinct actually reviewed source assignments required")
    pilot_rows = []
    for assignment in pilot_assignments:
        candidate = pool.get(assignment.get("candidate_id"), {})
        decision = final_review.get("review_decisions", {}).get(assignment.get("candidate_id"), {})
        pilot_rows.append({**candidate, **assignment, "domain": decision.get("final_domain"),
            "difficulty": decision.get("final_difficulty"), "sampling_tier": candidate.get("provisional_difficulty"),
            "template_group": assignment.get("source_id"), "review_decision": decision,
            "accepted_via_adjudication": decision.get("accepted_via_adjudication"),
            "agent_reviewed": True, "human_reviewed": False, "reviewed": False})
    evidence_hashes = final_review.get("review_evidence_sha256", {})
    require(isinstance(evidence_hashes, dict) and evidence_hashes, "Sealed original review evidence required")
    phase_records, evidence_paths = {}, {}
    for name, expected_hash in evidence_hashes.items():
        relative = name.removeprefix("data/v3/")
        path = gold_path.parent / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.resolve().is_relative_to(gold_path.parent.resolve())
                and seals.get(relative) == sha256_file(path) == expected_hash, "Original review evidence seal mismatch")
        evidence_paths[relative] = expected_hash
        if path.suffix == ".jsonl":
            for record in read_jsonl(path):
                if record.get("phase") in {"blind", "reference", "adjudication"}:
                    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False).encode("utf-8")).hexdigest()
                    phase_records[digest] = record
    input_by_id = {r["id"]: r for r in input_rows + pilot_inputs}
    assigned.update({r["id"]: r for r in pilot_assignments})
    for row in gold_rows + pilot_rows:
        require(row.get("reviewed") is False and row.get("human_reviewed") is False and row.get("agent_reviewed") is True,
                "Test rows require actual agent review true and human review false")
        require(type(row.get("gold_numerator")) is int and type(row.get("gold_denominator")) is int
                and row["gold_denominator"] > 0, "Gold must be an exact integer rational with positive denominator")
        value = Fraction(row["gold_numerator"], row["gold_denominator"])
        normalized = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        require(row.get("gold_answer") == normalized and isinstance(row.get("domain"), str) and row["domain"].strip()
                and row.get("difficulty") in {"easy", "medium", "hard"}, "Invalid strict gold metadata")
        assignment = assigned[row["id"]]
        require(all(row.get(k) == assignment.get(k) and row.get(k) is not None
                    for k in ("candidate_id", "candidate_sha256", "source_id", "agent_reviews")),
                "Gold final source/review assignment mismatch")
        candidate = pool.get(row["candidate_id"], {})
        digest = hashlib.sha256(json.dumps({k: v for k, v in candidate.items()
                                          if k not in {"id", "candidate_sha256", "priority"}},
                                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        require(candidate and candidate.get("candidate_sha256") == row["candidate_sha256"] == digest
                and all(row.get(k) == v for k, v in candidate.items() if k != "id")
                and candidate.get("problem") == input_by_id[row["id"]].get("problem")
                and row.get("template_group") == row["source_id"], "Gold original source candidate binding mismatch")
        require(row.get("sampling_tier") == candidate.get("provisional_difficulty")
                and row["sampling_tier"] in verifier.LEVELS, "Sampling tier must preserve the original source proxy")
        decision = final_review.get("review_decisions", {}).get(row["candidate_id"], {})
        require(decision and row.get("review_decision") == decision and decision.get("accepted") is True
                and decision.get("final_domain") == row["domain"]
                and decision.get("final_difficulty") == row["difficulty"]
                and decision.get("sampling_tier") == row["sampling_tier"]
                and decision.get("reviews") == row["agent_reviews"]
                and type(row.get("accepted_via_adjudication")) is bool
                and decision.get("accepted_via_adjudication") is row["accepted_via_adjudication"],
                "Gold final reviewed labels/decision mismatch")
        reviews = row["agent_reviews"]
        require(isinstance(reviews, list) and len(reviews) >= 2
                and len({r.get("reviewer_id") for r in reviews}) == len(reviews)
                and len({r.get("agent_run_id") for r in reviews}) == len(reviews)
                and all(isinstance(r.get(k), str) and r[k].strip() for r in reviews
                        for k in ("reviewer_id", "agent_run_id"))
                and all(re.fullmatch(r"[0-9a-f]{64}", r.get(k, "")) for r in reviews
                        for k in ("blind_record_sha256", "reference_record_sha256")),
                "Independent original review bindings required")
        for original in reviews:
            blind = phase_records.get(original["blind_record_sha256"], {})
            reference = phase_records.get(original["reference_record_sha256"], {})
            answer = blind.get("answer", {})
            require(all(record.get("phase") == phase and record.get("id") == row["candidate_id"]
                        and record.get("reviewer_id") == original["reviewer_id"]
                        and record.get("candidate_sha256") == row["candidate_sha256"]
                        for phase, record in (("blind", blind), ("reference", reference)))
                    and reference.get("blind_record_sha256") == original["blind_record_sha256"]
                    and set(answer) == {"numerator", "denominator"}
                    and type(answer.get("numerator")) is int and type(answer.get("denominator")) is int
                    and answer["denominator"] > 0 and Fraction(**answer) == value
                    and all(blind.get(k) is True for k in ("statement_ok", "unique_answer", "correctness"))
                    and all(reference.get(k) is True for k in ("gold_matches", "solution_consistent")),
                    "Original source-blind/reference phase binding mismatch")
            verdicts = {"blind": {k: blind.get(k) for k in
                        ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")},
                        "reference": {k: reference.get(k) for k in ("gold_matches", "solution_consistent", "approve")}}
            require(original.get("original_verdicts") == verdicts
                    and all(type(value) is bool for phase in verdicts.values() for value in phase.values()),
                    "Original review verdicts must remain bound to their original phases")
            if blind["unit_explicit"] is False:
                require(isinstance(blind.get("independent_solution"), str)
                        and isinstance(reference.get("reference_comparison"), str)
                        and original.get("original_unit_reason") == blind["independent_solution"] + "\n" + reference["reference_comparison"],
                        "Original unit concern must bind original review text")
            if not row["accepted_via_adjudication"]:
                require(all(value is True for phase in verdicts.values() for value in phase.values())
                        and blind.get("domain") == row["domain"] and blind.get("difficulty") == row["difficulty"],
                        "Ordinary approval cannot override original source-blind vetoes/labels")
        if row["accepted_via_adjudication"]:
            adjudication = decision.get("adjudication") or {}
            record = adjudication.get("record") or {}
            record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False).encode("utf-8")).hexdigest()
            answer = record.get("answer", {})
            require(decision.get("status") == "resolved" and decision.get("adjudication_eligible") is True
                    and adjudication.get("record_sha256") == record_hash
                    and phase_records.get(record_hash) == record
                    and record.get("id") == row["candidate_id"] and record.get("phase") == "adjudication"
                    and record.get("candidate_sha256") == row["candidate_sha256"]
                    and record.get("reviewer_id") == adjudication.get("reviewer_id")
                    and isinstance(adjudication.get("agent_run_id"), str) and adjudication["agent_run_id"].strip()
                    and adjudication.get("reviewer_id") not in {r["reviewer_id"] for r in reviews}
                    and adjudication["agent_run_id"] not in {r["agent_run_id"] for r in reviews}
                    and record.get("final_domain") == row["domain"] and record.get("final_difficulty") == row["difficulty"]
                    and set(answer) == {"numerator", "denominator"}
                    and type(answer.get("numerator")) is int and type(answer.get("denominator")) is int
                    and answer["denominator"] > 0 and Fraction(**answer) == value
                    and record.get("original_review_refs") == [{k: r[k] for k in
                        ("reviewer_id", "blind_record_sha256", "reference_record_sha256")}
                        for r in sorted(reviews, key=lambda r: r["reviewer_id"])]
                    and all(record.get(k) is True for k in ("answer_correct", "gold_matches", "solution_consistent",
                                                          "statement_ok", "unique_answer", "correctness", "approve")),
                     "Final adjudication source/original-review binding mismatch")
            concerns = [{"reviewer_id": r["reviewer_id"], "blind_record_sha256": r["blind_record_sha256"],
                         "unit_explicit": False, "original_unit_reason": r.get("original_unit_reason")}
                        for r in reviews if r["original_verdicts"]["blind"]["unit_explicit"] is False]
            require(record.get("original_unit_concerns") == concerns
                    and record.get("unit_resolution") in {"unchanged", "inherently_dimensionless"}
                    and type(record.get("dimensionless_query")) is bool,
                    "Adjudication must preserve original unit concerns/resolution")
            if concerns:
                query = record.get("query_quote", "")
                requested_unit = re.search(r"how many\s+(?:dollars?|cents?|euros?|percent)\b|"
                    r"\bpercent(?:age)?\s+(?:increase|decrease)|what\s+(?:is\s+the\s+)?percent(?:age)?\b|"
                    r"\bin\s+(?:dollars?|cents?|euros?|miles?|kilometers?|met(?:er|re)s?|centimeters?|"
                    r"feet|inches|seconds?|minutes?|hours?|days?|years?|lit(?:er|re)s?|gallons?|"
                    r"grams?|kilograms?|pounds?|ounces?|degrees?)\b", query, re.I) if isinstance(query, str) else True
                require(record["unit_resolution"] == "inherently_dimensionless"
                        and record["dimensionless_query"] is True
                        and record.get("dimensionless_kind") in {"pure_number", "count", "probability", "ratio", "index", "coefficient"}
                        and candidate.get("answer_unit") in (None, "dimensionless")
                        and isinstance(record.get("query_quote"), str) and len(record["query_quote"].strip()) >= 5
                        and record["query_quote"] in candidate["problem"] and not requested_unit,
                        "False unit verdict requires an original dimensionless query resolution")
        else:
            require(decision.get("status") == "approved" and decision.get("adjudication") is None
                    and all(r.get("approved") is True for r in reviews)
                    and decision.get("decision_type") == ("source_hint_contrast" if
                        (row["domain"], row["difficulty"]) != (row.get("provisional_domain"), row.get("provisional_difficulty"))
                        else "ordinary"), "Ordinary/source-hint-contrast final review binding mismatch")
    profiles = {split: verifier.sampling_profile(rows, split) for split, rows in (("test", gold_rows), ("pilot", pilot_rows))}
    require(all(profiles[s]["intrinsic_difficulty_counts"] == public_provenance["intrinsic_difficulty_counts"][s]
                for s in profiles), "Published intrinsic difficulty counts differ from actual550 reviewed labels")
    for field, axis in (("strata", "source_sampling_tier_counts"), ("intrinsic_reviewed_strata", "intrinsic_difficulty_counts")):
        expected = {s: {d: counts[axis] for d, counts in profile["by_domain"].items()} for s, profile in profiles.items()}
        require(final_review.get(field) == expected, "Final selected " + field + " mismatch")
    dataset_profile = {"quota_axis": verifier.QUOTA_AXIS, "splits": profiles, "n_agent_reviewed": 550,
        "human_reviewed": False, "source_sampling_tier_is_intrinsic_difficulty": False,
        "n_source_candidates": len(pool), "n_unused_source_candidates": len(pool) - 550}
    return gold, {"gold_sha256": sha256_file(gold_path), "inputs_sha256": sha256_file(inputs),
                  "pilot_inputs_sha256": sha256_file(pilot_path), "dataset_profile": dataset_profile,
                  "review_manifest_sha256": sha256_file(review_path), "source_candidate_pool_sha256": sha256_file(pool_path),
                  "review_evidence_sha256": evidence_paths,
                   "dataset_manifest_sha256": sha256_file(manifest_path), "review_policy_sha256": sha256_file(policy_path),
                   "dataset_seal_sha256": sha256_file(gold_path.with_name("SHA256SUMS")), "review": review,
                   "public_source_provenance": public_provenance}


def write_report(report, out):
    primary, declarations = report["primary"], report["declarations"]
    lines = ["# JEV-LLM v3 Offline Analysis", "", "General ~13B greedy is the fixed primary baseline.", "",
              "500 test problems; 6000 physical cases. Seeds are not additional independent problems.", "",
              "## Source Sampling And Intrinsic Difficulty", "",
              "Balance is by original source sampling tier within the reviewed mathematical domain, not by intrinsic difficulty.",
              "Test per domain:30/40/30 source tiers; pilot:3/4/3; latency:6/8/6. Source tier hard does not mean genuinely hard.",
              "550 selected test/pilot problems are agent-reviewed; human_reviewed=false. Unused source candidates do not enter these counts.",
              "The bootstrap remains DOMAIN-stratified; no source-tier or intrinsic-difficulty tuning of co-primary effects.", "",
              "| Split | Problems | Source-Tier Easy | Source-Tier Medium | Source-Tier Hard | Intrinsic Easy | Intrinsic Medium | Intrinsic Hard |",
              "|---|---|---|---|---|---|---|---|"]
    for split, profile in report["meta"]["dataset_profile"]["splits"].items():
        counts = [profile[axis][level] for axis in ("source_sampling_tier_counts", "intrinsic_difficulty_counts")
                  for level in ("easy", "medium", "hard")]
        lines.append(f"| {split} | {profile['n_problems']} | " + " | ".join(map(str, counts)) + " |")
    lines.extend(["", "Per-domain source-tier and intrinsic counts:", "```json",
              json.dumps(report["meta"]["dataset_profile"], indent=2), "```", "",
              "## Co-Primary Effects", "",
             "Approximate paired domain-stratified problem-cluster bootstrap: 10000 draws, seed271828.",
             "97.5% percentile CIs use Bonferroni over two co-primary effects; 95% CIs are descriptive.", "",
             f"Quality JFINAL mean3seeds - B13_GREEDY: {primary['quality']['acc_diff_pp']:.6g} pp; "
             f"CI95 {primary['quality']['ci95_descriptive']}; CI97.5 {primary['quality']['ci97_5_bonferroni']}.",
             f"Latency geometric paired speedup: {primary['latency']['speedup']:.6g}; "
             f"CI95 {primary['latency']['ci95_descriptive']}; CI97.5 {primary['latency']['ci97_5_bonferroni']}.",
             "Latency uses100 test-problem clusters, median9 perarm, all1800 actual records including failures.", "",
             "```json", json.dumps(declarations, indent=2), "```", "", "## Accuracy", "",
             "| Arm | Cases | Problems | Accuracy % | Descriptive CI95 | Incorrect Cases |",
              "|---|---|---|---|---|---|"])
    for row in report["table"]:
        lines.append(f"| {row['condition']} | {row['n_cases']} | {row['n_problems']} | {row['accuracy_pct']:.6g} | "
                     f"{row['accuracy_ci95_descriptive']} | {row['incorrect_cases']} |")
    lines.extend(["", "## Fixed Secondary", "",
                  "Five fixed contrasts; approximate two-sided centered paired stratified cluster bootstrap p-values; Holm5.",
                  "These are not exact tests. No McNemar is applied to mean3seed outcomes.", "",
                  "| Method | Baseline | Difference pp | Descriptive CI95 | Approximate p | Holm p |",
                  "|---|---|---|---|---|---|"])
    for row in report["secondary"]:
        lines.append(f"| {row['method']} | {row['baseline']} | {row['acc_diff_pp']:.6g} | {row['ci95_descriptive']} | "
                     f"{row['p_two_sided_approximate']:.6g} | {row['holm_adjusted_p']:.6g} |")
    lines.extend(["", "## Shared Controls", "",
                  "Fixed branch0 is chosen before permutation. Plurality uses only valid terminal parses, ties lowest original branch.",
                  "Partial/no-candidate sets count incorrect for every control, with no denominator omissions.",
                  "Uniform expectation and oracle@4 are gold-informed offline diagnostics, not realizable runtime policies.",
                  "No latency is assigned to any shared control.", "", "```json",
                  json.dumps({k: v for k, v in report["jfinal"].items() if k != "rows"}, indent=2), "```", "",
                  "## Audit", "", f"Independent strict parser: {report['parser_audit']['n_predictions']} predictions, "
                  f"{report['parser_audit']['n_disagreements']} disagreements.", "",
                  "## Limitations", "",
                  "- Agent-reviewed only; human reviewed=false. No human review is claimed.",
                  "- Adaptive motivation after prior experiments; historical data and pilots excluded from this test.",
                  "- Public benchmark contamination cannot be excluded; domain and difficulty labels are limited proxies.",
                  "- Third-party expanded ~13B baseline and distinct training; parameter matching is not compute matching.",
                  "- Four samples share generator weights; shared controls condition on the JFINAL pool, not independent generation.",
                  "- Different-run latencies are descriptive. Confirmatory latency is conditional on warm same-A10040 execution.",
                  "- Reload/loading/warmup excluded from case clocks and reported separately; no invented actual CU.",
                  "- Bootstrap CIs and p-values are approximate, not exact coverage or equivalence guarantees."])
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def run_analysis(results_root, gold_path, out_dir, run_tag, *, config, prompts=None, approval=None,
                 inputs=None, pilot_report=None, n_boot=BOOTSTRAP_RESAMPLES):
    """Explicit offline entry point; validate all immutable evidence before grading."""
    require(n_boot == BOOTSTRAP_RESAMPLES, "Confirmatory v3 analysis requires exactly10000 bootstrap draws")
    inputs = Path(inputs) if inputs is not None else Path(gold_path).with_name("test_inputs.jsonl")
    prompts = Path(prompts) if prompts is not None else Path(__file__).resolve().parents[3] / "prompts"
    out = Path(out_dir)
    require(out.resolve() not in (Path(results_root).resolve(), inputs.parent.resolve()), "Analysis output must be separate from input evidence")
    out.mkdir(parents=True, exist_ok=True)
    filenames = ("analysis.json", "report.md", "parser_audit.json", "graded_runs.csv", "final_table.csv",
                 "primary_comparison.json", "secondary_comparisons.csv", "shared_controls.csv", "latency_per_problem.csv")
    for filename in filenames:
        (out / filename).unlink(missing_ok=True)
    runs = load_runs(results_root, inputs=inputs, config=config, run_tag=run_tag, prompts=prompts,
                     approval=approval, pilot_report=pilot_report)
    gold, gold_provenance = validate_gold_v3(gold_path, inputs, config)
    snapshots = runs["snapshots"]
    predictions = [p for condition in ORDER for p in snapshots[condition]["predictions"]]
    require(len(predictions) == 6000, "Exactly6000 physical test predictions required")
    for condition in ORDER:
        rows = snapshots[condition]["predictions"]
        expected = {(pid, seed) for pid in gold for seed in (SEEDS if condition in SAMPLED else (17,))}
        require(len(rows) == len(expected) and all(p.get("condition") == condition
                and type(p.get("seed")) is int for p in rows)
                and {(p.get("problem_id"), p.get("seed")) for p in rows} == expected,
                "Physical matrix requires every arm bound to the same500 sealed IDs and planned seeds")
    audit, grades = audit_predictions(predictions, gold)
    audit["immutable_finals"] = runs["provenance"]
    write_json(out / "parser_audit.json", audit)
    require(audit["n_disagreements"] == 0, "Independent parser disagrees; no confirmatory report emitted")

    proposals = snapshots["JFINAL"]["proposals"]
    candidates = []
    for proposal in proposals:
        for candidate in proposal["candidates"]:
            require(candidate.get("candidate_set_id", proposal["candidate_set_id"]) == proposal["candidate_set_id"],
                    "Foreign candidate_set_id in embedded pool")
            # The pool seed is the master seed; candidate seed is generation provenance.
            candidates.append({**candidate, "candidate_set_id": proposal["candidate_set_id"],
                               "generation_seed": candidate.get("seed_branch"), "seed": proposal["seed"],
                               "problem_id": proposal["problem_id"], "condition": "JFINAL"})
    control_grades, jfinal = shared_controls(predictions, proposals, candidates, gold)
    df = pd.DataFrame([*grades, *control_grades])
    df["sampling_tier"] = df.problem_id.map({pid: g["sampling_tier"] for pid, g in gold.items()})
    quality = paired_accuracy_v3(df)
    require(quality["n_problems"] == 500, "Quality requires500 complete paired problem clusters")
    latency_snapshot = snapshots["LATENCY"]
    plan = latency_snapshot["latency_plan"]
    require(len(plan["items"]) == 100 and len({item["item_id"] for item in plan["items"]}) == 100
            and all(item["item_id"] in gold for item in plan["items"]),
            "Latency requires exactly100 unique frozen test IDs")
    metadata = {item["item_id"]: gold[item["item_id"]] for item in plan["items"]}
    require(len(metadata) == 100, "Latency requires100 frozen test-problem clusters")
    require(all(item["domain"] == metadata[item["item_id"]]["domain"] and
                item["difficulty"] == metadata[item["item_id"]]["difficulty"] and
                item.get("sampling_tier") == metadata[item["item_id"]]["sampling_tier"] for item in plan["items"]),
            "Latency plan domain/difficulty/source tier differs from sealed test metadata")
    gold_provenance["dataset_profile"]["splits"]["latency"] = _integrity_module().sampling_profile(plan["items"], "latency")
    unused_test = [row for pid, row in gold.items() if pid not in metadata]
    gold_provenance["dataset_profile"]["unused_subsets"] = {
        "source_candidates_not_selected": gold_provenance["dataset_profile"]["n_unused_source_candidates"],
        "pilot_excluded_from_confirmatory_quality": gold_provenance["dataset_profile"]["splits"]["pilot"]["n_problems"],
        "test_not_in_latency": {"n_problems": len(unused_test),
            "source_sampling_tier_counts": {level: sum(r["sampling_tier"] == level for r in unused_test) for level in ("easy", "medium", "hard")},
            "intrinsic_difficulty_counts": {level: sum(r["difficulty"] == level for r in unused_test) for level in ("easy", "medium", "hard")}}}
    records = [{**r, "problem_id": r["item_id"]} for r in latency_snapshot["latency_metrics"]]
    latency = paired_latency_v3(records, metadata)
    require(latency["n_records"] == 1800, "Latency requires1800 actual observations")
    secondary = secondary_comparisons(df)
    metrics = [m for condition in ORDER for m in snapshots[condition]["metrics"]]
    table = summary_table(df, metrics)
    # Check snapshots again before emitting declarations, not just before reading gold.
    for condition, evidence in runs["provenance"].items():
        require(sha256_file(evidence["archive"]) == evidence["archive_sha256"], "Final archive changed during analysis: " + condition)
    require(sha256_file(gold_path) == gold_provenance["gold_sha256"] and sha256_file(inputs) == gold_provenance["inputs_sha256"],
             "Sealed test evidence changed during analysis")
    require(sha256_file(inputs.with_name("pilot_inputs.jsonl")) == gold_provenance["pilot_inputs_sha256"],
            "Unused pilot inputs changed during analysis")
    require(sha256_file(Path(gold_path).with_name("dataset_manifest.json")) == gold_provenance["dataset_manifest_sha256"]
            and sha256_file(Path(gold_path).with_name("review_manifest.json")) == gold_provenance["review_manifest_sha256"]
            and sha256_file(Path(gold_path).parent / "staging" / "candidate_pool.jsonl") == gold_provenance["source_candidate_pool_sha256"]
            and sha256_file(Path(gold_path).with_name("REVIEW_CONTRACT.md")) == gold_provenance["review_policy_sha256"]
            and sha256_file(Path(gold_path).with_name("SHA256SUMS")) == gold_provenance["dataset_seal_sha256"],
            "Sealed review evidence changed during analysis")
    require(all(sha256_file(Path(gold_path).parent / name) == digest
                 for name, digest in gold_provenance["review_evidence_sha256"].items()),
             "Original review evidence changed during analysis")
    seals = {}
    for line in Path(gold_path).with_name("SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            seals[name.lstrip("*")] = digest
    require(_integrity_module().check_public_provenance(read_json(inputs.with_name("dataset_manifest.json")), seals, inputs.parent)
            == gold_provenance["public_source_provenance"], "Public source/policy evidence changed during analysis")
    paths = {Path(name).stem: str(out / name) for name in filenames}
    report = {"meta": {"protocol_version": "3", "code_version": "0.3.0", "run_tag": run_tag,
                       "generated_utc": utc_now(), "n_test_problems": 500, "n_physical_cases": 6000,
                       "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                       "historical_compatibility": False, **gold_provenance, "immutable_finals": runs["provenance"]},
              "primary": {"quality": quality, "latency": {k: v for k, v in latency.items() if k != "per_problem"}},
              "declarations": co_primary_conclusions(quality, latency), "secondary": secondary, "table": table,
              "jfinal": {k: v for k, v in jfinal.items() if k != "rows"},
              "parser_audit": {k: v for k, v in audit.items() if k != "rows"}, "paths": paths}
    df.to_csv(out / "graded_runs.csv", index=False)
    pd.DataFrame(table).to_csv(out / "final_table.csv", index=False)
    pd.DataFrame(secondary).to_csv(out / "secondary_comparisons.csv", index=False)
    pd.DataFrame(control_grades).to_csv(out / "shared_controls.csv", index=False)
    pd.DataFrame(latency["per_problem"]).to_csv(out / "latency_per_problem.csv", index=False)
    write_json(out / "primary_comparison.json", report["primary"])
    text = write_report(report, out)
    write_json(out / "analysis.json", report)
    return {**report, "report": text, "df": df}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", help="Explicitly authorize reading frozen test gold offline")
    for name in ("results", "gold", "output", "config", "approval"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--pilot-report", type=Path, help="Technical pilot go report bound by the separate approval")
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args(argv)
    if not args.analyze:
        parser.error("--analyze is required; default/help never read gold")
    try:
        result = run_analysis(args.results, args.gold, args.output, args.run_tag, config=args.config,
                              prompts=args.prompts, approval=args.approval, inputs=args.inputs, pilot_report=args.pilot_report)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps({"ok": True, "paths": result["paths"], "declarations": result["declarations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
