"""Generated CPU fixtures only. Never read or change actual dataset review flags."""

import copy
import json
import math
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevlab.evaluate import grade_prediction
from jevlab.v3 import analysis
from test_verify_v3 import approved, complete_fixture, refresh_traces, write_archive


def gold_row(pid="fake"):
    return {"id": pid, "domain": "fake-domain", "difficulty": "medium", "template_group": pid,
            "gold_numerator": 1, "gold_denominator": 2, "gold_answer": "1/2",
            "reviewed": False, "human_reviewed": False, "agent_reviewed": True}


def prediction(pid="fake", seed=17, condition="JFINAL", text="FINAL: 0.5\n", status="final"):
    return {"problem_id": pid, "seed": seed, "condition": condition, "status": status,
            "public_output": text, "candidate_set_id": f"set-{pid}-{seed}"}


def small_accuracy():
    rows = []
    outcomes = {"JFINAL": [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]],
                "G_SINGLE": [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
                "B13": [[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0]],
                "G_GREEDY": [[0], [1], [0], [1]], "B13_GREEDY": [[0], [1], [1], [0]],
                "Q9_GREEDY": [[1], [0], [1], [0]], "VOTE4SHARED": [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]]}
    for condition, problems in outcomes.items():
        for i, seeds in enumerate(problems):
            for seed, correct in zip(analysis.SEEDS, seeds):
                rows.append({"problem_id": f"p{i}", "seed": seed, "condition": condition,
                             "correct": bool(correct), "domain": "a" if i < 2 else "b"})
    return pd.DataFrame(rows)


def independent_draws(values, domains, n_boot, seed=271828):
    # Intentionally do not call analyzer helpers; reproduce the declared draw order.
    rng = np.random.default_rng(seed)
    sums = [0.0] * n_boot
    for domain in sorted(set(domains)):
        indices = [i for i, d in enumerate(domains) if d == domain]
        for boot in range(n_boot):
            drawn = rng.choice(indices, size=len(indices), replace=True)
            sums[boot] += sum(values[int(index)] for index in drawn)
    return [value / len(values) for value in sums]


def test_quality_means_three_seeds_and_paired_problem_bootstrap_computed_expected():
    df = small_accuracy()
    result = analysis.paired_accuracy_v3(df, n_boot=101)
    differences = [float(Fraction(200, 3)), float(Fraction(-200, 3)), 0.0, 0.0]
    samples = independent_draws(differences, ["a", "a", "b", "b"], 101)
    assert result["acc_diff_pp"] == pytest.approx(sum(differences) / 4)
    assert result["ci95_descriptive"] == pytest.approx(np.percentile(samples, [2.5, 97.5]))
    assert result["ci97_5_bonferroni"] == pytest.approx(np.percentile(samples, [1.25, 98.75]))
    assert result["n_problems"] == result["n_clusters"] == 4
    assert result["strata"] == {"a": 2, "b": 2}
    assert not any("mcnemar" in key.lower() or "exact" in key.lower() for key in result)


def test_cluster_bootstrap_keeps_domains_fixed_and_seed_reproducible():
    values = [1., 2., 10., 20.]
    expected = independent_draws(values, ["a", "a", "b", "b"], 23)
    assert analysis.cluster_bootstrap(values, ["a", "a", "b", "b"], 23) == pytest.approx(expected)
    assert np.array_equal(analysis.cluster_bootstrap(values, ["a", "a", "b", "b"], 23),
                          analysis.cluster_bootstrap(values, ["a", "a", "b", "b"], 23))
    assert analysis.cluster_bootstrap([1., 10.], ["a", "b"], 10).tolist() == [5.5] * 10


@pytest.mark.parametrize("fault", ["missing_seed", "extra_seed", "duplicate", "unpaired", "domain", "nonbinary"])
def test_quality_does_not_intersect_or_drop_incomplete_problems(fault):
    df = small_accuracy()
    if fault == "missing_seed":
        df = df.drop(0)
    elif fault == "extra_seed":
        df.loc[0, "seed"] = 99
    elif fault == "duplicate":
        df = pd.concat([df, df.iloc[:1]])
    elif fault == "unpaired":
        df.loc[df.condition == "B13_GREEDY", "problem_id"] += "-other"
    elif fault == "domain":
        df.loc[df.condition == "B13_GREEDY", "domain"] = "other"
    else:
        df["correct"] = df.correct.astype(float)
        df.loc[0, "correct"] = .5
    with pytest.raises(ValueError):
        analysis.paired_accuracy_v3(df, n_boot=10)


def test_centered_approximate_pvalue_computed_null_tail_and_holm_family():
    values, domains = [.5, -.25, 1., 0.], ["a", "a", "b", "b"]
    mean = sum(values) / len(values)
    null = independent_draws([v - mean for v in values], domains, 137)
    expected = (1 + sum(abs(v) >= abs(mean) - 1e-12 for v in null)) / 138
    assert analysis.centered_bootstrap_pvalue(values, domains, 137) == expected
    assert analysis.centered_bootstrap_pvalue([0., 0.], ["a", "a"], 11) == 1
    assert analysis.holm_adjusted_pvalues([.04, .001, .01, .02, .9]) == pytest.approx([.08, .005, .04, .06, .9])
    assert analysis.holm_adjusted_pvalues([.01] * 5) == pytest.approx([.05] * 5)
    rows = analysis.secondary_comparisons(small_accuracy(), 71)
    assert [(r["method"], r["baseline"]) for r in rows] == list(analysis.SECONDARY)
    assert len(rows) == 5 and all("approximate" in r["p_method"] for r in rows)
    assert all("ci97_5_bonferroni" not in r for r in rows)
    assert all(not any("mcnemar" in k or "exact" in k for k in r) for r in rows)


def latency_fixture():
    rows, metadata = [], {}
    pairs = [(1., 2.), (10., 10.), (100., 25.), (4., 16.)]
    for i, (tj, tb) in enumerate(pairs):
        pid = f"fake-{i}"
        metadata[pid] = {"domain": "a" if i < 2 else "b"}
        for condition, median in (("JFINAL", tj), ("B13_GREEDY", tb)):
            for j, seed in enumerate(analysis.SEEDS):
                for repetition in range(3):
                    # All nine observations affect median; failure is retained as a real measurement.
                    rows.append({"problem_id": pid, "condition": condition, "seed": seed, "repetition": repetition,
                                 "status": "timeout" if j == 0 else "final", "T_total": median * (1 + (j * 3 + repetition - 4) / 10)})
    return rows, metadata, pairs


def test_latency_is_median9_then_geometric_paired_not_ratio_of_arm_medians():
    rows, metadata, pairs = latency_fixture()
    result = analysis.paired_latency_v3(rows, metadata, 89)
    logs = [math.log(tb / tj) for tj, tb in pairs]
    expected = math.exp(sum(logs) / len(logs))
    samples = independent_draws(logs, ["a", "a", "b", "b"], 89)
    assert result["speedup"] == pytest.approx(expected)
    assert result["speedup"] != pytest.approx(np.median([b for a, b in pairs]) / np.median([a for a, b in pairs]))
    assert result["ci95_descriptive"] == pytest.approx(np.percentile(np.exp(samples), [2.5, 97.5]))
    assert result["ci97_5_bonferroni"] == pytest.approx(np.percentile(np.exp(samples), [1.25, 98.75]))
    assert result["n_records"] == 72 and result["n_clusters"] == 4
    assert result["per_problem"][0]["T_JFINAL_median9"] == 1
    assert result["status_counts"]["JFINAL"] == {"timeout": 12, "final": 24}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "zero", "nan", "seed", "rep", "plan"])
def test_latency_requires_complete_real_measurement_slots(fault):
    rows, metadata, _ = latency_fixture()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault in {"zero", "nan"}:
        rows[0]["T_total"] = 0 if fault == "zero" else float("nan")
    elif fault == "seed":
        rows[0]["seed"] = 99
    elif fault == "rep":
        rows[0]["repetition"] = 3
    else:
        metadata.pop("fake-0")
    with pytest.raises(ValueError):
        analysis.paired_latency_v3(rows, metadata, 10)


@pytest.mark.parametrize("text", ["FINAL: 2/4\n", "FINAL: .5", "FINAL: +1 / +2\n", "FINAL: 1/-2",
    "FINAL: \u22120.5", "FINAL: 1/0", "FINAL: 2+2", "FINAL: 0.5 kg", "FINAL: 1\n FINAL: .5",
    "final: .5", "Answer: .5", "FINAL: 1 \t/\t 2", "FINAL: .", "FINAL: 1e-1", "FINAL:"])
@pytest.mark.parametrize("status", ["final", "timeout", "truncated", "exception"])
def test_independent_parser_matches_canonical_strict_fields(text, status):
    pred, gold = prediction(text=text, status=status), gold_row()
    expected = grade_prediction(pred, gold)
    assert analysis.independent_grade_v3(pred, gold) == {k: expected[k] for k in analysis.GRADE_FIELDS}


def test_independent_parser_audits_actual_computed_6000_cases_and_detects_bad_canonical(monkeypatch):
    gold = {f"fake-{i}": gold_row(f"fake-{i}") for i in range(500)}
    predictions = [prediction(pid, seed, condition, text="FINAL: 2/4\n" if seed != 29 else "FINAL: 5\n")
                   for condition in analysis.ORDER for pid in gold
                   for seed in (analysis.SEEDS if condition in analysis.SAMPLED else (17,))]
    audit, grades = analysis.audit_predictions(predictions, gold)
    assert audit["n_predictions"] == 6000 and audit["n_disagreements"] == 0
    assert sum(g["correct"] for g in grades) == 4500
    original = analysis.grade_prediction
    monkeypatch.setattr(analysis, "grade_prediction", lambda p, g: {**original(p, g), "answer_normalized": "BAD"})
    audit, _ = analysis.audit_predictions(predictions[:3], gold)
    assert audit["n_disagreements"] == 3
    assert all(r["disagreements"] == ["answer_normalized"] for r in audit["rows"])


def pool(status="complete", texts=("FINAL: 1/2\n", "FINAL: 2\n", "FINAL: 2/4\n", "FINAL: 2\n")):
    pred = prediction()
    proposal = {k: pred[k] for k in ("candidate_set_id", "problem_id", "seed", "condition")}
    proposal["status"] = status
    candidates = [{**pred, "branch": i, "public_output": text} for i, text in enumerate(texts)]
    return pred, proposal, candidates


def test_shared_fixed_before_permutation_and_plurality_tie_is_lowest_original_branch():
    _, proposal, candidates = pool()
    proposal["permutation"] = [3, 2, 1, 0]
    result = analysis.shared_control_choices(proposal, list(reversed(candidates)))
    assert result["FIXED_BRANCH0_SHARED"]["branch"] == 0
    assert result["VOTE4SHARED"]["branch"] == 0
    # Gold-free selection is independent of a correct/wrong label on candidates.
    candidates[0]["correct"] = False
    candidates[1]["correct"] = True
    assert analysis.shared_control_choices(proposal, candidates)["VOTE4SHARED"]["branch"] == 0
    candidates[0]["status"] = "truncated"
    assert analysis.shared_control_choices(proposal, candidates)["VOTE4SHARED"]["branch"] == 1


def test_all_invalid_returns_branch0_normal_incorrect_and_no_latency_invented():
    pred, proposal, candidates = pool(texts=("No final", "FINAL: 1/0", "FINAL: x", "FINAL:"))
    grades, diagnostics = analysis.shared_controls([pred], [proposal], candidates, {"fake": gold_row()})
    assert all(not g["correct"] and g["chosen_branch"] == 0 for g in grades)
    assert diagnostics["oracle_at4_pct_diagnostic"] == 0
    assert diagnostics["correct_choice_given_any_correct_pct"] is None
    assert all("T_total" not in g for g in grades)


@pytest.mark.parametrize("status,count", [("partial", 1), ("no_candidates", 0)])
def test_failed_candidate_sets_all_controls_incorrect_and_four_slot_denominator(status, count):
    pred, proposal, candidates = pool(status=status)
    pred["status"] = "timeout"
    grades, diagnostics = analysis.shared_controls([pred], [proposal], candidates[:count], {"fake": gold_row()})
    assert all(not g["correct"] and g["chosen_branch"] is None for g in grades)
    assert diagnostics["n_cases"] == 1 and diagnostics["planned_candidate_slots"] == 4
    assert diagnostics["uniform_expected_accuracy_pct_diagnostic"] == 0
    assert diagnostics["pool_status_counts"] == {status: 1}


def test_uniform_expected_is_four_slots_not_only_valid_candidates():
    pred, proposal, candidates = pool(texts=("FINAL: .5", "invalid", "FINAL: 9", "invalid"))
    grades, diagnostics = analysis.shared_controls([pred], [proposal], candidates, {"fake": gold_row()})
    assert diagnostics["uniform_expected_accuracy_pct_diagnostic"] == 25
    assert diagnostics["oracle_at4_pct_diagnostic"] == 100
    assert all(g["candidate_set_id"] == pred["candidate_set_id"] for g in grades)


@pytest.mark.parametrize("status", ["timeout", "truncated", "exception", "engine_error"])
def test_failed_prediction_with_complete_pool_never_becomes_correct_control(status):
    pred, proposal, candidates = pool()
    pred["status"] = status
    grades, _ = analysis.shared_controls([pred], [proposal], candidates, {"fake": gold_row()})
    assert all(not row["correct"] and row["status"] == status for row in grades)


@pytest.mark.parametrize("fault", ["missing_proposal", "duplicate", "foreign_set", "foreign_case", "branch"])
def test_shared_pool_linkage_fail_closed(fault):
    pred, proposal, candidates = pool()
    proposals = [proposal]
    if fault == "missing_proposal":
        proposals = []
    elif fault == "duplicate":
        proposals.append(proposal)
    elif fault == "foreign_set":
        candidates[0]["candidate_set_id"] = "other"
    elif fault == "foreign_case":
        candidates[0]["seed"] = 29
    else:
        candidates[0]["branch"] = 1
    with pytest.raises(ValueError):
        analysis.shared_controls([pred], proposals, candidates, {"fake": gold_row()})


def test_co_primary_strict_ci_boundaries_and_practical_10pct_not_speedup1_10():
    quality = {"acc_diff_pp": 5., "ci97_5_bonferroni": [0., 10.]}
    latency = {"speedup": 1.10, "ci97_5_bonferroni": [1., 1.2]}
    result = analysis.co_primary_conclusions(quality, latency)
    assert not result["quality_superiority"] and not result["latency_superiority"] and not result["joint_superiority"]
    assert result["quality_practical_point_5pp"] and not result["latency_practical_point_10pct_reduction"]
    quality["ci97_5_bonferroni"][0] = .001
    latency.update(speedup=1 / .9, ci97_5_bonferroni=[1.001, 1.3])
    result = analysis.co_primary_conclusions(quality, latency)
    assert result["joint_superiority"] and result["latency_practical_point_10pct_reduction"]


def sealed_fake_data(tmp_path):
    verifier = analysis._integrity_module()
    data = tmp_path / "data"
    data.mkdir()
    gold = [gold_row(f"fake-{i:03}") for i in range(500)]
    for i, row in enumerate(gold):
        row["domain"] = f"d{i // 100}"
        row["sampling_tier"] = "easy" if i % 100 < 30 else "medium" if i % 100 < 70 else "hard"
        row["difficulty"] = "easy" if i % 100 < 70 else "medium"
    inputs = [{"id": g["id"], "problem": "Synthetic fixture " + g["id"], "language": "en"} for g in gold]
    pilot_gold = [gold_row(f"pilot-{i}") for i in range(50)]
    for i, row in enumerate(pilot_gold):
        row.update(domain=f"d{i // 10}", sampling_tier="easy" if i % 10 < 3 else "medium" if i % 10 < 7 else "hard",
                   difficulty="easy" if i % 10 < 7 else "medium")
    pilot_inputs = [{"id": g["id"], "problem": "Synthetic pilot fixture " + g["id"], "language": "en"} for g in pilot_gold]
    pool, decisions, assignments, review_records = [], {}, [], []
    for row, item in zip(gold + pilot_gold, inputs + pilot_inputs):
        candidate = {"problem": item["problem"], "language": "en", "source_id": "source-" + row["id"],
                     "provisional_domain": row["domain"], "provisional_difficulty": row["sampling_tier"],
                     "gold_numerator": 1, "gold_denominator": 2, "gold_answer": "1/2"}
        candidate["candidate_sha256"] = verifier.canonical_hash(candidate)
        candidate["id"] = "candidate-" + row["id"]
        pool.append(candidate)
        reviews = []
        for i in range(2):
            blind = {"id": candidate["id"], "reviewer_id": f"reviewer-{i}", "phase": "blind",
                     "candidate_sha256": candidate["candidate_sha256"], "answer": {"numerator": 1, "denominator": 2},
                     "domain": row["domain"], "difficulty": row["difficulty"],
                     "statement_ok": True, "unique_answer": True, "correctness": True,
                     "domain_ok": True, "difficulty_ok": True, "unit_explicit": True}
            reference = {"id": candidate["id"], "reviewer_id": f"reviewer-{i}", "phase": "reference",
                         "candidate_sha256": candidate["candidate_sha256"], "blind_record_sha256": verifier.canonical_hash(blind),
                         "gold_matches": True, "solution_consistent": True, "approve": True}
            review_records.extend((blind, reference))
            reviews.append({"reviewer_id": f"reviewer-{i}", "agent_run_id": f"run-{i}", "approved": True,
                            "blind_record_sha256": verifier.canonical_hash(blind),
                            "reference_record_sha256": verifier.canonical_hash(reference),
                            "original_verdicts": {"blind": {k: blind[k] for k in
                                ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")},
                                "reference": {k: reference[k] for k in ("gold_matches", "solution_consistent", "approve")}}})
        decision = {"status": "approved", "accepted": True, "final_domain": row["domain"],
                    "final_difficulty": row["difficulty"], "sampling_tier": row["sampling_tier"], "reviews": reviews,
                    "decision_type": "source_hint_contrast" if row["difficulty"] != row["sampling_tier"] else "ordinary",
                    "accepted_via_adjudication": False, "adjudication": None}
        decisions[candidate["id"]] = decision
        row.update({k: v for k, v in candidate.items() if k != "id"})
        row.update(candidate_id=candidate["id"], template_group=candidate["source_id"], agent_reviews=reviews,
                   review_decision=decision, accepted_via_adjudication=False)
        assignments.append({k: row[k] for k in ("id", "candidate_id", "candidate_sha256", "source_id", "agent_reviews")})
    unused = {"problem": "Unused synthetic source candidate", "language": "en", "source_id": "unused-source",
              "provisional_domain": "d0", "provisional_difficulty": "hard", "gold_numerator": 9,
              "gold_denominator": 1, "gold_answer": "9"}
    pool.append({**unused, "candidate_sha256": verifier.canonical_hash(unused), "id": "unused-candidate"})
    (data / "staging").mkdir()
    (data / "staging" / "candidate_pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool), encoding="utf-8")
    (data / "staging" / "original_reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in review_records), encoding="utf-8")
    for name, rows in (("test_gold", gold), ("test_inputs", inputs)):
        (data / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (data / "pilot_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pilot_inputs), encoding="utf-8")
    selected = [i for d in range(5) for i in [*(range(d * 100, d * 100 + 6)),
                                           *(range(d * 100 + 30, d * 100 + 38)), *(range(d * 100 + 70, d * 100 + 76))]]
    (data / "latency_inputs.jsonl").write_text("".join(json.dumps(inputs[i]) + "\n" for i in selected), encoding="utf-8")
    (data / "latency_plan.json").write_text(json.dumps({"quota_axis": verifier.QUOTA_AXIS,
        "quota_per_domain": verifier.SAMPLING_QUOTAS["latency"],
        "intrinsic_difficulty_counts": {"easy": 70, "medium": 30, "hard": 0}, "items": [
        {"item_id": gold[i]["id"], "problem": inputs[i]["problem"], "domain": gold[i]["domain"],
         "difficulty": gold[i]["difficulty"], "sampling_tier": gold[i]["sampling_tier"]} for i in selected]}), encoding="utf-8")
    policy = data / "REVIEW_CONTRACT.md"
    policy.write_text("Synthetic fixture policy only. Never authorizes actual agent review.\n", encoding="utf-8")
    profiles = {s: verifier.sampling_profile(rows, s) for s, rows in (("test", gold), ("pilot", pilot_gold))}
    review = {"agent_reviewed_all": True, "human_reviewed": False, "policy_id": verifier.REVIEW_POLICY_ID,
              "policy_sha256": analysis.sha256_file(policy), "source_sampling_tier_is_intrinsic_difficulty": False,
              "intrinsic_difficulty_counts": {s: profile["intrinsic_difficulty_counts"] for s, profile in profiles.items()}}
    metadata = {"dataset_version": "v3", "sealed": True, "actual_n": {"test": 500, "pilot": 50, "dev": 20}, "review": review}
    (data / "review_policy.json").write_text(json.dumps({"policy_id": verifier.REVIEW_POLICY_ID,
        "contract_sha256": review["policy_sha256"], "quota_axis": verifier.QUOTA_AXIS, "quota": verifier.SAMPLING_QUOTAS}), encoding="utf-8")
    (data / "ELIGIBILITY_POLICY.md").write_text("Synthetic eligibility policy only.\n", encoding="utf-8")
    metadata.update(status="FROZEN", construction_seed=20261002, latency_seed=20261003,
                    quota_axis=verifier.QUOTA_AXIS, quota=verifier.SAMPLING_QUOTAS,
                    test_reserved_before_pilot=True, latency_test_subset_only=True,
                    builder_sha256=verifier.sha256(b"synthetic builder fixture"),
                    source_revisions={"fixture": {"repo": "synthetic-source", "revision": "d" * 40}},
                    policy_artifact_sha256={name: analysis.sha256_file(data / name) for name in
                                           ("review_policy.json", "ELIGIBILITY_POLICY.md")})
    private = {"dataset_version": "v3", "status": "FROZEN_ANALYSIS_ONLY", "review": review,
               "quota_axis": verifier.QUOTA_AXIS, "selected_twice_approved_candidates": 550,
               "review_evidence_sha256": {"data/v3/staging/original_reviews.jsonl": analysis.sha256_file(data / "staging" / "original_reviews.jsonl")},
               "review_decisions": decisions, "selected_candidate_assignments": {"test": assignments[:500], "pilot": assignments[500:]},
               "strata": {s: {d: counts["source_sampling_tier_counts"] for d, counts in profile["by_domain"].items()} for s, profile in profiles.items()},
               "intrinsic_reviewed_strata": {s: {d: counts["intrinsic_difficulty_counts"] for d, counts in profile["by_domain"].items()} for s, profile in profiles.items()}}
    (data / "review_manifest.json").write_text(json.dumps(private), encoding="utf-8")
    metadata["analysis_manifest_sha256"] = analysis.sha256_file(data / "review_manifest.json")
    metadata["input_sha256"] = {name: analysis.sha256_file(data / name) for name in
        ("test_inputs.jsonl", "pilot_inputs.jsonl", "latency_inputs.jsonl")}
    (data / "dataset_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    (data / "SHA256SUMS").write_text("".join(f"{analysis.sha256_file(data / name)}  {name}\n" for name in
        ("test_gold.jsonl", "test_inputs.jsonl", "pilot_inputs.jsonl", "latency_inputs.jsonl", "latency_plan.json", "dataset_manifest.json", "REVIEW_CONTRACT.md",
         "review_manifest.json", "review_policy.json", "ELIGIBILITY_POLICY.md",
         "staging/candidate_pool.jsonl", "staging/original_reviews.jsonl")), encoding="utf-8")
    frozen = {"protocol_version": "3", "code_version": "0.3.0", "implementation_profile": "1COPY-G4-VLLM-GRAPH-V3",
              "models": {role: {"repo": "synthetic-" + role, "revision": "a" * 40} for role in "GJBO"},
               "jevk5_runtime": {"commit": "b" * 40}, "limits": {"n_candidates": 4, "max_output_tokens": 1024},
               "selector": {"calibration_temperature_expected": 1.0},
              "dataset_review_policy": {"id": review["policy_id"], "sha256": review["policy_sha256"]}}
    frozen["models"]["O"] = {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(frozen), encoding="utf-8")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("generador", "criterio_paso", "criterio_final"):
        (prompts / (name + ".txt")).write_text(name + "\n", encoding="utf-8")
    approval = approved((data, data / "test_inputs.jsonl", config, prompts, frozen))
    return verifier, data, gold, inputs, config, frozen, prompts, approval


def reseal(data):
    path = data / "SHA256SUMS"
    names = [line.split(maxsplit=1)[1] for line in path.read_text().splitlines()]
    path.write_text("".join(f"{analysis.sha256_file(data / name)}  {name}\n" for name in names), encoding="utf-8")


@pytest.mark.parametrize("fault", [None, "duplicate", "human", "agent", "policy", "seal", "version", "500", "answer"])
def test_sealed_gold_review_and_exact500_inputs_no_legacy_autofallback(tmp_path, fault):
    _, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    if fault in {"duplicate", "human", "agent", "500", "answer"}:
        if fault == "duplicate":
            gold[-1] = gold[0]
        elif fault == "human":
            gold[0]["human_reviewed"] = True
        elif fault == "agent":
            gold[0]["agent_reviewed"] = False
        elif fault == "500":
            gold.pop()
        else:
            gold[0]["gold_answer"] = "3/4"
        (data / "test_gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
        reseal(data)
    elif fault == "policy":
        (data / "REVIEW_CONTRACT.md").write_text("different policy")
        reseal(data)
    elif fault == "seal":
        (data / "test_gold.jsonl").write_text((data / "test_gold.jsonl").read_text() + "\n")
    elif fault == "version":
        frozen = json.loads(config.read_text())
        frozen["protocol_version"] = "2"
        config.write_text(json.dumps(frozen))
    if fault is None:
        result, meta = analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)
        assert len(result) == 500 and meta["review"]["human_reviewed"] is False
    else:
        with pytest.raises(ValueError):
            analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)


def bind_fake_review(data, gold, private):
    records = [json.loads(line) for line in (data / "staging" / "original_reviews.jsonl").read_text().splitlines()
               if json.loads(line).get("phase") != "adjudication"]
    records.extend(row["review_decision"]["adjudication"]["record"] for row in gold
                   if row.get("accepted_via_adjudication") is True)
    (data / "staging" / "original_reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    private["review_evidence_sha256"]["data/v3/staging/original_reviews.jsonl"] = analysis.sha256_file(data / "staging" / "original_reviews.jsonl")
    levels = ("easy", "medium", "hard")
    private["intrinsic_reviewed_strata"]["test"] = {d: {level: sum(r["domain"] == d and r["difficulty"] == level for r in gold)
                                                      for level in levels} for d in {r["domain"] for r in gold}}
    (data / "test_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
    (data / "review_manifest.json").write_text(json.dumps(private))
    metadata = json.loads((data / "dataset_manifest.json").read_text())
    metadata["review"]["intrinsic_difficulty_counts"]["test"] = {level: sum(r["difficulty"] == level for r in gold) for level in levels}
    metadata["analysis_manifest_sha256"] = analysis.sha256_file(data / "review_manifest.json")
    (data / "dataset_manifest.json").write_text(json.dumps(metadata))
    reseal(data)


def adjudicate_fake_row(data, gold, private, verifier):
    row = gold[0]
    decision = row["review_decision"]
    # Keep the original false label verdict and its source-blind phase hashes.
    row["agent_reviews"][0]["approved"] = False
    path = data / "staging" / "original_reviews.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0].update(difficulty="hard", difficulty_ok=False)
    records[1].update(approve=False, blind_record_sha256=verifier.canonical_hash(records[0]))
    original = row["agent_reviews"][0]
    original.update(blind_record_sha256=verifier.canonical_hash(records[0]),
                    reference_record_sha256=verifier.canonical_hash(records[1]))
    original["original_verdicts"]["blind"]["difficulty_ok"] = False
    original["original_verdicts"]["reference"]["approve"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    row["difficulty"] = "hard"
    record = {"id": row["candidate_id"], "phase": "adjudication", "reviewer_id": "third-agent",
              "candidate_sha256": row["candidate_sha256"], "final_domain": row["domain"], "final_difficulty": "hard",
               "answer": {"numerator": 1, "denominator": 2},
              "original_unit_concerns": [], "unit_resolution": "unchanged", "dimensionless_query": False,
              "dimensionless_kind": "not_dimensionless",
              "original_review_refs": [{k: r[k] for k in ("reviewer_id", "blind_record_sha256", "reference_record_sha256")}
                                       for r in row["agent_reviews"]],
              **{k: True for k in ("answer_correct", "gold_matches", "solution_consistent", "statement_ok",
                                  "unique_answer", "correctness", "approve")}}
    decision.update(status="resolved", final_difficulty="hard", accepted_via_adjudication=True, adjudication_eligible=True,
                    adjudication={"reviewer_id": "third-agent", "agent_run_id": "third-run", "record": record,
                                  "record_sha256": verifier.canonical_hash(record)})
    row["accepted_via_adjudication"] = True
    private["review_decisions"][row["candidate_id"]] = decision
    private["selected_candidate_assignments"]["test"][0]["agent_reviews"] = row["agent_reviews"]


def test_final_adjudicated_labels_not_provisional_classes_define_clusters(tmp_path):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private = json.loads((data / "review_manifest.json").read_text())
    adjudicate_fake_row(data, gold, private, verifier)
    bind_fake_review(data, gold, private)
    result, provenance = analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)
    assert result[gold[0]["id"]]["difficulty"] == "hard"
    assert result[gold[0]["id"]]["provisional_difficulty"] == "easy"
    assert result[gold[0]["id"]]["agent_reviews"][0]["approved"] is False
    assert provenance["review_manifest_sha256"] == analysis.sha256_file(data / "review_manifest.json")


@pytest.mark.parametrize("phase", ["blind", "adjudication"])
def test_original_review_answers_bind_exact_rational_value_not_representation(tmp_path, phase):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private = json.loads((data / "review_manifest.json").read_text())
    row = gold[0]
    if phase == "adjudication":
        adjudicate_fake_row(data, gold, private, verifier)
        adj = row["review_decision"]["adjudication"]
        adj["record"]["answer"] = {"numerator": 2, "denominator": 4}
        adj["record_sha256"] = verifier.canonical_hash(adj["record"])
    else:
        path = data / "staging" / "original_reviews.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["answer"] = {"numerator": 2, "denominator": 4}
        records[1]["blind_record_sha256"] = verifier.canonical_hash(records[0])
        original = row["agent_reviews"][0]
        original.update(blind_record_sha256=verifier.canonical_hash(records[0]),
                        reference_record_sha256=verifier.canonical_hash(records[1]))
        private["selected_candidate_assignments"]["test"][0]["agent_reviews"] = row["agent_reviews"]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    private["review_decisions"][row["candidate_id"]] = row["review_decision"]
    bind_fake_review(data, gold, private)
    result, _ = analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)
    assert result[row["id"]]["gold_answer"] == "1/2"


@pytest.mark.parametrize("fault", [None, "reason", "concerns", "resolution", "query"])
def test_unit_adjudication_preserves_original_false_verdict_and_source_reason(tmp_path, fault):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private = json.loads((data / "review_manifest.json").read_text())
    adjudicate_fake_row(data, gold, private, verifier)
    row = gold[0]
    path = data / "staging" / "original_reviews.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0].update(unit_explicit=False, independent_solution="Original synthetic independent unit concern.")
    records[1].update(reference_comparison="Original synthetic source comparison.",
                      blind_record_sha256=verifier.canonical_hash(records[0]))
    original = row["agent_reviews"][0]
    original.update(blind_record_sha256=verifier.canonical_hash(records[0]),
                    reference_record_sha256=verifier.canonical_hash(records[1]),
                    original_unit_reason=records[0]["independent_solution"] + "\n" + records[1]["reference_comparison"])
    original["original_verdicts"]["blind"]["unit_explicit"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    adj = row["review_decision"]["adjudication"]
    adj["record"].update(original_unit_concerns=[{k: original[k] for k in
        ("reviewer_id", "blind_record_sha256", "original_unit_reason")} | {"unit_explicit": False}],
        original_review_refs=[{k: r[k] for k in ("reviewer_id", "blind_record_sha256", "reference_record_sha256")}
                              for r in row["agent_reviews"]],
        unit_resolution="inherently_dimensionless", dimensionless_query=True, dimensionless_kind="ratio",
        query_quote=row["problem"])
    if fault == "reason":
        original["original_unit_reason"] = "Replacement reason"
        adj["record"]["original_unit_concerns"][0]["original_unit_reason"] = "Replacement reason"
    elif fault == "concerns":
        adj["record"]["original_unit_concerns"] = []
    elif fault == "resolution":
        adj["record"]["unit_resolution"] = "unchanged"
    elif fault == "query":
        adj["record"]["query_quote"] = "Not in the original statement"
    adj["record_sha256"] = verifier.canonical_hash(adj["record"])
    private["review_decisions"][row["candidate_id"]] = row["review_decision"]
    private["selected_candidate_assignments"]["test"][0]["agent_reviews"] = row["agent_reviews"]
    bind_fake_review(data, gold, private)
    if fault is None:
        result, _ = analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)
        assert result[row["id"]]["agent_reviews"][0]["original_verdicts"]["blind"]["unit_explicit"] is False
    else:
        with pytest.raises(ValueError, match="unit|dimensionless"):
            analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)


@pytest.mark.parametrize("fault", ["commitment", "assignment", "source", "candidate", "problem", "final_label",
                                  "missing_decision", "unaccepted", "duplicate_agent", "original_hash",
                                  "adjudicator", "adjudication_gold", "original_refs", "adjudication_hash",
                                  "missing_evidence", "phase", "phase_answer", "phase_candidate", "original_verdict",
                                  "adjudication_eligibility", "adjudication_units"])
def test_final_label_and_source_provenance_fail_closed_even_when_resealed(tmp_path, fault):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private = json.loads((data / "review_manifest.json").read_text())
    row = gold[0]
    if fault.startswith("adjudication") or fault in {"adjudicator", "original_refs"}:
        adjudicate_fake_row(data, gold, private, verifier)
        adj = row["review_decision"]["adjudication"]
        if fault == "adjudicator":
            adj["agent_run_id"] = "run-0"
        elif fault == "adjudication_gold":
            adj["record"]["answer"]["numerator"] = 9
        elif fault == "original_refs":
            adj["record"]["original_review_refs"].pop()
        elif fault == "adjudication_eligibility":
            row["review_decision"]["adjudication_eligible"] = False
        elif fault == "adjudication_units":
            adj["record"]["original_unit_concerns"] = [{"unit_explicit": False}]
        else:
            adj["record_sha256"] = "f" * 64
        if fault != "adjudication_hash":
            adj["record_sha256"] = verifier.canonical_hash(adj["record"])
    elif fault == "assignment":
        private["selected_candidate_assignments"]["test"].pop()
    elif fault == "source":
        row["source_id"] = "other"
        private["selected_candidate_assignments"]["test"][0]["source_id"] = "other"
    elif fault == "candidate":
        row["candidate_sha256"] = "f" * 64
        private["selected_candidate_assignments"]["test"][0]["candidate_sha256"] = "f" * 64
    elif fault == "problem":
        row["problem"] = "Altered statement"
    elif fault == "final_label":
        row["difficulty"] = "hard"
    elif fault == "missing_decision":
        row.pop("review_decision")
    elif fault == "unaccepted":
        row["review_decision"]["accepted"] = False
    elif fault == "duplicate_agent":
        row["agent_reviews"][1]["agent_run_id"] = "run-0"
    elif fault == "original_hash":
        row["agent_reviews"][0]["blind_record_sha256"] = "missing"
    elif fault == "original_verdict":
        row["agent_reviews"][0]["original_verdicts"]["blind"]["difficulty_ok"] = False
    elif fault == "missing_evidence":
        private["review_evidence_sha256"] = {}
    elif fault in {"phase", "phase_answer", "phase_candidate"}:
        path = data / "staging" / "original_reviews.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "phase":
            records[0]["phase"] = "reference"
        elif fault == "phase_answer":
            records[0]["answer"]["numerator"] = 9
        else:
            records[0]["candidate_sha256"] = "f" * 64
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    if "review_decision" in row:
        private["review_decisions"][row["candidate_id"]] = row["review_decision"]
        private["selected_candidate_assignments"]["test"][0]["agent_reviews"] = row["agent_reviews"]
    bind_fake_review(data, gold, private)
    if fault == "missing_evidence":
        private["review_evidence_sha256"] = {}
        (data / "review_manifest.json").write_text(json.dumps(private))
        metadata = json.loads((data / "dataset_manifest.json").read_text())
        metadata["analysis_manifest_sha256"] = analysis.sha256_file(data / "review_manifest.json")
        (data / "dataset_manifest.json").write_text(json.dumps(metadata))
        reseal(data)
    if fault == "commitment":
        metadata = json.loads((data / "dataset_manifest.json").read_text())
        metadata["analysis_manifest_sha256"] = "f" * 64
        (data / "dataset_manifest.json").write_text(json.dumps(metadata))
        reseal(data)
    with pytest.raises(ValueError):
        analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)


def test_analysis_cli_requires_explicit_gold_authorization_without_reading_paths(capsys):
    with pytest.raises(SystemExit) as error:
        analysis.main(["--results", "missing", "--gold", "never-read", "--output", "missing", "--config", "missing",
                       "--approval", "missing", "--run-tag", "mock"])
    assert error.value.code == 2 and "--analyze is required" in capsys.readouterr().err


def test_analysis_cli_corrupt_zip_returns_error_schema_before_gold(tmp_path, monkeypatch, capsys):
    root, data, config, prompts, approval, _, _ = fake_matrix(tmp_path)
    next(root.rglob("*_final.zip")).write_bytes(b"not a ZIP archive")
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "test_gold.jsonl", "Gold read before corrupt ZIP rejection"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    code = analysis.main(["--analyze", "--results", str(root), "--gold", str(data / "test_gold.jsonl"),
        "--inputs", str(data / "test_inputs.jsonl"), "--output", str(tmp_path / "analysis"), "--config", str(config),
        "--prompts", str(prompts), "--approval", str(approval), "--pilot-report", str(data / "pilot_go.json"), "--run-tag", "mock"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and result["ok"] is False and "ZIP integrity failure" in result["error"]


def fake_matrix(tmp_path):
    verifier, data, gold, inputs, config, frozen, prompts, approval = sealed_fake_data(tmp_path)
    root = tmp_path / "results"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    subsets = [i for d in range(5) for i in [*(range(d * 100, d * 100 + 6)),
                                           *(range(d * 100 + 30, d * 100 + 38)), *(range(d * 100 + 70, d * 100 + 76))]]
    snapshots = {}
    for condition in (*analysis.ORDER, "LATENCY"):
        cfg = {"protocol_version": "3", "code_version": "0.3.0", "models": frozen["models"],
               "code_sha256": {"v3/runner.py": "a" * 64},
               "profile": frozen["implementation_profile"], "jevk5_runtime": frozen["jevk5_runtime"],
               "experiment_config_sha256": analysis.sha256_file(config),
               "data_sha256": {name: analysis.sha256_file(data / name) for name in
                               ("pilot_inputs.jsonl", "test_inputs.jsonl", "dataset_manifest.json",
                                "latency_inputs.jsonl", "latency_plan.json", "SHA256SUMS")},
               "prompt_sha256": {name: verifier.sha256(name.encode()) for name in
                                 ("generador", "criterio_paso", "criterio_final")},
                "params": {"CONDITION": condition, "SPLIT": "test", "RUN_TAG": "mock", "N_CANDIDATES": 4,
                           "CANDIDATE_BUDGET": 4096, "GB_CONTEXT": 4096,
                           "TEMPERATURE": .7, "TOP_P": .9, "TOP_K": 0, "REPETITION_PENALTY": 1.,
                           "PRESENCE_PENALTY": 0., "FREQUENCY_PENALTY": 0.,
                           "MAX_OUTPUT_TOKENS": 1024, "SEEDS": "17,29,43" if condition in analysis.SAMPLED or condition == "LATENCY" else "17"}}
        digest = verifier.canonical_hash(cfg)
        snap = {name: [] for name in verifier.LEDGERS}
        plan_items = [{"item_id": gold[i]["id"], "domain": gold[i]["domain"], "difficulty": gold[i]["difficulty"],
                       "sampling_tier": gold[i]["sampling_tier"], "problem": inputs[i]["problem"]} for i in subsets]
        snap["latency_plan"] = {"items": plan_items} if condition == "LATENCY" else None
        if condition == "LATENCY":
            scheduled = [(i, arm, seed, rep) for i in subsets for arm in ("JFINAL", "B13_GREEDY")
                         for seed in analysis.SEEDS for rep in range(3)]
        else:
            scheduled = [(i, condition, seed, None) for i in range(500)
                         for seed in (analysis.SEEDS if condition in analysis.SAMPLED else (17,))]
        for case_index, (i, arm, seed, rep) in enumerate(scheduled):
            pid = gold[i]["id"]
            key = verifier.sha256((f"{digest}|{pid}|{seed}|{arm}|{cfg['profile']}" if rep is None
                                   else f"{digest}|{pid}|{arm}|{seed}|{rep}").encode())
            total = 8. if condition == "LATENCY" and arm == "B13_GREEDY" else 4.
            row = {**prediction(pid, seed, arm, text="FINAL: 1/2\n" if arm == "JFINAL" or i < 250 else "FINAL: 9\n"),
                   "config_hash": digest, "profile": cfg["profile"], "resume_key": key, "gpu_uuid": "GPU-fixture",
                   "T_total": total, "T_first_accepted": total, "t_start_perf": case_index * 10., "t_end_perf": case_index * 10. + total,
                   "T_proposals": 2. if arm == "JFINAL" else total, "T_selector": .5 if arm == "JFINAL" else 0.,
                   "accepted_tokens": 20, "candidate_tokens_total": 80 if arm == "JFINAL" else 20,
                   "effective_seed": seed if arm in analysis.SAMPLED else None,
                   "t_start_utc": (start + timedelta(seconds=case_index * 10)).isoformat(),
                   "t_end_utc": (start + timedelta(seconds=case_index * 10 + total)).isoformat(),
                   "clock_scope": "candidate_generation_inclusive", "execution_id": f"fixture-{condition}-{case_index}"}
            if rep is not None:
                row.update(item_id=pid, repetition=rep, **{k: gold[i][k] for k in ("domain", "difficulty", "sampling_tier")})
                snap["latency_metrics"].append(copy.deepcopy(row))
            snap["predictions"].append(copy.deepcopy(row))
            snap["metrics"].append(copy.deepcopy(row))
            if arm == "JFINAL":
                row["winner_branch"] = 0
                snap["predictions"][-1]["winner_branch"] = snap["metrics"][-1]["winner_branch"] = 0
                if rep is not None:
                    snap["latency_metrics"][-1]["winner_branch"] = 0
                proposal = {**row, "status": "complete", "candidates": [
                    {"branch": branch, "seed": seed, "seed_branch": seed * 100 + branch, "status": "final", "public_output": "FINAL: 1/2\n"}
                    for branch in range(4)]}
                snap["proposals"].append(proposal)
        snap["progress"] = {"done": len(scheduled), "total": len(scheduled)}
        snap["manifest"] = {"condition": condition, "config": cfg, "config_hash": digest,
                            "run_name": f"{condition}_mock_{digest[:8]}", "created_utc": start.isoformat(),
                            "finished_utc": (start + timedelta(days=1)).isoformat(), "gpu_uuid": "GPU-fixture",
                            "counts": {name: len(snap[name]) for name in verifier.LEDGERS}, "stages": {"fixture": {"ok": True}},
                            "preflight": {name: {"blocking": True, "ok": True} for name in
                                          verifier.BLOCKING | {"selector_native_reference", "selector_equivalence"}},
                            "model_lock": {r: {**frozen["models"][r], "ok": True} for r in "GJBO"},
                            "environment": {"cuda_available": True, "gpus": [
                                {"name": "NVIDIA A100", "memory.total": 40960, "uuid": "GPU-fixture"}]},
                            "hw_probe": {"cuda": "synthetic", "device": "NVIDIA A100", "bf16_gemm_tflops": 1.,
                                         "d2d_copy_gbps": 1., "small_kernel_us": 1., "python_loop_ms": 1.}}
        snapshots[condition] = snap
        complete_fixture(snap, (data, data / "test_inputs.jsonl", config, prompts, frozen))
    for condition, snap in snapshots.items():
        save_snapshot(root / condition, snap, verifier)
    return root, data, config, prompts, approval, snapshots, verifier


def save_snapshot(directory, snapshot, verifier):
    refresh_traces(snapshot)
    write_archive(directory, snapshot)


def test_full_generated_matrix_emits_actual_500_6000_1800_analysis(tmp_path):
    root, data, config, prompts, approval, _, _ = fake_matrix(tmp_path)
    result = analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                                   config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")
    saved = json.loads((tmp_path / "analysis" / "analysis.json").read_text())
    assert saved["meta"]["historical_compatibility"] is False
    assert saved["primary"]["quality"]["acc_diff_pp"] == 50
    assert saved["primary"]["quality"]["n_clusters"] == 500
    assert saved["primary"]["latency"]["speedup"] == pytest.approx(2.)
    assert saved["primary"]["latency"]["n_records"] == 1800 and saved["primary"]["latency"]["n_clusters"] == 100
    assert saved["parser_audit"]["n_predictions"] == 6000 and saved["parser_audit"]["n_disagreements"] == 0
    assert saved["jfinal"]["n_cases"] == 1500 and saved["jfinal"]["planned_candidate_slots"] == 6000
    assert result["declarations"]["joint_superiority"]
    assert [(r["method"], r["baseline"]) for r in result["secondary"]] == list(analysis.SECONDARY)
    assert all(Path(path).is_file() for path in result["paths"].values())
    controls = [r for r in saved["table"] if r["condition"] not in analysis.ORDER]
    assert all(r["latency_p50_s_descriptive"] is None and r["n_cases"] == 1500 for r in controls)
    profile = saved["meta"]["dataset_profile"]
    assert profile["quota_axis"] == "source_sampling_tier" and profile["n_agent_reviewed"] == 550
    assert profile["human_reviewed"] is False and profile["n_unused_source_candidates"] == 1
    assert profile["splits"]["test"]["source_sampling_tier_counts"] == {"easy": 150, "medium": 200, "hard": 150}
    assert profile["splits"]["test"]["intrinsic_difficulty_counts"] == {"easy": 350, "medium": 150, "hard": 0}
    assert profile["splits"]["pilot"]["source_sampling_tier_counts"] == {"easy": 15, "medium": 20, "hard": 15}
    assert profile["splits"]["latency"]["source_sampling_tier_counts"] == {"easy": 30, "medium": 40, "hard": 30}
    assert profile["splits"]["latency"]["intrinsic_difficulty_counts"] == {"easy": 70, "medium": 30, "hard": 0}
    assert profile["unused_subsets"] == {"source_candidates_not_selected": 1, "pilot_excluded_from_confirmatory_quality": 50,
        "test_not_in_latency": {"n_problems": 400, "source_sampling_tier_counts": {"easy": 120, "medium": 160, "hard": 120},
                                "intrinsic_difficulty_counts": {"easy": 280, "medium": 120, "hard": 0}}}
    assert "source-tier or intrinsic-difficulty tuning" in result["report"]
    assert "Source-Tier Hard" in result["report"] and "Intrinsic Hard" in result["report"]
    assert "Source tier hard does not mean genuinely hard" in result["report"]
    assert set(result["df"].sampling_tier) == {"easy", "medium", "hard"}


def test_co_primary_domain_statistics_ignore_source_tier_and_intrinsic_labels():
    original = small_accuracy()
    labelled = original.assign(sampling_tier="hard", difficulty="easy")
    assert analysis.paired_accuracy_v3(original, n_boot=101) == analysis.paired_accuracy_v3(labelled, n_boot=101)
    records, metadata, _ = latency_fixture()
    a = analysis.paired_latency_v3(records, metadata, 101)
    labelled_metadata = {pid: {**row, "sampling_tier": "hard", "difficulty": "easy"} for pid, row in metadata.items()}
    b = analysis.paired_latency_v3(records, labelled_metadata, 101)
    assert {k: v for k, v in a.items() if k != "per_problem"} == {k: v for k, v in b.items() if k != "per_problem"}


def test_source_hint_contrast_uses_fresh_agreed_intrinsic_domains_without_editing_source(tmp_path):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private = json.loads((data / "review_manifest.json").read_bytes())
    source_bytes = (data / "staging" / "candidate_pool.jsonl").read_bytes()
    records_path = data / "staging" / "original_reviews.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    # Opposite-domain source-easy IDs retain every100/domain and30/40/30 tier count.
    for row, domain in ((gold[0], "d1"), (gold[100], "d0")):
        row["domain"] = domain
        row["review_decision"].update(final_domain=domain, decision_type="source_hint_contrast")
        for original in row["agent_reviews"]:
            blind = next(r for r in records if r["id"] == row["candidate_id"] and r["reviewer_id"] == original["reviewer_id"] and r["phase"] == "blind")
            reference = next(r for r in records if r["id"] == row["candidate_id"] and r["reviewer_id"] == original["reviewer_id"] and r["phase"] == "reference")
            blind["domain"] = domain
            reference["blind_record_sha256"] = verifier.canonical_hash(blind)
            original.update(blind_record_sha256=verifier.canonical_hash(blind),
                            reference_record_sha256=verifier.canonical_hash(reference), reasons=["domain_mismatch"])
        private["review_decisions"][row["candidate_id"]] = row["review_decision"]
        next(r for r in private["selected_candidate_assignments"]["test"] if r["id"] == row["id"])["agent_reviews"] = row["agent_reviews"]
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    bind_fake_review(data, gold, private)
    result, profile = analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)
    assert (data / "staging" / "candidate_pool.jsonl").read_bytes() == source_bytes
    assert result[gold[0]["id"]]["domain"] == "d1" and result[gold[0]["id"]]["provisional_domain"] == "d0"
    assert profile["dataset_profile"]["n_agent_reviewed"] == 550


@pytest.mark.parametrize("fault", ["tier", "tier_quota", "intrinsic_quota_axis", "intrinsic_counts", "selected_source_counts",
                                   "pilot_missing", "pilot_overlap", "pilot_review", "ordinary_false_veto"])
def test_source_sampling_selection_and_all550_reviewed_bindings_fail_closed(tmp_path, fault):
    verifier, data, gold, _, config, _, _, _ = sealed_fake_data(tmp_path)
    private_path = data / "review_manifest.json"
    private = json.loads(private_path.read_bytes())
    if fault in {"tier", "tier_quota"}:
        gold[0]["sampling_tier"] = "hard"
        if fault == "tier_quota":
            private["review_decisions"][gold[0]["candidate_id"]]["sampling_tier"] = "hard"
        (data / "test_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
    elif fault == "intrinsic_quota_axis":
        private["quota_axis"] = "intrinsic_difficulty"
    elif fault == "intrinsic_counts":
        metadata = json.loads((data / "dataset_manifest.json").read_bytes())
        metadata["review"]["intrinsic_difficulty_counts"]["test"] = {"easy": 150, "medium": 200, "hard": 150}
        (data / "dataset_manifest.json").write_text(json.dumps(metadata))
    elif fault == "selected_source_counts":
        private["strata"]["test"]["d0"]["hard"] = 29
    elif fault == "pilot_missing":
        private["selected_candidate_assignments"]["pilot"].pop()
    elif fault == "pilot_overlap":
        private["selected_candidate_assignments"]["pilot"][0]["candidate_id"] = gold[0]["candidate_id"]
    elif fault == "pilot_review":
        private["selected_candidate_assignments"]["pilot"][0]["agent_reviews"] = []
    else:
        row = gold[30]
        row["agent_reviews"][0]["original_verdicts"]["blind"]["difficulty_ok"] = False
        (data / "test_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
        private["review_decisions"][row["candidate_id"]] = row["review_decision"]
        private["selected_candidate_assignments"]["test"][30]["agent_reviews"] = row["agent_reviews"]
    private_path.write_text(json.dumps(private))
    metadata_path = data / "dataset_manifest.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["analysis_manifest_sha256"] = analysis.sha256_file(private_path)
    metadata_path.write_text(json.dumps(metadata))
    reseal(data)
    with pytest.raises(ValueError):
        analysis.validate_gold_v3(data / "test_gold.jsonl", data / "test_inputs.jsonl", config)


@pytest.mark.parametrize("fault", ["duplicate", "foreign_id", "wrong_seed", "wrong_arm"])
def test_real_verifier_rejects_fullmatrix_binding_before_gold_read(tmp_path, monkeypatch, fault):
    root, data, config, prompts, approval, snapshots, verifier = fake_matrix(tmp_path)
    snapshot = snapshots["Q9_GREEDY"]
    for ledger in ("predictions", "metrics"):
        rows = snapshot[ledger]
        if fault == "duplicate":
            rows[-1] = copy.deepcopy(rows[0])
        elif fault == "foreign_id":
            rows[0]["problem_id"] = "foreign"
        elif fault == "wrong_seed":
            rows[0]["seed"] = 29
        else:
            rows[0]["condition"] = "B13_GREEDY"
    save_snapshot(root / "Q9_GREEDY", snapshot, verifier)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "test_gold.jsonl", "Malformed matrix reached gold"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="Execution integrity failed"):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                              config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")


@pytest.mark.parametrize("fault", ["duplicate", "missing_arm_id", "foreign_id", "wrong_arm", "wrong_seed", "fractional_seed"])
def test_analysis_rechecks_full_physical_matrix_when_load_runs_is_replaced(tmp_path, monkeypatch, fault):
    root, data, config, prompts, approval, snapshots, _ = fake_matrix(tmp_path)
    rows = snapshots["Q9_GREEDY"]["predictions"]
    if fault in {"duplicate", "missing_arm_id"}:
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "foreign_id":
        rows[0]["problem_id"] = "foreign"
    elif fault == "wrong_arm":
        rows[0]["condition"] = "B13_GREEDY"
    elif fault == "wrong_seed":
        rows[0]["seed"] = 29
    else:
        rows[0]["seed"] = 17.0
    monkeypatch.setattr(analysis, "load_runs", lambda *a, **k: {"snapshots": snapshots, "provenance": {}})
    monkeypatch.setattr(analysis, "audit_predictions", lambda *a: pytest.fail("Malformed matrix reached grading"))
    with pytest.raises(ValueError, match="Physical matrix"):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                              config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")


@pytest.mark.parametrize("fault", ["omitted_branch", "omitted_partial_pool", "omitted_proposal", "duplicate_set"])
def test_analysis_rechecks_shared_pool_without_loader_validation(tmp_path, monkeypatch, fault):
    root, data, config, prompts, approval, snapshots, _ = fake_matrix(tmp_path)
    snap = snapshots["JFINAL"]
    if fault == "omitted_branch":
        snap["proposals"][0]["candidates"].pop()
    elif fault == "omitted_partial_pool":
        snap["proposals"][0].update(status="partial", candidates=[])
    elif fault == "omitted_proposal":
        snap["proposals"].pop()
    else:
        snap["predictions"][1]["candidate_set_id"] = snap["predictions"][0]["candidate_set_id"]
    monkeypatch.setattr(analysis, "load_runs", lambda *a, **k: {"snapshots": snapshots, "provenance": {}})
    with pytest.raises(ValueError):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                              config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")


def test_independent_full_matrix_failure_is_incorrect_and_all_latency_failures_retained(tmp_path, monkeypatch):
    root, data, config, prompts, approval, snapshots, _ = fake_matrix(tmp_path)
    snap = snapshots["JFINAL"]
    snap["predictions"][0]["status"] = "timeout"
    snap["proposals"][0].update(status="partial", candidates=snap["proposals"][0]["candidates"][:1])
    for row in snapshots["LATENCY"]["latency_metrics"]:
        row["status"] = "timeout"
    monkeypatch.setattr(analysis, "load_runs", lambda *a, **k: {"snapshots": snapshots, "provenance": {}})
    result = analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                                   config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")
    assert result["primary"]["quality"]["acc_diff_pp"] == pytest.approx(50 - 100 / 1500)
    assert result["primary"]["quality"]["n_clusters"] == 500
    assert result["primary"]["latency"]["n_clusters"] == 100
    assert result["primary"]["latency"]["n_records"] == 1800
    assert result["primary"]["latency"]["status_counts"] == {arm: {"timeout": 900} for arm in ("JFINAL", "B13_GREEDY")}
    assert result["primary"]["latency"]["speedup"] == pytest.approx(2)
    assert result["meta"]["bootstrap_resamples"] == 10000 and result["meta"]["bootstrap_seed"] == 271828
    assert result["jfinal"]["planned_candidate_slots"] == 6000
    assert all(r["n_cases"] == 1500 and r["incorrect_cases"] == 1
               for r in result["table"] if r["condition"] not in analysis.ORDER)


def test_analysis_rejects_duplicate_plan_items_even_with100_distinct_ids(tmp_path, monkeypatch):
    root, data, config, prompts, approval, snapshots, _ = fake_matrix(tmp_path)
    items = snapshots["LATENCY"]["latency_plan"]["items"]
    items.append(copy.deepcopy(items[0]))
    monkeypatch.setattr(analysis, "load_runs", lambda *a, **k: {"snapshots": snapshots, "provenance": {}})
    with pytest.raises(ValueError, match="exactly100 unique"):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                              config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")


def test_failed_analysis_removes_previous_declarations_and_saves_disagreement_audit(tmp_path, monkeypatch):
    root, data, config, prompts, approval, _, _ = fake_matrix(tmp_path)
    out = tmp_path / "analysis"
    analysis.run_analysis(root, data / "test_gold.jsonl", out, "mock", config=config, prompts=prompts, approval=approval,
                          pilot_report=data / "pilot_go.json")
    original = analysis.grade_prediction
    monkeypatch.setattr(analysis, "grade_prediction", lambda p, g: {**original(p, g), "answer_normalized": "broken"})
    with pytest.raises(ValueError, match="Independent parser disagrees"):
        analysis.run_analysis(root, data / "test_gold.jsonl", out, "mock", config=config, prompts=prompts, approval=approval,
                              pilot_report=data / "pilot_go.json")
    assert not (out / "analysis.json").exists() and not (out / "report.md").exists()
    assert json.loads((out / "parser_audit.json").read_text())["n_disagreements"] == 6000


def test_missing_fulltest_approval_blocks_before_gold_is_read(tmp_path, monkeypatch):
    root, data, config, prompts, _, _, _ = fake_matrix(tmp_path)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "test_gold.jsonl", "Gold read before execution/approval validation"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="budget approval required"):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock", config=config, prompts=prompts)


def test_resealing_replacement_gold_cannot_change_truth_bound_to_saved_runs(tmp_path, monkeypatch):
    root, data, config, prompts, approval, _, _ = fake_matrix(tmp_path)
    path = data / "test_gold.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(gold_numerator=9, gold_denominator=1, gold_answer="9")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal(data)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "test_gold.jsonl", "Replacement gold read before checking immutable public seal SHA"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="SHA256SUMS"):
        analysis.run_analysis(root, path, tmp_path / "analysis", "mock", config=config, prompts=prompts, approval=approval,
                              pilot_report=data / "pilot_go.json")


def test_full_matrix_partial_pool_retains_every_denominator(tmp_path):
    root, data, config, prompts, approval, snapshots, verifier = fake_matrix(tmp_path)
    snapshot = snapshots["JFINAL"]
    snapshot["predictions"][0]["status"] = snapshot["metrics"][0]["status"] = "timeout"
    snapshot["predictions"][0].pop("winner_branch")
    snapshot["metrics"][0].pop("winner_branch")
    snapshot["proposals"][0]["status"] = "partial"
    snapshot["proposals"][0]["candidates"] = snapshot["proposals"][0]["candidates"][:1]
    save_snapshot(root / "JFINAL", snapshot, verifier)
    result = analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                                   config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")
    assert result["primary"]["quality"]["acc_diff_pp"] == pytest.approx(50 - 100 / 1500)
    assert result["parser_audit"]["n_predictions"] == 6000
    assert result["jfinal"]["pool_status_counts"] == {"partial": 1, "complete": 1499}
    assert result["jfinal"]["planned_candidate_slots"] == 6000 and result["jfinal"]["logged_candidate_slots"] == 5997
    assert result["jfinal"]["n_cases"] == 1500
    controls = [r for r in result["table"] if r["condition"] not in analysis.ORDER]
    assert all(r["n_cases"] == 1500 and r["incorrect_cases"] == 1 for r in controls)


def test_lat_plan_labels_are_checked_against_sealed_gold_not_only_counts(tmp_path):
    root, data, config, prompts, approval, snapshots, verifier = fake_matrix(tmp_path)
    snapshot = snapshots["LATENCY"]
    # Swap labels, maintaining every domain/difficulty count and frozen ID.
    items = snapshot["latency_plan"]["items"]
    items[0]["difficulty"], items[14]["difficulty"] = items[14]["difficulty"], items[0]["difficulty"]
    save_snapshot(root / "LATENCY", snapshot, verifier)
    with pytest.raises(ValueError, match="frozen latency plan"):
        analysis.run_analysis(root, data / "test_gold.jsonl", tmp_path / "analysis", "mock",
                              config=config, prompts=prompts, approval=approval, pilot_report=data / "pilot_go.json")
