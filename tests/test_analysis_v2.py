"""CPU-only fixed contrasts, provenance gates and independent parser audit."""

import json
from fractions import Fraction

import pandas as pd
import pytest

from jevlab import analysis
from jevlab.common import config_hash, sha256_file
from jevlab.evaluate import grade_prediction


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, n=10, version="2", code_pin=None, run_version="0.2.0"):
    data = tmp_path / "data" / "v2"
    data.mkdir(parents=True)
    gold = [{"id": f"id-{i:03}", "domain": f"domain-{i % 5}", "difficulty": "easy",
             "gold_numerator": 1, "gold_denominator": 2, "gold_answer": "1/2", "reviewed": False}
            for i in range(n)]
    jsonl(data / "test_gold.jsonl", gold)
    jsonl(data / "test_inputs.jsonl", [{"id": g["id"], "problem": "Synthetic"} for g in gold])
    (data / "SHA256SUMS").write_text("".join(f"{sha256_file(data / name)}  {name}\n" for name in
                                            ("test_gold.jsonl", "test_inputs.jsonl")))
    frozen = {"models": {r: {"repo": "synthetic-" + r, "revision": "fixed"} for r in "GJB"},
              "implementation_profile": "1COPY-G4-VLLM-GRAPH", "jevk5_runtime": {"commit": "fixed"},
               "protocol_version": version}
    if code_pin is not None:
        frozen["code_version"] = code_pin
        frozen["runtime_defaults"] = {"LATENCY_SWAP_MODE": "reload"}
    config = tmp_path / "config" / ("experiment_v2_reload.json" if code_pin else "experiment_v2.json")
    config.parent.mkdir()
    config.write_text(json.dumps(frozen))
    root = tmp_path / "results"
    for cond in analysis.ORDER:
        cfg = {"protocol_version": version, "models": frozen["models"], "profile": frozen["implementation_profile"],
               "jevk5_runtime": frozen["jevk5_runtime"], "code_version": run_version,
               "experiment_config_sha256": sha256_file(config), "prompt_sha256": {"generator": "fixed"},
               "data_sha256": {"test_inputs.jsonl": sha256_file(data / "test_inputs.jsonl")},
               "params": {"CONDITION": cond, "SPLIT": "test", "RUN_TAG": "synthetic", "SEEDS": "17",
                          "CONFIG_FILE": "config/" + config.name}}
        if code_pin is not None:
            cfg["params"]["LATENCY_SWAP_MODE"] = "reload"
        digest = config_hash(cfg)
        name = f"{cond}_synthetic_{digest[:8]}"
        directory = root / cond / name
        directory.mkdir(parents=True)
        preds, metrics, candidates, decisions = [], [], [], []
        for i, g in enumerate(gold):
            correct = i < (n if cond == "JFINAL" else n // 2)
            text = "FINAL: 2/4\n" if correct else "FINAL: 9\n"
            common = {"problem_id": g["id"], "seed": 17, "condition": cond,
                      "config_hash": digest, "profile": cfg["profile"], "t_start_utc": "start", "t_end_utc": "end"}
            preds.append({**common, "status": "final", "public_output": text})
            metrics.append({**common, "T_total": 2.0, "accepted_tokens": 5, "candidate_tokens_total": 20,
                            "selector_input_tokens": 25, "T_selector": .2, "processed_prompt_tokens": 100,
                            "nvml_peak_used_bytes": 2**30})
            if cond == "JFINAL":
                candidates.extend({**common, "branch": b, "text": text if b == 0 else "FINAL: 9\n",
                                   "round": 0, "chosen": b == 0} for b in range(4))
                decisions.append({**common, "input_tokens": 25, "probabilities": [.7, .1, .1, .1],
                                  "round": 0,
                                  "winner_position": 0, "winner_branch": 0, "permutation": [0, 1, 2, 3],
                                  "n_unique_candidates": 2, "status": "ok"})
        ledgers = {"predictions": preds, "metrics": metrics, "candidates": candidates,
                   "decisions": decisions, "failures": []}
        for name_ledger, rows in ledgers.items():
            jsonl(directory / (name_ledger + ".jsonl"), rows)
        man = {"config": cfg, "config_hash": digest, "condition": cond, "run_name": name,
               "condition_label": "Description for " + cond, "finished_utc": "end", "stages": {"main": {"ok": True}},
               "preflight": {name: {"blocking": True, "ok": True} for name in analysis.BLOCKING_V2 |
                             ({"selector_native_reference", "selector_equivalence"} if cond in analysis.PRIMARY else set())},
               "model_lock": {r: {"ok": True} for r in "GJB"},
               "counts": {key: len(rows) for key, rows in ledgers.items()}}
        (directory / "manifest.json").write_text(json.dumps(man))
        (directory / "progress.json").write_text(json.dumps({"done": n, "total": n}))
    return root, data / "test_gold.jsonl", tmp_path / "analysis"


def ledger(root, cond, filename="predictions.jsonl"):
    return next((root / cond).rglob(filename))


def test_notebook_existing_call_autodetects_v2_and_fixed_family(tmp_path):
    root, gold, out = fixture(tmp_path, n=100)
    result = analysis.run_analysis(str(root), str(gold), str(out), "synthetic", 50, None)
    saved = json.loads((out / "analysis.json").read_text())
    assert saved["meta"]["protocol_version"] == "2"
    assert (result["primary"]["method"], result["primary"]["baseline"]) == ("JFINAL", "B13")
    assert result["primary"]["more_accurate"] and result["primary"]["practical_point_gt5pp"]
    assert [(r["method"], r["baseline"]) for r in result["secondary"]] == analysis.SECONDARY_V2
    assert all(r["simultaneous_ci_level"] == .99 for r in result["secondary"])
    assert "acc_diff_ci99_bonferroni" not in result["primary"]
    assert not any("speedup" in k for r in result["comparisons"] for k in r)
    assert saved["parser_audit"]["n_predictions"] == 600
    assert saved["parser_audit"]["n_disagreements"] == 0
    assert result["jfinal"]["oracle_capture_ratio"] == 1
    assert "NOT Holm rank-dependent" in saved["secondary_interval_note"]
    assert "reviewed=false" in result["report"] and "13.159554560B" in result["report"]
    assert "12.413584128B" in result["report"] and "secondary descriptive" in result["report"]
    for filename in ("graded_runs.csv", "final_table.csv", "secondary_comparisons.csv"):
        assert "Description for" in (out / filename).read_text()
    assert len(saved["figures"]) == 3


def test_exploratory_last_number_never_changes_strict_h1(tmp_path):
    root, gold, out = fixture(tmp_path)
    path = ledger(root, "JFINAL")
    rows = analysis.read_jsonl(path)
    for row in rows:
        row.update(status="eos_invalid", public_output="The result is 0.5")
    jsonl(path, rows)
    result = analysis.run_analysis(str(root), str(gold), str(out), n_boot=20)
    exploratory = pd.read_csv(out / "exploratory_last_number.csv")
    assert exploratory[exploratory.condition == "JFINAL"].correct_exploratory.all()
    assert not result["df"].loc[result["df"].condition == "JFINAL", "correct"].any()
    assert result["primary"]["acc_diff_pp"] == -50
    assert not result["primary"]["more_accurate"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_seed", "row_hash", "models", "data_hash", "pilot",
                                  "history", "no_finish", "preflight", "abort", "metric_duplicate", "gold_duplicate",
                                  "candidate_duplicate", "partial_progress", "duplicate_run", "config_changed"])
def test_fail_closed_before_hypothesis_analysis(tmp_path, fault):
    root, gold, out = fixture(tmp_path)
    path = ledger(root, "JFINAL")
    rows = analysis.read_jsonl(path)
    manifest_path = path.with_name("manifest.json")
    man = json.loads(manifest_path.read_text())
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "wrong_seed":
        rows[0]["seed"] = 29
    elif fault == "row_hash":
        rows[0]["config_hash"] = "bad"
    elif fault == "models":
        man["config"]["models"]["G"]["revision"] = "bad"
    elif fault == "data_hash":
        man["config"]["data_sha256"]["test_inputs.jsonl"] = "bad"
    elif fault == "pilot":
        man["config"]["params"]["SPLIT"] = "pilot"
    elif fault == "history":
        man["config"]["protocol_version"] = "3"
    elif fault == "no_finish":
        del man["finished_utc"]
    elif fault == "preflight":
        man["preflight"]["parser_selftest"]["ok"] = False
    elif fault == "abort":
        rows[0]["status"] = "run_aborted"
    elif fault == "metric_duplicate":
        metric_path = path.with_name("metrics.jsonl")
        metrics = analysis.read_jsonl(metric_path)
        metrics[-1] = metrics[0]
        jsonl(metric_path, metrics)
    elif fault == "gold_duplicate":
        gs = analysis.read_jsonl(gold)
        gs[-1] = gs[0]
        jsonl(gold, gs)
    elif fault == "candidate_duplicate":
        cp = path.with_name("candidates.jsonl")
        cs = analysis.read_jsonl(cp)
        cs[3] = cs[0]
        jsonl(cp, cs)
    elif fault == "partial_progress":
        path.with_name("progress.json").write_text(json.dumps({"done": 9, "total": 10}))
    elif fault == "duplicate_run":
        clone = root / "duplicate" / path.parent.name
        clone.mkdir(parents=True)
        for original in path.parent.iterdir():
            (clone / original.name).write_bytes(original.read_bytes())
    elif fault == "config_changed":
        config = tmp_path / "config" / "experiment_v2.json"
        config.write_text(config.read_text() + "\n")
    jsonl(path, rows)
    manifest_path.write_text(json.dumps(man))
    with pytest.raises(ValueError):
        analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    assert not (out / "analysis.json").exists()


@pytest.mark.parametrize("text,status", [("FINAL: 2/4\n", "final"), ("FINAL: .5", "final"),
    ("FINAL: +1 / +2\n", "final"), ("FINAL: 1/-2", "final"), ("FINAL: \u22120.5", "final"),
    ("FINAL: 1/0", "final"), ("FINAL: 2+2", "final"), ("FINAL: 0.5 kg", "final"),
    ("FINAL: 1\n FINAL: 0.5", "final"), ("final: .5", "final"), ("Answer: 0.5", "eos_invalid"),
    ("FINAL: 0.5", "timeout"), ("FINAL: 0.5", "truncated"), ("FINAL: 1 \t/\t 2", "final")])
def test_independent_parser_matches_canonical_fields_and_rationals(text, status):
    g = {"id": "id", "domain": "domain", "difficulty": "easy", "gold_numerator": 1,
         "gold_denominator": 2, "gold_answer": "1/2"}
    pred = {"problem_id": "id", "seed": 17, "condition": "JFINAL", "status": status, "public_output": text}
    canonical = grade_prediction(pred, g)
    independent = analysis.independent_grade_v2(pred, g)
    assert independent == {k: canonical[k] for k in independent}


def test_audit_is_independent_and_reports_disagreements(tmp_path, monkeypatch):
    root, gold, out = fixture(tmp_path)
    original = analysis.grade_prediction

    def broken(pred, g):
        return {**original(pred, g), "answer_normalized": "broken"}

    monkeypatch.setattr(analysis, "grade_prediction", broken)
    with pytest.raises(ValueError, match="Independent parser disagrees"):
        analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    audit = json.loads((out / "parser_audit.json").read_text())
    assert audit["n_predictions"] == audit["n_disagreements"] == 60
    assert all(r["disagreements"] == ["answer_normalized"] for r in audit["rows"])
    assert not (out / "analysis.json").exists()


def test_holm_stepdown_adjusted_pvalues_ties_and_monotonicity():
    assert analysis.holm_adjusted_pvalues([.04, .001, .01, .02, .9]) == pytest.approx([.08, .005, .04, .06, .9])
    assert analysis.holm_adjusted_pvalues([.01, .01, .01, .01, .01]) == pytest.approx([.05] * 5)


@pytest.mark.parametrize("code_pin,run_version,accepted", [
    (None, "0.2.0", True), (None, "0.2.1", False), (None, "0.2.2", False),
    ("0.2.1", "0.2.1", True), ("0.2.1", "0.2.0", False), ("0.2.1", "0.2.2", False)])
def test_analysis_requires_exact_frozen_code_pin(tmp_path, code_pin, run_version, accepted):
    root, gold, _ = fixture(tmp_path, code_pin=code_pin, run_version=run_version)
    runs = analysis.load_runs(str(root), str(gold))
    if accepted:
        assert analysis.validate_v2(runs, str(gold))["n_problems"] == 10
    else:
        with pytest.raises(ValueError, match="Frozen models/config/runtime mismatch"):
            analysis.validate_v2(runs, str(gold))


def test_reload_version_keeps_all_six_scores_and_fixed_statistics_identical(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old_root, old_gold, old_out = fixture(old)
    new_root, new_gold, new_out = fixture(new, code_pin="0.2.1", run_version="0.2.1")
    before = analysis.run_analysis(str(old_root), str(old_gold), str(old_out), n_boot=20)
    after = analysis.run_analysis(str(new_root), str(new_gold), str(new_out), n_boot=20)
    assert before["primary"] == after["primary"]
    assert before["secondary"] == after["secondary"]
    pd.testing.assert_frame_equal(before["table"], after["table"])
    assert before["df"].correct.tolist() == after["df"].correct.tolist()
    assert set(after["table"].condition) == set(analysis.ORDER)
    assert json.loads((new_out / "analysis.json").read_text())["meta"]["code_version"] == "0.2.1"


def test_jfinal_timeout_without_auxiliary_logs_stays_in_h1_denominator(tmp_path):
    root, gold, out = fixture(tmp_path)
    path = ledger(root, "JFINAL")
    rows = analysis.read_jsonl(path)
    rows[0].update(status="timeout", public_output="")
    pid = rows[0]["problem_id"]
    jsonl(path, rows)
    manifest_path = path.with_name("manifest.json")
    man = json.loads(manifest_path.read_text())
    for name in ("candidates", "decisions"):
        auxiliary = [r for r in analysis.read_jsonl(path.with_name(name + ".jsonl")) if r["problem_id"] != pid]
        jsonl(path.with_name(name + ".jsonl"), auxiliary)
        man["counts"][name] = len(auxiliary)
    manifest_path.write_text(json.dumps(man))
    result = analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    assert result["primary"]["n_problems"] == 10
    assert result["primary"]["acc_diff_pp"] == 40
    assert result["jfinal"]["n"] == 9
    assert result["jfinal"]["n_cases_without_candidate_logs"] == 1
    assert result["jfinal"]["oracle_capture_ratio"] == 1


def test_failed_rerun_invalidates_previous_success_report(tmp_path, monkeypatch):
    root, gold, out = fixture(tmp_path)
    analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    assert (out / "analysis.json").exists()
    original = analysis.grade_prediction
    monkeypatch.setattr(analysis, "grade_prediction", lambda pred, g: {**original(pred, g), "answer_raw": "bad"})
    with pytest.raises(ValueError, match="Independent parser disagrees"):
        analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    assert not (out / "analysis.json").exists()
    assert not (out / "report.md").exists()
    assert not (out / "primary_comparison.csv").exists()
    assert not (out / "figures" / "accuracy.png").exists()
    assert json.loads((out / "parser_audit.json").read_text())["n_disagreements"] == 60


@pytest.mark.parametrize("code_pin", [None, "0.2.1"])
@pytest.mark.parametrize("fault", [None, "missing", "failed", "duplicate", "provenance", "timers", "scoring", "gpu"])
def test_latency_study_is_only_complete_matching_secondary_descriptive(tmp_path, fault, code_pin):
    root, gold, out = fixture(tmp_path, code_pin=code_pin, run_version=code_pin or "0.2.0")
    cfg = json.loads(ledger(root, "JFINAL", "manifest.json").read_text())["config"]
    directory = root / "latency"
    directory.mkdir()
    latency_config = {**cfg, "params": {**cfg["params"], "CONDITION": "LATENCY", "SPLIT": "dev"}}
    digest = config_hash(latency_config)
    rows = [{"item_id": "dev-1", "condition": cond, "seed": 17, "T_total": 1.5, "status": "truncated",
             "config_hash": digest} for cond in analysis.ORDER]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "provenance":
        latency_config["protocol_version"] = "3"
    elif fault == "timers":
        latency_config["params"]["TIMEOUT_S"] = 601
    elif fault == "scoring":
        latency_config["params"]["MAX_OUTPUT_TOKENS"] = 1025
    if fault in {"timers", "scoring"}:
        digest = config_hash(latency_config)
        for row in rows:
            row["config_hash"] = digest
    man = {"config": latency_config, "config_hash": digest, "finished_utc": "end",
           "environment": {"gpus": [{"name": "NVIDIA A100", "memory.total": 40960 if fault != "gpu" else 81920}]},
           "counts": {"latency_metrics": len(rows)}, "stages": {"main": {"ok": True}}}
    (directory / "manifest.json").write_text(json.dumps(man))
    (directory / "progress.json").write_text(json.dumps({"done": 6, "total": 6}))
    (directory / "latency_plan.json").write_text(json.dumps({"items": [{"item_id": "dev-1"}]}))
    jsonl(directory / "latency_metrics.jsonl", rows)
    if code_pin:
        jsonl(directory / "swaps.jsonl", [{"to": "B", "mode": "reload", "seconds": 1000.0}])
    if fault == "failed":
        jsonl(directory / "failures.jsonl", [{"condition": "JFINAL", "error": "exception"}])
    if fault:
        with pytest.raises(ValueError):
            analysis.latency_descriptive_v2(str(root), cfg, {c: "Description " + c for c in analysis.ORDER})
    else:
        result = analysis.latency_descriptive_v2(str(root), cfg, {c: "Description " + c for c in analysis.ORDER})
        assert result["scope"] == "same-GPU secondary descriptive"
        assert result["hardware"] == "same A100 40GB"
        assert result["swap_mode"] == ("reload" if code_pin else "sleep")
        assert "outside" in result["swap_timing"]
        assert "confirmatory" not in json.dumps(result)
        if code_pin:
            assert "reload swaps outside the chronometer" in result["swap_description"]
        assert result["n"] == 6
        assert all(r["latency_p50_s"] == 1.5 for r in result["by_condition"].values())
        assert all(r["condition_label"].startswith("Description") for r in result["by_condition"].values())
        assert all(not any("ci" in k or "confirmatory" in k for k in r) for r in result["by_condition"].values())


def test_exact_paired_mcnemar_not_unpaired_or_one_sided():
    rows = [{"problem_id": str(i), "condition": cond, "domain": "d", "correct": correct, "T_total": 1}
            for cond in ("J64", "B13") for i, correct in enumerate(([1] * 9 + [0]) if cond == "J64" else [0] * 9 + [1])]
    result = analysis.paired_accuracy_v2(pd.DataFrame(rows), "J64", "B13", 100)
    assert (result["wins"], result["losses"]) == (9, 1)
    assert result["mcnemar_p_two_sided_exact"] == float(Fraction(22, 1024))
    assert result["acc_diff_ci99_bonferroni"][0] <= result["acc_diff_ci95"][0]
    assert result["acc_diff_ci99_bonferroni"][1] >= result["acc_diff_ci95"][1]
    identical = pd.DataFrame(rows)
    identical["correct"] = True
    assert analysis.paired_accuracy_v2(identical, "J64", "B13", 10)["mcnemar_p_two_sided_exact"] == 1


def test_v1_existing_outputs_and_comparison_family_preserved(tmp_path):
    root, gold, out = fixture(tmp_path, version="3")
    result = analysis.run_analysis(str(root), str(gold), str(out), n_boot=10)
    assert "primary" not in result and "secondary" not in result
    assert [r["method"] for r in result["comparisons"]] == ["J64", "JSTEP", "JFINAL", "G_SINGLE", "B13_GREEDY"]
    assert all("acc_diff_ci99_1667" in r and "speedup_ci99_1667" in r for r in result["comparisons"])
    assert "diff_ci99_1667" in result["table"] and "condition_label" not in result["table"]
    assert not (out / "parser_audit.json").exists()
