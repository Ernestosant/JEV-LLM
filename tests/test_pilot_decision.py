"""Actual synthetic v2 final ZIPs and verifier evidence; CPU, offline, no models."""

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_decision", ROOT / "tools/pilot_decision.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def lines(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def archive(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in members.items():
            z.writestr("run/" + name, value)


@pytest.fixture
def fixture(tmp_path, request):
    data = tmp_path / "data"
    data.mkdir()
    inputs, gold = data / "pilot_inputs.jsonl", data / "pilot_gold.jsonl"
    ids = [f"jev-v2-p-{i:03}" for i in range(1, 41)]
    inputs.write_text(lines([{"id": pid, "problem": "Find one half", "language": "en"} for pid in ids]))
    gold.write_text(lines([{"id": pid, "gold_numerator": 1, "gold_denominator": 2, "gold_answer": "1/2",
                           "domain": "arithmetic", "difficulty": "easy"} for pid in ids]))
    (data / "SHA256SUMS").write_text("".join(f"{pilot.sha256(p.read_bytes())}  {p.name}\n" for p in (inputs, gold)))
    # Use the real frozen budgets, models, revisions and canonical labels.
    config = ROOT / ("config/experiment_v2_reload.json" if getattr(request, "param", None) == "reload"
                     else "config/experiment_v2.json")
    frozen = json.loads(config.read_text())
    prompts = ROOT / "prompts"
    results = tmp_path / "results"
    artifacts = {}
    for condition in pilot.CONDITIONS:
        directory = results / condition / "artifacts"
        directory.mkdir(parents=True)
        # Matches runner.NON_SEMANTIC: actual config.params never includes N_PROBLEMS.
        params = {"CONDITION": condition, "RUN_TAG": "pilot_v2", "SPLIT": "pilot", "SEEDS": "17",
                  "KV_CACHE_GB_G": 3.0, "ENFORCE_EAGER": False}
        if "LATENCY_SWAP_MODE" in frozen["runtime_defaults"]:
            params["LATENCY_SWAP_MODE"] = frozen["runtime_defaults"]["LATENCY_SWAP_MODE"]
        limit_params = {"n_candidates": "N_CANDIDATES", "max_output_tokens": "MAX_OUTPUT_TOKENS",
                        "hybrid_candidate_budget": "CANDIDATE_BUDGET", "j64_block": "J64_BLOCK",
                        "jstep_step": "JSTEP_STEP", "jstep_max_rounds": "JSTEP_MAX_ROUNDS",
                        "gb_context": "GB_CONTEXT", "j_max_input": "J_MAX_INPUT",
                        "max_prompt_tokens": "MAX_PROMPT_TOKENS", "timeout_s": "TIMEOUT_S"}
        params.update({parameter: frozen["limits"][name] for name, parameter in limit_params.items()})
        params.update({name.upper(): frozen["sampling"][name] for name in
                       ("temperature", "top_p", "top_k", "repetition_penalty")})
        cfg = {"params": params, "models": frozen["models"], "profile": frozen["implementation_profile"],
               "jevk5_runtime": frozen["jevk5_runtime"], "protocol_version": "2", "code_version": frozen.get("code_version", "0.2.0"),
               "experiment_config_sha256": pilot.sha256(config.read_bytes()),
               "data_sha256": {inputs.name: pilot.sha256(inputs.read_bytes())},
               "prompt_sha256": {name: pilot.sha256((prompts / (name + ".txt")).read_text(encoding="utf-8").strip("\n").encode())
                                 for name in ("generador", "criterio_paso", "criterio_final")}}
        digest = pilot.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        run_name = f"{condition}_pilot_v2_{digest[:8]}"
        predictions, metrics, candidates, decisions = [], [], [], []
        for pid in reversed(ids):
            common = {"problem_id": pid, "condition": condition, "condition_label": frozen["condition_labels"][condition],
                      "seed": 17, "config_hash": digest, "profile": cfg["profile"], "run_name": run_name,
                      "resume_key": pilot.sha256(f"{digest}|{pid}|17|{condition}|{cfg['profile']}".encode())}
            predictions.append({**common, "status": "final", "public_output": "Reasoning\nFINAL: 0.5\n",
                                "t_start_utc": "start", "t_end_utc": "end"})
            metrics.append({**common, "T_total": 4, "T_proposals": 3, "T_selector": 1,
                            "nvml_peak_used_bytes": 20 * 2**30, "rss_gib": 1})
            if condition == "JFINAL":
                candidates.extend({**common, "round": 0, "branch": b, "chosen": b == 2,
                                   "text": "FINAL: 1/2" if b == 2 else "FINAL: 1"} for b in range(4))
                decisions.append({**common, "round": 0, "status": "ok", "permutation": [3, 1, 2, 0],
                                  "winner_position": 2, "winner_branch": 2})
        ledgers = {"predictions": predictions, "metrics": metrics, "candidates": candidates,
                   "decisions": decisions, "failures": []}
        manifest = {"condition": condition, "condition_label": frozen["condition_labels"][condition],
                    "run_name": run_name, "config": cfg, "config_hash": digest, "finished_utc": "finished",
                    "counts": {name: len(rows) for name, rows in ledgers.items()},
                     "stages": {"run": {"ok": True}}, "preflight": {name: {"blocking": True, "ok": True} for name in
                         pilot.BLOCKING | ({"selector_native_reference", "selector_equivalence"} if condition == "JFINAL" else set())},
                    "model_lock": {role: {"ok": True} for role in ("G", "J", "B")},
                    "environment": {"gpus": [{"name": "NVIDIA A100", "memory.total": 40960}]}}
        members = {name + ".jsonl": lines(rows) for name, rows in ledgers.items()}
        members.update({"manifest.json": json.dumps(manifest), "progress.json": json.dumps({"done": 40, "total": 40})})
        path = directory / (run_name + "_final.zip")
        archive(path, members)
        prefix = {"JFINAL": "06", "B13": "02", "G_SINGLE": "01"}[condition]
        nb = directory / f"{prefix}_fake.out.timestamp.ipynb"
        nb.write_text(json.dumps({"metadata": {"papermill": {"end_time": "end", "exception": False}},
                                 "cells": [{"cell_type": "code", "source": ["pass"], "outputs": [],
                                            "metadata": {"papermill": {"status": "completed"}}}]}))
        evidence_path = directory.parent / "verification.json"
        artifacts[condition] = (path, members, nb, evidence_path)
        evidence_path.write_text(json.dumps(pilot.verify_run(directory, conditions=[condition], inputs=inputs,
                                 config=config, expected_count=40, split="pilot", run_tag="pilot_v2", prompts=prompts)))
    return results, gold, artifacts


def change(fixture, condition, ledger, edit, reverify=False):
    results, gold, artifacts = fixture
    path, members, nb, evidence = artifacts[condition]
    rows = json.loads(members[ledger]) if ledger.endswith(".json") else pilot.jsonl(members[ledger])
    edit(rows)
    members[ledger] = json.dumps(rows) if ledger.endswith(".json") else lines(rows)
    archive(path, members)
    if reverify:
        evidence.write_text(json.dumps(pilot.verify_run(path.parent, conditions=[condition],
                            inputs=gold.with_name("pilot_inputs.jsonl"), config=ROOT / "config/experiment_v2.json",
                            expected_count=40, split="pilot", run_tag="pilot_v2")))


def test_go_and_diagnostics(fixture):
    results, gold, _ = fixture
    report = pilot.decide(results, gold)
    assert report["go"], report["errors"]
    assert report["decision"] == "go"
    j = report["conditions"]["JFINAL"]
    assert j["accuracy"] == j["valid_format_rate"] == j["oracle"]["capture_ratio"] == 1
    assert j["oracle"]["oracle_at_4"] == 1
    assert j["metrics"]["T_total"]["sum"] == 160
    assert j["metrics"]["nvml_peak_used_bytes"]["max"] == 20 * 2**30
    assert j["grades"][0]["answer_normalized"] == "1/2"
    assert report["aggregate"]["timeout_infra_rate"] == 0


@pytest.mark.parametrize("fixture", ["reload"], indirect=True)
def test_reload_pilot_real_verifier_with_explicit_config(fixture, tmp_path):
    config = ROOT / "config/experiment_v2_reload.json"
    report = pilot.decide(*fixture[:2], config=config)
    assert report["go"], report["errors"]
    assert report["frozen"]["code_version"] == "0.2.1"
    assert all(e["verified"] and e["correct"] == 40 and e["metrics"]["T_total"]["sum"] == 160
               for e in report["conditions"].values())
    assert report["jfinal_at_least_b13"]
    assert not pilot.decide(*fixture[:2])["go"]  # Default old config must not accept reload finals.
    output = tmp_path / "reload_decision.json"
    assert pilot.main(["--results", str(fixture[0]), "--gold", str(fixture[1]), "--config", str(config),
                       "--output", str(output)]) == 0


@pytest.mark.parametrize("code_pin,run_version,accepted", [
    (None, "0.2.0", True), (None, "0.2.1", False), (None, "0.2.2", False),
    ("0.2.1", "0.2.1", True), ("0.2.1", "0.2.0", False), ("0.2.1", "0.2.2", False)])
def test_pilot_owns_exact_frozen_pin_even_if_verifier_accepts(fixture, tmp_path, monkeypatch,
                                                            code_pin, run_version, accepted):
    """Isolate gate validation from the separately owned verifier's version policy."""
    results, gold, artifacts = fixture
    frozen = json.loads((ROOT / "config/experiment_v2.json").read_text())
    if code_pin is not None:
        frozen["code_version"] = code_pin
        frozen["runtime_defaults"]["LATENCY_SWAP_MODE"] = "reload"
    else:
        frozen.pop("code_version", None)
    config = tmp_path / ("experiment_v2_reload.json" if code_pin else "experiment_v2.json")
    config.write_text(json.dumps(frozen))

    def verified(directory, *, conditions, code_version, **kwargs):
        assert code_version == (code_pin or "0.2.0")
        condition = conditions[0]
        path = next(directory.glob("*_final.zip"))
        snapshot = pilot.read_archive(path)
        notebook = next(directory.glob("*.ipynb"))
        return {"ok": True, "runs": {condition: {"ok": True, "condition": condition, "records": 40,
                "archive_sha256": snapshot["archive_sha256"], "notebook_sha256": pilot.sha256(notebook.read_bytes()),
                "config_hash": snapshot["manifest"]["config_hash"]}}}

    monkeypatch.setattr(pilot, "verify_run", verified)
    for condition, (old_path, members, nb, evidence) in artifacts.items():
        manifest = json.loads(members["manifest.json"])
        cfg = manifest["config"]
        cfg.update(code_version=run_version, experiment_config_sha256=pilot.sha256(config.read_bytes()))
        if code_pin:
            cfg["params"]["LATENCY_SWAP_MODE"] = "reload"
        digest = pilot.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        run_name = f"{condition}_pilot_v2_{digest[:8]}"
        manifest.update(config_hash=digest, run_name=run_name)
        members["manifest.json"] = json.dumps(manifest)
        for ledger in ("predictions", "metrics", "candidates", "decisions"):
            rows = pilot.jsonl(members[ledger + ".jsonl"])
            for row in rows:
                row.update(config_hash=digest, run_name=run_name, resume_key=pilot.sha256(
                    f"{digest}|{row['problem_id']}|17|{condition}|{cfg['profile']}".encode()))
            members[ledger + ".jsonl"] = lines(rows)
        path = old_path.with_name(run_name + "_final.zip")
        archive(path, members)
        if old_path != path:
            old_path.unlink()
        evidence.write_text(json.dumps(verified(path.parent, conditions=[condition], code_version=code_pin or "0.2.0")))
    report = pilot.decide(results, gold, config=config)
    assert report["go"] is accepted, report["errors"]
    if not accepted:
        assert all(not r["graded"] for r in report["conditions"].values())
        assert all("Frozen code version mismatch" in error for error in report["errors"])
    else:
        assert all(r["correct"] == 40 and r["metrics"]["T_total"]["sum"] == 160 for r in report["conditions"].values())
        output = tmp_path / "decision.json"
        assert pilot.main(["--results", str(results), "--gold", str(gold), "--config", str(config),
                           "--output", str(output)]) == 0


@pytest.mark.parametrize("fault", ["missing", "nonblocking"])
def test_all_required_blocking_checks_must_be_present_and_pass(fixture, fault):
    def edit(manifest):
        if fault == "missing":
            manifest["preflight"].pop("selector_equivalence")
        else:
            manifest["preflight"]["selector_equivalence"]["blocking"] = False
    change(fixture, "JFINAL", "manifest.json", edit, True)
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert "blocking preflight" in " ".join(report["errors"])


def test_markdown_output_cannot_overwrite_json(tmp_path):
    with pytest.raises(ValueError, match="must use .json"):
        pilot.write_report({"decision": "go"}, tmp_path / "decision.md")


def test_accuracy_no_go(fixture):
    change(fixture, "JFINAL", "predictions.jsonl", lambda rows: rows[0].update(public_output="FINAL: 2"), True)
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert report["conditions"]["JFINAL"]["verified"]
    assert report["conditions"]["JFINAL"]["oracle"]["capture_ratio"] == 39 / 40
    assert "below B13" in " ".join(report["errors"])


@pytest.mark.parametrize("count,go", [(1, True), (2, False), (3, False)])
def test_timeout_strict_per_condition_not_only_aggregate(fixture, count, go):
    change(fixture, "G_SINGLE", "predictions.jsonl",
           lambda rows: [r.update(status="timeout") for r in rows[:count]], True)
    report = pilot.decide(*fixture[:2])
    assert report["go"] is go, report["errors"]
    assert report["conditions"]["G_SINGLE"]["rates"]["timeout_rate"] == count / 40
    assert report["aggregate"]["timeout_rate"] == count / 120


@pytest.mark.parametrize("fault", ["archive_hash", "notebook_hash", "evidence_failed", "missing_evidence",
                                  "notebook_error", "notebook_retry", "archive_changed"])
def test_bad_provenance(fixture, fault):
    _, _, artifacts = fixture
    path, members, nb, evidence = artifacts["B13"]
    record = json.loads(evidence.read_text())
    if fault == "archive_hash":
        record["runs"]["B13"]["archive_sha256"] = "bad"
    elif fault == "notebook_hash":
        record["runs"]["B13"]["notebook_sha256"] = "bad"
    elif fault == "evidence_failed":
        record["ok"] = False
    elif fault == "missing_evidence":
        evidence.unlink()
    elif fault == "notebook_error":
        obj = json.loads(nb.read_text()); obj["cells"][0]["outputs"] = [{"output_type": "error"}]
        nb.write_text(json.dumps(obj))
    elif fault == "notebook_retry":
        shutil.copyfile(nb, nb.with_name("02_retry.out.timestamp.ipynb"))
    else:
        change(fixture, "B13", "predictions.jsonl", lambda rows: rows[0].update(public_output="FINAL: 3"))
    if fault in ("archive_hash", "notebook_hash", "evidence_failed"):
        evidence.write_text(json.dumps(record))
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert not report["conditions"]["B13"]["graded"]


@pytest.mark.parametrize("fault", ["missing_case", "duplicate_case", "wrong_seed", "test_id", "preflight", "no_preflight",
                                  "aborted", "infra", "failure_ledger", "label", "broken_zip", "missing_final"])
def test_incomplete_or_failed_final(fixture, fault):
    path, _, _, _ = fixture[2]["G_SINGLE"]
    if fault == "missing_case":
        change(fixture, "G_SINGLE", "predictions.jsonl", lambda rows: rows.pop())
    elif fault == "duplicate_case":
        change(fixture, "G_SINGLE", "predictions.jsonl", lambda rows: rows.__setitem__(0, rows[1]))
    elif fault == "wrong_seed":
        change(fixture, "G_SINGLE", "predictions.jsonl", lambda rows: rows[0].update(seed=29))
    elif fault == "test_id":
        change(fixture, "G_SINGLE", "predictions.jsonl", lambda rows: rows[0].update(problem_id="jev-v2-t-001"))
    elif fault in ("preflight", "no_preflight"):
        change(fixture, "G_SINGLE", "manifest.json", lambda m: m.update(preflight={} if fault == "no_preflight"
               else {"budget": {"blocking": True, "ok": False}}))
    elif fault in ("aborted", "infra"):
        change(fixture, "G_SINGLE", "predictions.jsonl", lambda rows: rows[0].update(
               status="run_aborted" if fault == "aborted" else "infrastructure"))
    elif fault == "failure_ledger":
        change(fixture, "G_SINGLE", "failures.jsonl", lambda rows: rows.append(
               {"kind": "infrastructure", "problem_id": "jev-v2-p-040", "seed": 17}))
    elif fault == "label":
        change(fixture, "G_SINGLE", "manifest.json", lambda m: m.update(condition_label="alias"))
    elif fault == "broken_zip":
        path.write_bytes(b"incomplete zip")
    else:
        path.unlink()
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert not report["conditions"]["G_SINGLE"]["graded"]
    if fault in ("infra", "aborted", "failure_ledger"):
        assert report["conditions"]["G_SINGLE"]["rates"]["infra_rate"] == 1 / 40


def test_duplicate_final_different_tag_is_not_selected_by_quality(fixture):
    path = fixture[2]["JFINAL"][0]
    shutil.copyfile(path, path.with_name("JFINAL_retry_second_final.zip"))
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert "found 2" in " ".join(report["errors"])


def test_ignore_checkpoint_and_loose_ledgers(fixture):
    results, gold, artifacts = fixture
    path = artifacts["JFINAL"][0]
    path.with_name("JFINAL_pilot_checkpoint.zip").write_bytes(b"bad checkpoint")
    checkpoints = path.parent.parent / "checkpoints"; checkpoints.mkdir()
    shutil.copyfile(path, checkpoints / path.name)
    (results / "predictions.jsonl").write_text("not grading input")
    assert pilot.decide(results, gold)["go"]
    path.unlink()
    assert not pilot.decide(results, gold)["go"]


@pytest.mark.parametrize("text,status,normalized", [("FINAL: +2/4", "ok", "1/2"), ("FINAL: .5", "ok", "1/2"),
    ("FINAL: 1/0", "invalid_format", None), ("FINAL: 0.5 meters", "invalid_format", None),
    ("FINAL: 1e-1", "invalid_format", None), ("FINAL: \\boxed{0.5}", "invalid_format", None),
    ("FINAL: 0.5\nFINAL: 0.5", "multiple_final", None), ("Answer is 0.5", "no_final", None)])
def test_canonical_parser_edges(fixture, text, status, normalized):
    for condition in pilot.CONDITIONS:
        change(fixture, condition, "predictions.jsonl", lambda rows: rows[0].update(public_output=text), True)
    report = pilot.decide(*fixture[:2])
    assert report["go"], report["errors"]
    grade = report["conditions"]["JFINAL"]["grades"][0]
    assert grade["format_status"] == status
    assert grade["answer_normalized"] == normalized
    assert grade["correct"] is (status == "ok")


@pytest.mark.parametrize("fault", ["gold_changed", "missing_seal", "inputs_changed", "duplicate_gold", "gold_test_ids"])
def test_frozen_gold_input_fail_closed(fixture, fault):
    _, gold, _ = fixture
    if fault == "missing_seal":
        (gold.parent / "SHA256SUMS").unlink()
    elif fault == "inputs_changed":
        inputs = gold.with_name("pilot_inputs.jsonl"); inputs.write_text(inputs.read_text() + "\n")
    else:
        rows = pilot.jsonl(gold.read_text())
        if fault == "gold_changed":
            rows[0]["gold_numerator"] = 9
        elif fault == "duplicate_gold":
            rows[0] = rows[1]
        else:
            rows[0]["id"] = "jev-v2-t-001"
        gold.write_text(lines(rows))
        if fault != "gold_changed":
            sums = gold.parent / "SHA256SUMS"
            inputs = gold.with_name("pilot_inputs.jsonl")
            sums.write_text("".join(f"{pilot.sha256(p.read_bytes())}  {p.name}\n" for p in (gold, inputs)))
    assert not pilot.decide(*fixture[:2])["go"]


def test_cli_outputs_even_invalid_inputs(tmp_path):
    output = tmp_path / "out" / "decision.json"
    proc = subprocess.run([sys.executable, str(ROOT / "tools/pilot_decision.py"), "--results", str(tmp_path / "missing"),
                           "--gold", str(tmp_path / "pilot_gold.jsonl"), "--output", str(output)],
                          capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    assert json.loads(output.read_text())["decision"] == "no-go"
    assert "NO-GO" in output.with_suffix(".md").read_text()


def test_cli_go(fixture, tmp_path):
    output = tmp_path / "decision.json"
    assert pilot.main(["--results", str(fixture[0]), "--gold", str(fixture[1]), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["go"]
    assert "capture_ratio" in output.with_suffix(".md").read_text()


def test_aggregate_rate_and_all_condition_rates(fixture):
    for condition in pilot.CONDITIONS:
        change(fixture, condition, "predictions.jsonl",
               lambda rows: [r.update(status="timeout") for r in rows[:2]], True)
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert report["aggregate"]["timeout_infra_rate"] == 0.05
    assert report["aggregate"]["timeout_infra_count"] == 6
    assert all(e["rates"]["timeout_rate"] == 0.05 and e["rates"]["infra_rate"] == 0
               for e in report["conditions"].values())
    assert "Aggregate" in " ".join(report["errors"])


@pytest.mark.parametrize("fault", ["protocol", "smoke", "split", "models", "hash", "stage", "unknown_status", "malformed_row"])
def test_config_and_schema_failures(fixture, tmp_path, fault):
    if fault in ("unknown_status", "malformed_row"):
        change(fixture, "B13", "predictions.jsonl", lambda rows: rows[0].update(status="aborted")
               if fault == "unknown_status" else rows.__setitem__(0, "invalid row"), fault == "unknown_status")
    else:
        def edit(m):
            if fault == "protocol":
                m["config"]["protocol_version"] = "v2"
            elif fault == "smoke":
                m["config"]["params"]["N_PROBLEMS"] = 1
            elif fault == "split":
                m["config"]["params"]["SPLIT"] = "test"
            elif fault == "models":
                m["config"]["models"]["G"]["revision"] = "unfrozen"
            elif fault == "hash":
                m["config"]["experiment_config_sha256"] = "bad"
            else:
                m["stages"]["run"]["ok"] = False
        change(fixture, "B13", "manifest.json", edit)
    output = tmp_path / "invalid_decision.json"
    assert pilot.main(["--results", str(fixture[0]), "--gold", str(fixture[1]), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["go"]
    assert not report["conditions"]["B13"]["graded"]
    assert output.with_suffix(".md").exists()


def test_failure_events_are_deduplicated_with_predictions():
    data = {"predictions": [{"problem_id": "a", "seed": 17, "status": "infrastructure"}],
            "failures": [{"problem_id": "a", "seed": 17, "kind": "infrastructure"}] * 2}
    rates = pilot.failure_rates(data, 40)
    assert rates["infra_count"] == rates["timeout_infra_count"] == 1


def test_undefined_oracle_and_absent_metrics_are_not_zero(fixture):
    for condition in pilot.CONDITIONS:
        change(fixture, condition, "predictions.jsonl", lambda rows: [r.update(public_output="FINAL: 1") for r in rows], True)
    change(fixture, "JFINAL", "candidates.jsonl", lambda rows: [r.update(text="FINAL: 1") for r in rows], True)
    change(fixture, "JFINAL", "metrics.jsonl", lambda rows: [r.pop("T_total") for r in rows], True)
    report = pilot.decide(*fixture[:2])
    assert report["go"], report["errors"]
    j = report["conditions"]["JFINAL"]
    assert j["oracle"]["capture_ratio"] is None
    assert j["metrics"]["T_total"]["mean"] is None
    assert j["metrics"]["T_total"]["missing_or_invalid"] == 40


def test_dataset_manifest_must_be_sealed(fixture):
    dataset = fixture[1].parent / "dataset_manifest.json"
    dataset.write_text(json.dumps({"dataset_version": "v2", "actual_n": {"pilot": 40}}))
    assert not pilot.decide(*fixture[:2])["go"]
    sums = dataset.parent / "SHA256SUMS"
    sums.write_text(sums.read_text() + f"{pilot.sha256(dataset.read_bytes())}  {dataset.name}\n")
    assert pilot.decide(*fixture[:2])["go"]


def test_flat_verification_interface_and_remote_evidence_paths(fixture):
    results, gold, artifacts = fixture
    for condition, (path, _, nb, evidence) in artifacts.items():
        record = json.loads(evidence.read_text())
        record["runs"][condition]["archive"] = "/content/remote/" + path.name
        evidence.write_text(json.dumps(record))
        shutil.move(path, path.parent.parent / path.name)
        shutil.move(nb, nb.parent.parent / nb.name)
    assert pilot.decide(results, gold)["go"]


@pytest.mark.parametrize("state", [{"status": "failed", "verified": False, "completed_execution": False},
                                  {"status": "running", "verified": True, "completed_execution": False},
                                  {"status": "completed", "verified": False, "completed_execution": True}])
def test_operator_abort_or_incomplete_is_no_go(fixture, state):
    evidence = fixture[2]["B13"][3]
    evidence.with_name("status.json").write_text(json.dumps(state))
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert "Operator aborted" in " ".join(report["errors"])


def test_shared_data_hashes_must_match_even_when_individually_verified(fixture):
    results, gold, artifacts = fixture
    old_path, members, _, evidence = artifacts["G_SINGLE"]
    manifest = json.loads(members["manifest.json"])
    # Both maps are locally correct, but only one arm adds a frozen gold hash.
    cfg = manifest["config"]
    cfg["data_sha256"][gold.name] = pilot.sha256(gold.read_bytes())
    digest = pilot.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    run = f"G_SINGLE_pilot_v2_{digest[:8]}"
    manifest.update(config_hash=digest, run_name=run)
    members["manifest.json"] = json.dumps(manifest)
    for ledger in ("predictions", "metrics"):
        rows = pilot.jsonl(members[ledger + ".jsonl"])
        for row in rows:
            row.update(config_hash=digest, run_name=run, resume_key=pilot.sha256(
                f"{digest}|{row['problem_id']}|17|G_SINGLE|{cfg['profile']}".encode()))
        members[ledger + ".jsonl"] = lines(rows)
    new_path = old_path.with_name(run + "_final.zip")
    archive(new_path, members)
    old_path.unlink()
    fresh = pilot.verify_run(new_path.parent, conditions=["G_SINGLE"], inputs=gold.with_name("pilot_inputs.jsonl"),
                             config=ROOT / "config/experiment_v2.json", expected_count=40,
                             split="pilot", run_tag="pilot_v2")
    assert fresh["ok"], fresh
    evidence.write_text(json.dumps(fresh))
    report = pilot.decide(results, gold)
    assert all(e["verified"] for e in report["conditions"].values())
    assert not report["go"]
    assert "hashes or run tags differ" in " ".join(report["errors"])


def test_archive_snapshot_is_bound_to_the_graded_bytes(fixture, monkeypatch):
    read = pilot.read_archive
    target = fixture[2]["B13"][0]

    def changing_archive(path):
        snapshot = read(path)
        if path == target:
            change(fixture, "B13", "predictions.jsonl", lambda rows: rows[0].update(public_output="FINAL: 7"), True)
        return snapshot

    monkeypatch.setattr(pilot, "read_archive", changing_archive)
    report = pilot.decide(*fixture[:2])
    assert not report["go"]
    assert not report["conditions"]["B13"]["graded"]
    assert "Archive changed" in " ".join(report["errors"])
