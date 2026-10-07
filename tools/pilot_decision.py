"""CPU-only, fail-closed v2 pilot gate; never retries inference.

Interface: --results contains three operator directories, each holding artifacts/
with one canonical *_final.zip and one executed *.out.*.ipynb, plus the operator's
verification.json above artifacts/. Flat archive/notebook/verification.json
directories are also accepted. Evidence must use verify_run's {ok, runs} schema
and match freshly verified archive_sha256, notebook_sha256, config_hash and
records. Paths in evidence may be remote; hashes, not paths, bind the artifacts.
When present, operator status.json/execution_handover.json must show completed,
verified execution, not an abort or an in-progress run.
Checkpoint directories/ZIPs and extracted ledgers are never grading sources.
Defaults are the repository's frozen v2 config, sibling pilot_inputs.jsonl, and
repository prompts. Sibling SHA256SUMS must seal both pilot files (and the
dataset_manifest.json when present). --config/--prompts support offline fixtures.
Exit 0 means go, 1 means no-go; JSON and same-stem Markdown explain both.
"""

import argparse
import io
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from jevlab.evaluate import grade_prediction, grade_text
from verify_run import jsonl, require, sha256, verify_run

CONDITIONS = ("JFINAL", "B13", "G_SINGLE")
INFRA = {"infrastructure", "exception", "run_aborted", "aborted"}
BLOCKING = {"parser_selftest", "lock", "chat_template", "prompt_length", "sampling_semantics", "tokenizer", "stress"}


def read_archive(path):
    """Read members without extracting or trusting any loose ledger."""
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")

        def member(name, optional=False):
            matches = [n for n in names if n == name or n.endswith("/" + name)]
            if optional and not matches:
                return ""
            require(len(matches) == 1, f"Expected exactly one {name}")
            return archive.read(matches[0]).decode("utf-8")

        return {"archive_sha256": sha256(raw), "manifest": json.loads(member("manifest.json")),
                **{name: jsonl(member(name + ".jsonl", optional=True))
                   for name in ("predictions", "metrics", "candidates", "failures")}}


def failure_rates(data, denominator):
    # A failure ledger may repeat a prediction; count failed cases, not events.
    timeout, infra = set(), set()
    for row in data["predictions"] + data["failures"]:
        key = (row.get("problem_id"), row.get("seed"))
        status = row.get("kind", row.get("status"))
        if status == "timeout":
            timeout.add(key)
        if status in INFRA:
            infra.add(key)
    return {"denominator": denominator, "observed_predictions": len(data["predictions"]),
            "timeout_count": len(timeout), "infra_count": len(infra),
            "timeout_infra_count": len(timeout | infra),
            "timeout_rate": len(timeout) / denominator,
            "infra_rate": len(infra) / denominator,
            "timeout_infra_rate": len(timeout | infra) / denominator}


def metric_summary(rows):
    out = {}
    for field in ("T_total", "T_proposals", "T_selector", "T_prefill", "T_cache_sync",
                  "T_first_accepted", "nvml_peak_used_bytes", "rss_gib"):
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))
                  and not isinstance(r[field], bool) and math.isfinite(r[field]) and r[field] >= 0]
        require(not values or math.isfinite(sum(values)), f"Metric overflow: {field}")
        out[field] = {"n": len(values), "missing_or_invalid": len(rows) - len(values),
                      "mean": statistics.mean(values) if values else None,
                      "median": statistics.median(values) if values else None,
                      "max": max(values) if values else None,
                      "sum": sum(values) if values else None}
    return out


def decide(results, gold, *, config=ROOT / "config/experiment_v2.json", prompts=ROOT / "prompts"):
    report = {"decision": "no-go", "go": False, "protocol_version": "2", "seed": 17,
              "expected_cases_per_condition": 40, "failure_rate_limit_exclusive": 0.05,
              "errors": [], "conditions": {}, "aggregate": None,
              "note": "Only verified immutable finals are graded. No quality retry is authorized."}
    errors = report["errors"]
    gold_map, frozen, common_configs = None, None, []
    results, gold, config, prompts = map(Path, (results, gold, config, prompts))
    inputs = gold.with_name("pilot_inputs.jsonl")
    try:
        config_bytes = config.read_bytes()
        frozen = json.loads(config_bytes.decode("utf-8"))
        require(frozen["protocol_version"] == "2", "Frozen protocol must be '2'")
        require(gold.name == "pilot_gold.jsonl", "Expected frozen pilot_gold.jsonl")
        sums_path = gold.parent / "SHA256SUMS"
        seals = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            require(name not in seals, f"Duplicate SHA256SUMS entry: {name}")
            seals[name] = digest
        sealed_bytes = {path.name: path.read_bytes() for path in (gold, inputs)}
        for path in (gold, inputs):
            require(seals.get(path.name) == sha256(sealed_bytes[path.name]), f"Frozen pilot hash mismatch: {path.name}")
        dataset = gold.parent / "dataset_manifest.json"
        if dataset.exists():
            require(seals.get(dataset.name) == sha256(dataset.read_bytes()), "Dataset manifest hash mismatch")
            metadata = json.loads(dataset.read_text(encoding="utf-8"))
            require(metadata["dataset_version"] == "v2" and metadata["actual_n"]["pilot"] == 40,
                    "Not the v2 40-case pilot dataset")
        gs = jsonl(sealed_bytes[gold.name].decode("utf-8"))
        ins = jsonl(sealed_bytes[inputs.name].decode("utf-8"))
        gold_map = {g["id"]: g for g in gs}
        ids = [r["id"] for r in ins]
        require(len(gs) == len(gold_map) == len(ins) == len(set(ids)) == 40
                and set(ids) == set(gold_map), "Gold/inputs must match exactly 40 unique pilot IDs")
        require(all(set(r) == {"id", "problem", "language"} for r in ins), "Inputs contain non-input fields")
        for g in gs:
            require(grade_text("FINAL: " + g["gold_answer"], g)["correct"], "Inconsistent rational gold")
        report["frozen"] = {"config_sha256": sha256(config_bytes), "code_version": frozen.get("code_version", "0.2.0"),
                            "gold_sha256": sha256(sealed_bytes[gold.name]), "inputs_sha256": sha256(sealed_bytes[inputs.name]),
                            "SHA256SUMS_sha256": sha256(sums_path.read_bytes()), "ids": sorted(ids)}
    except Exception as error:
        errors.append(f"Frozen data/config: {type(error).__name__}: {error}")
        gold_map = None

    archives = defaultdict(list)
    try:
        require(results.is_dir(), "Results directory does not exist")
        for path in sorted(results.rglob("*_final.zip")):
            if any(p in {"checkpoints", "checkpoint_staging"} for p in path.relative_to(results).parts[:-1]):
                continue
            try:
                data = read_archive(path)
                condition = data["manifest"]["condition"]
                require(condition in CONDITIONS, f"Unexpected final condition: {condition}")
                archives[condition].append((path, data))
            except Exception as error:
                errors.append(f"Final {path}: {type(error).__name__}: {error}")
    except Exception as error:
        errors.append(f"Results: {type(error).__name__}: {error}")

    for condition in CONDITIONS:
        entry = report["conditions"][condition] = {"verified": False, "graded": False}
        found = archives[condition]
        if len(found) != 1:
            errors.append(f"{condition}: Expected one final archive, found {len(found)} (no retries)")
            continue
        path, data = found[0]
        try:
            entry.update(archive=str(path), archive_sha256=data["archive_sha256"],
                         rates=failure_rates(data, 40), statuses=dict(Counter(r.get("status") for r in data["predictions"])))
            require(frozen is not None and gold_map is not None, "Frozen pilot validation failed")
            manifest = data["manifest"]
            cfg = manifest["config"]
            require(cfg["protocol_version"] == "2", "Manifest protocol must be '2'")
            require(cfg.get("code_version") == frozen.get("code_version", "0.2.0"), "Frozen code version mismatch")
            require(cfg.get("experiment_config_sha256") == report["frozen"]["config_sha256"]
                    and cfg["data_sha256"].get(inputs.name) == report["frozen"]["inputs_sha256"],
                    "Run does not bind the frozen config/inputs used for this decision")
            # Runner excludes N_PROBLEMS from semantic config_hash; verified coverage fixes N.
            require(cfg["params"].get("N_PROBLEMS", 40) == 40, "Not a 40-case pilot (smoke/test mixture)")
            require(manifest.get("condition_label") == frozen["condition_labels"][condition]
                    and all(r.get("condition_label") == frozen["condition_labels"][condition]
                            for r in data["predictions"]), "Canonical condition label mismatch")
            preflight = manifest.get("preflight", {})
            required = BLOCKING | ({"selector_native_reference", "selector_equivalence"} if condition == "JFINAL" else set())
            require(required <= set(preflight) and all(preflight[k].get("blocking") is True
                    and preflight[k].get("ok") is True for k in required)
                    and all(p.get("ok") is True for p in preflight.values() if p.get("blocking")),
                    "Missing or failed blocking preflight evidence")
            entry["preflight"] = preflight
            entry["condition_label"] = manifest["condition_label"]
            tag = cfg["params"]["RUN_TAG"]
            fresh = verify_run(path.parent, conditions=[condition], inputs=inputs, config=config,
                               expected_count=40, split="pilot", run_tag=tag, seeds=(17,), prompts=prompts,
                               code_version=frozen.get("code_version", "0.2.0"))
            entry["verification"] = fresh
            require(fresh["ok"], f"Verification failed: {fresh['runs'][condition].get('error')}")
            require(all(r.get("status") in {"final", "truncated", "max_rounds", "timeout", "eos_invalid",
                                           "selector_context_limit"} for r in data["predictions"]),
                    "Aborted or unknown prediction status")
            evidence_path = (path.parent.parent if path.parent.name == "artifacts" else path.parent) / "verification.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            require(evidence.get("ok") is True, "Stored verification failed")
            recorded, current = evidence["runs"][condition], fresh["runs"][condition]
            require(recorded.get("ok") is True and all(recorded.get(k) == current[k] for k in
                    ("condition", "archive_sha256", "notebook_sha256", "config_hash", "records")),
                    "Verification evidence does not bind this exact archive and notebook")
            require(current["archive_sha256"] == entry["archive_sha256"], "Archive changed during verification")
            for name in ("status.json", "execution_handover.json"):
                state_path = evidence_path.parent / name
                if state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    require(state.get("status") == "completed" and state.get("verified") is True
                            and state.get("completed_execution") is True,
                            f"Operator aborted, incomplete or unverified: {name}")
            entry.update(verified=True, evidence=str(evidence_path), notebook_sha256=current["notebook_sha256"])
            # Per-condition config hashes differ; shared frozen provenance must not.
            common_configs.append({k: cfg[k] for k in ("models", "profile", "jevk5_runtime", "protocol_version",
                                                       "code_version", "experiment_config_sha256", "data_sha256", "prompt_sha256")} |
                                   {"params": {k: v for k, v in cfg["params"].items()
                                               if k not in {"CONDITION", "N_PROBLEMS"}}})
            grades = [grade_prediction(r, gold_map[r["problem_id"]]) for r in data["predictions"]]
            entry.update(graded=True, grades=grades, correct=sum(g["correct"] for g in grades),
                         accuracy=sum(g["correct"] for g in grades) / 40,
                         valid_format_rate=sum(g["valid_format"] for g in grades) / 40,
                         metrics=metric_summary(data["metrics"]))
            if condition == "JFINAL":
                candidates = defaultdict(list)
                for c in data["candidates"]:
                    candidates[c["problem_id"]].append(grade_text(c["text"], gold_map[c["problem_id"]]))
                oracle_ids = {pid for pid, cs in candidates.items() if any(c["correct"] for c in cs)}
                captured = sum(g["correct"] for g in grades if g["problem_id"] in oracle_ids)
                entry["oracle"] = {"correct_available_cases": len(oracle_ids), "captured_cases": captured,
                                   "oracle_at_4": len(oracle_ids) / 40,
                                   "capture_ratio": captured / len(oracle_ids) if oracle_ids else None,
                                   "candidate_valid_format_rate": sum(c["valid_format"] for cs in candidates.values()
                                                                      for c in cs) / 160}
        except Exception as error:
            errors.append(f"{condition}: {type(error).__name__}: {error}")
        if entry.get("rates", {}).get("timeout_infra_rate", 0) >= 0.05:
            errors.append(f"{condition}: timeout/infra rate must be strictly below 5%")

    if len(common_configs) == 3 and any(c != common_configs[0] for c in common_configs[1:]):
        errors.append("Frozen models/config/data/prompt hashes or run tags differ across conditions")
    rate_entries = [e["rates"] for e in report["conditions"].values() if "rates" in e]
    if rate_entries:
        aggregate = {"denominator": 120, "conditions_observed": len(rate_entries),
                     "complete": len(rate_entries) == 3}
        for name in ("timeout", "infra", "timeout_infra"):
            aggregate[name + "_count"] = sum(e[name + "_count"] for e in rate_entries)
            aggregate[name + "_rate"] = aggregate[name + "_count"] / 120
        report["aggregate"] = aggregate
        if aggregate["timeout_infra_rate"] >= 0.05:
            errors.append("Aggregate timeout/infra rate must be strictly below 5%")
    if all(e.get("graded") for e in report["conditions"].values()):
        primary = report["conditions"]["JFINAL"]["accuracy"] >= report["conditions"]["B13"]["accuracy"]
        report["jfinal_at_least_b13"] = primary
        if not primary:
            errors.append("JFINAL strict accuracy is below B13")
    report["go"] = not errors
    report["decision"] = "go" if report["go"] else "no-go"
    return report


def write_report(report, output):
    output = Path(output)
    require(output.suffix.lower() == ".json", "Decision output must use .json (Markdown is written separately)")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pilot Decision: " + report["decision"].upper(), "", report["note"], "",
             "Rates use 40 planned cases per condition and 120 overall; missing cases never count as success.", "",
             "| Condition | Verified | Accuracy | Valid format | Timeout | Infra | Combined |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for condition, entry in report["conditions"].items():
        rates = entry.get("rates", {})
        values = [entry.get("accuracy"), entry.get("valid_format_rate"), rates.get("timeout_rate"),
                  rates.get("infra_rate"), rates.get("timeout_infra_rate")]
        label = entry.get("condition_label", condition)
        lines.append(f"| {condition}: {label} | {entry['verified']} | " + " | ".join(
            "N/A" if v is None else f"{v:.2%}" for v in values) + " |")
    lines.extend(["", "## Aggregate", "", "```json", json.dumps(report["aggregate"], indent=2), "```",
                  "", "## Diagnostics", ""])
    for condition, entry in report["conditions"].items():
        if entry.get("graded"):
            lines.extend([f"### {condition}: {entry['condition_label']}", "", "```json",
                          json.dumps({k: entry[k] for k in ("oracle", "metrics") if k in entry}, indent=2), "```", ""])
    lines.extend(["## Blocking Errors", ""] + (["- " + e for e in report["errors"]] or ["None."]))
    for target, text in ((output, json.dumps(report, indent=2, allow_nan=False) + "\n"),
                         (output.with_suffix(".md"), "\n".join(lines) + "\n")):
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment_v2.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json; same-stem Markdown is generated separately")
    try:
        report = decide(args.results, args.gold, config=args.config, prompts=args.prompts)
    except Exception as error:
        # Invalid artifact schemas must still leave an explicit no-go decision.
        report = {"decision": "no-go", "go": False, "conditions": {}, "aggregate": None,
                  "note": "Pilot validation aborted; no grading or quality retry authorized.",
                  "errors": [f"{type(error).__name__}: {error}"]}
    write_report(report, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
