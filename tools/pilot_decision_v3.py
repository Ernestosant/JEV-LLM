"""Offline v3 technical-only pilot gate. A GO is not budget approval."""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_run_v3 import CONDITIONS, SAMPLED, failure_rates, read_archive, require, verify_run


def estimate(rows, test_count):
    out = {"test_cases": test_count, "actual_cu": None}
    for field, label in (("T_total", "walltime_s"), ("candidate_tokens_total", "generated_tokens"),
                         ("processed_prompt_tokens", "processed_prompt_tokens"),
                         ("selector_input_tokens", "selector_input_tokens")):
        values = [r.get(field) for r in rows]
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in values)
        total = sum(values) if values and valid else None
        out["estimated_test_" + label] = total / len(values) * test_count if total is not None and math.isfinite(total) else None
    out["note"] = "Pilot mean scaled to planned test cases; excludes setup and model loading. CU unavailable."
    return out


def decide(results, inputs, *, config=ROOT / "config/experiment_v3.json", prompts=ROOT / "prompts", run_tag=None):
    report = {"decision": "no-go", "go": False, "protocol_version": "3", "code_version": "0.3.0",
              "conditions": {}, "aggregate": None, "errors": [], "budget_approval": False,
              "actual_cu": None, "note": "Technical-only pilot; no grading, baseline-win criterion, or test/LATENCY authorization."}
    found = {c: [] for c in CONDITIONS}
    try:
        require(Path(results).is_dir(), "Results directory missing")
        for path in sorted(Path(results).rglob("*_final.zip")):
            if any(p in {"checkpoints", "checkpoint_staging"} for p in path.parts):
                continue
            data = read_archive(path)
            condition = data["manifest"]["condition"]
            require(condition in found, "Unexpected pilot condition")
            found[condition].append((path, data))
    except Exception as error:
        report["errors"].append(f"Discovery: {type(error).__name__}: {error}")
    shared = []
    for condition in CONDITIONS:
        denominator = 150 if condition in SAMPLED else 50
        entry = report["conditions"][condition] = {"verified": False, "denominator": denominator}
        try:
            require(len(found[condition]) == 1, f"Expected one final archive, found {len(found[condition])}")
            path, data = found[condition][0]
            cfg = data["manifest"]["config"]
            tag = run_tag or cfg["params"]["RUN_TAG"]
            fresh = verify_run(path.parent, conditions=[condition], inputs=inputs, config=config,
                               expected_count=50, split="pilot", run_tag=tag, prompts=prompts)
            require(fresh["ok"], f"Verification failed: {fresh['runs'][condition].get('error')}")
            current = fresh["runs"][condition]
            require(current["archive_sha256"] == data["archive_sha256"], "Archive changed during verification")
            evidence_path = (path.parent.parent if path.parent.name == "artifacts" else path.parent) / "verification.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            recorded = evidence.get("runs", {}).get(condition, {})
            require(evidence.get("ok") is True and recorded.get("ok") is True
                    and all(recorded.get(k) == current[k] for k in
                            ("archive_sha256", "notebook_sha256", "config_hash", "records", "condition")),
                    "Stored verification does not bind archive/notebook")
            for name in ("status.json", "execution_handover.json"):
                state_path = evidence_path.parent / name
                if state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    require(state.get("status") == "completed" and state.get("verified") is True
                            and state.get("completed_execution") is True, "Incomplete/aborted operator state")
            rates = failure_rates(data, denominator)
            entry.update(verified=True, verification=current, rates=rates,
                         estimate=estimate(data["metrics"], denominator * 10),
                         latency_estimate=estimate(data["metrics"], 900) if condition in {"JFINAL", "B13_GREEDY"} else None)
            require(rates["timeout_infra_rate"] < 0.05, "Timeout/infra union must be strictly below 5%")
            shared.append((tag, cfg["experiment_config_sha256"], cfg["prompt_sha256"], cfg["data_sha256"], cfg["code_sha256"]))
        except Exception as error:
            report["errors"].append(f"{condition}: {type(error).__name__}: {error}")
    entries = [entry for entry in report["conditions"].values() if entry.get("verified")]
    report["dataset_profile"] = entries[0]["verification"]["dataset_provenance"] if entries else None
    combined = sum(e["rates"]["timeout_infra_count"] for e in entries)
    report["aggregate"] = {"denominator": 600, "observed_cases": sum(e["denominator"] for e in entries),
                           "complete": len(entries) == 6, "timeout_infra_count": combined,
                           "timeout_infra_rate": combined / 600}
    if combined / 600 >= 0.05:
        report["errors"].append("Aggregate timeout/infra union must be strictly below 5%")
    if shared and any(s != shared[0] for s in shared):
        report["errors"].append("Shared provenance/run tags differ across physical arms")
    estimates = [e["estimate"] for e in entries]
    report["estimated_test_walltime_s"] = (sum(e["estimated_test_walltime_s"] for e in estimates)
        if len(estimates) == 6 and all(e["estimated_test_walltime_s"] is not None for e in estimates) else None)
    latency_estimates = [e["latency_estimate"] for e in entries if e.get("latency_estimate")]
    report["estimated_latency_walltime_s"] = (sum(e["estimated_test_walltime_s"] for e in latency_estimates)
        if len(latency_estimates) == 2 and all(e["estimated_test_walltime_s"] is not None for e in latency_estimates) else None)
    report["estimates"] = {}
    for phase, source, size in (("test", estimates, 6), ("latency", latency_estimates, 2)):
        report["estimates"][phase] = {field.removeprefix("estimated_test_"): (
            sum(e[field] for e in source) if len(source) == size and all(e[field] is not None for e in source) else None)
            for field in ("estimated_test_walltime_s", "estimated_test_generated_tokens",
                          "estimated_test_processed_prompt_tokens", "estimated_test_selector_input_tokens")}
    report["walltime_estimate_scope"] = "Sum of serial hot case seconds; not elapsed time with concurrent VMs. Setup/loading excluded."
    report["go"] = not report["errors"] and len(entries) == 6
    report["decision"] = "go" if report["go"] else "no-go"
    return report


def write_report(report, output):
    output = Path(output)
    require(output.suffix == ".json", "Decision output must use .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report["paths"] = {"json": str(output), "markdown": str(output.with_suffix(".md"))}
    previous = json.loads(output.read_bytes()) if output.exists() else {}
    meta = report.setdefault("meta", {})
    meta.setdefault("generated_utc", previous.get("meta", {}).get("generated_utc")
                    or datetime.now(timezone.utc).isoformat())
    json_bytes = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    lines = ["# Pilot Decision: " + report["decision"].upper(), "", report["note"], "",
             "600 planned cases: 150 per sampled arm, 50 per greedy arm. Missing coverage stops the pilot.", "",
             "| Arm | Verified | Planned Cases | Timeout/Infra Union | Rate |", "|---|---|---|---|---|"]
    for condition, entry in report["conditions"].items():
        rates = entry.get("rates", {})
        lines.append(f"| {condition} | {entry['verified']} | {entry['denominator']} | "
                     f"{rates.get('timeout_infra_count', 'N/A')} | {rates.get('timeout_infra_rate', 'N/A')} |")
    lines.extend(["", "## Source Sampling And Intrinsic Difficulty", "",
                   "Source-tier quotas do not establish how many intrinsically hard test problems exist.",
                   "Actual intrinsic labels/counts are sealed agent-review metadata; this technical pilot never grades gold.",
                   "```json", json.dumps(report.get("dataset_profile"), indent=2), "```", "",
                   "## Pilot-Based Estimates", "", "No actual CU is available; estimates exclude setup/loading.",
                  "```json", json.dumps(report.get("estimates"), indent=2), "```", "",
                  "A technical GO does not create or replace the separate config-SHA budget approval sentinel.", "",
                  "## Blocking Errors", "", *("- " + e for e in report["errors"])])
    rendered = {output: json_bytes, output.with_suffix(".md"): ("\n".join(lines) + "\n").encode("utf-8")}
    for path, content in rendered.items():
        require(not path.exists() or path.read_bytes() == content,
                "Pinned pilot report differs; use a new filename: " + str(path))
    for path, content in rendered.items():
        if path.exists():
            require(path.read_bytes() == content, "Pinned pilot report differs; use a new filename: " + str(path))
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pilot-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # Linking a complete temp file publishes atomically without replacing a pinned file.
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(path.read_bytes() == content, "Pinned pilot report differs; use a new filename: " + str(path))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment_v3.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts")
    parser.add_argument("--run-tag")
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    report = decide(**args)
    write_report(report, output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
