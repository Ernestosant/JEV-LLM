"""Offline execution-integrity audit. Never reads gold or grades model answers."""

import argparse
import ast
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def expected_ids(inputs, count, split, schedule=None):
    rows = jsonl(Path(inputs).read_text(encoding="utf-8"))
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate input IDs")
    require(0 < count <= len(ids), "Expected count exceeds available inputs or is not positive")
    if count == len(ids):
        return set(ids)
    if split == "dev":
        return set(sorted(ids)[:count])
    path = Path(schedule) if schedule else Path(inputs).parent / "schedule.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    order = plan["pilot_order" if split == "pilot" else "order"]
    require(len(order) == len(set(order)) == len(ids) and set(order) == set(ids), "Schedule/input mismatch")
    return set(order[:count])


def check_notebook(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        # Actual CLI 0.7.4 downloads sometimes contain repr(dict), not JSON. Preserve raw.
        nb = ast.literal_eval(text)
    pm = nb.get("metadata", {}).get("papermill", {})
    require(pm.get("end_time") and "exception" in pm and pm["exception"] in (None, False),
            "Notebook incomplete or exception")
    cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code" and cell.get("source")]
    require(cells, "No executed code cells")
    for cell in cells:
        require(cell.get("metadata", {}).get("papermill", {}).get("status") == "completed",
                "Unfinished notebook cell")
        require(not any(output.get("output_type") == "error" for output in cell.get("outputs", [])),
                "Notebook error output")


def verify_archive(path, condition, *, inputs, config, expected_count, split, run_tag,
                   seeds=(17,), schedule=None, latency_rows=192, latency_dev_count=20,
                   prompts=ROOT / "prompts", code_version=None):
    path, inputs, config = Path(path), Path(inputs), Path(config)
    frozen = json.loads(config.read_text(encoding="utf-8"))
    require(path.name.endswith("_final.zip") and path.stat().st_size > 0, "Not a nonempty final ZIP")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
                    and "\\" not in n for n in names), "Unsafe ZIP member")
        require(archive.testzip() is None, "ZIP integrity failure")

        def member(name, optional=False):
            matches = [n for n in names if n == name or n.endswith("/" + name)]
            if optional and not matches:
                return ""
            require(len(matches) == 1, f"Expected exactly one {name}")
            return archive.read(matches[0]).decode("utf-8")

        manifest = json.loads(member("manifest.json"))
        cfg = manifest["config"]
        digest = sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), default=str).encode("utf-8"))
        require(manifest["config_hash"] == digest, "Manifest config hash mismatch")
        require(manifest["condition"] == condition and cfg["params"]["CONDITION"] == condition,
                "Manifest condition mismatch")
        require(manifest.get("finished_utc"), "Missing finished_utc final marker")
        require(manifest["run_name"] == f"{condition}_{run_tag}_{digest[:8]}"
                and path.name == manifest["run_name"] + "_final.zip", "Run/archive identity mismatch")
        require(cfg["params"].get("RUN_TAG") == run_tag and cfg["params"].get("SPLIT") == split,
                "Requested run tag or split mismatch")
        require(cfg["models"] == frozen["models"] and cfg["profile"] == frozen["implementation_profile"],
                "Frozen models/profile mismatch")
        require(cfg["jevk5_runtime"] == frozen["jevk5_runtime"]
                and cfg["protocol_version"] == frozen["protocol_version"], "Runtime/protocol mismatch")
        recorded_seeds = tuple(int(s) for s in str(cfg["params"]["SEEDS"]).replace(" ", "").split(","))
        require(set(recorded_seeds) == set(seeds) and len(recorded_seeds) == len(seeds), "Requested seeds mismatch")
        limit_params = {"n_candidates": "N_CANDIDATES", "max_output_tokens": "MAX_OUTPUT_TOKENS",
                        "hybrid_candidate_budget": "CANDIDATE_BUDGET", "j64_block": "J64_BLOCK",
                        "jstep_step": "JSTEP_STEP", "jstep_max_rounds": "JSTEP_MAX_ROUNDS",
                        "gb_context": "GB_CONTEXT", "j_max_input": "J_MAX_INPUT",
                        "max_prompt_tokens": "MAX_PROMPT_TOKENS", "timeout_s": "TIMEOUT_S"}
        for name, parameter in limit_params.items():
            if name in frozen.get("limits", {}):
                require(cfg["params"].get(parameter) == frozen["limits"][name], f"Frozen budget mismatch: {parameter}")
        for name, parameter in (("temperature", "TEMPERATURE"), ("top_p", "TOP_P"),
                                ("top_k", "TOP_K"), ("repetition_penalty", "REPETITION_PENALTY")):
            if name in frozen.get("sampling", {}):
                require(cfg["params"].get(parameter) == frozen["sampling"][name], f"Frozen sampling mismatch: {parameter}")
        if code_version:
            require(cfg.get("code_version") == code_version, "Code version mismatch")
        if "experiment_config_sha256" in cfg:
            require(cfg["experiment_config_sha256"] == sha256(config.read_bytes()), "Frozen config file hash mismatch")
        require(cfg["data_sha256"].get(inputs.name) == sha256(inputs.read_bytes()), "Inputs hash mismatch")
        for name, data_hash in cfg["data_sha256"].items():
            require(PurePosixPath(name).name == name, "Unsafe data hash name")
            frozen_data = inputs.parent / name
            require(frozen_data.is_file() and sha256(frozen_data.read_bytes()) == data_hash,
                    f"Frozen data hash mismatch: {name}")
        for name, digest_prompt in cfg["prompt_sha256"].items():
            text = (Path(prompts) / (name + ".txt")).read_text(encoding="utf-8").strip("\n")
            require(sha256(text.encode("utf-8")) == digest_prompt, f"Prompt hash mismatch: {name}")
        require(set(cfg["prompt_sha256"]) == {"generador", "criterio_paso", "criterio_final"}, "Missing prompt hashes")
        require(bool(manifest.get("stages")) and all(s.get("ok") is True for s in manifest["stages"].values()),
                "Missing or failed execution stages")
        require(all(not p.get("blocking") or p.get("ok") is True for p in manifest.get("preflight", {}).values()),
                "Failed blocking preflight")
        require(not jsonl(member("failures.jsonl", optional=True)), "Execution failures present")
        require(manifest.get("model_lock") and all(lock.get("ok") is True for lock in manifest["model_lock"].values()),
                "Missing or failed model lock")
        roles = {"B"} if condition.startswith("B13") else ({"G"} if condition in ("G_SINGLE", "G4_SINGLE")
                 else ({"G", "J", "B"} if condition == "LATENCY" else {"G", "J"}))
        require(roles <= set(manifest["model_lock"]), "Missing required model locks")
        if cfg["profile"] == "1COPY-G4-VLLM-GRAPH":
            require(cfg.get("code_version") == frozen.get("code_version", "0.2.0"), "v2 code version mismatch")
            gpus = manifest.get("environment", {}).get("gpus", [])
            require(len(gpus) == 1 and "A100" in gpus[0]["name"] and int(gpus[0]["memory.total"]) == 40960,
                    "v2 requires one A100 40GB")
            require(cfg["params"].get("KV_CACHE_GB_G") == 3.0 and cfg["params"].get("ENFORCE_EAGER") is False,
                    "v2 fixed infrastructure/profile mismatch")
            require(cfg.get("experiment_config_sha256") == sha256(config.read_bytes()), "Missing v2 config file hash")
            if condition != "LATENCY":
                require(bool(manifest.get("preflight")), "Missing v2 preflight evidence")
        # Check manifest counts against the actual ledgers, including empty/missing zero ledgers.
        require(bool(manifest.get("counts")), "Missing final counts")
        for ledger, count in manifest["counts"].items():
            require(len(jsonl(member(ledger + ".jsonl", optional=count == 0))) == count,
                    f"Manifest count mismatch: {ledger}")
        ledger = "latency_metrics.jsonl" if condition == "LATENCY" else "predictions.jsonl"
        records = jsonl(member(ledger))
        require(all(r.get("config_hash") == digest and r.get("t_start_utc") and r.get("t_end_utc")
                    and r.get("status") not in (None, "exception", "infrastructure", "run_aborted") for r in records),
                "Row hash, timestamps or execution status invalid")
        if condition == "LATENCY":
            plan = json.loads(member("latency_plan.json"))
            items = {item["item_id"]: item for item in plan["items"]}
            require(len(items) == len(plan["items"]), "Duplicate latency plan IDs")
            dev = {item["item_id"] for item in items.values() if item["kind"] == "dev"}
            input_rows = jsonl(inputs.read_text(encoding="utf-8"))
            expected_dev = {r["id"] for r in input_rows[:latency_dev_count]}
            require(len(expected_dev) == latency_dev_count and dev == expected_dev, "Latency dev ID membership mismatch")
            synth_count = int(cfg["params"].get("LAT_SYNTH_PER_BAND", 3))
            expected_synth = {f"synth-{band}-{k}": "synthetic_" + band
                              for band in ("short", "medium", "long", "xlong") for k in range(synth_count)}
            require(set(items) == expected_dev | set(expected_synth)
                    and all(items[item]["kind"] == kind for item, kind in expected_synth.items()),
                    "Latency synthetic ID membership mismatch")
            require(all(items[r["id"]]["problem"] == r["problem"] for r in input_rows[:latency_dev_count]),
                    "Latency dev prompt mismatch")
            blocks = {item: index for index, block in enumerate(plan["blocks"]) for item in block}
            require(sum(map(len, plan["blocks"])) == len(items) and set(blocks) == set(items), "Latency blocks mismatch")
            arms = frozen["conditions"]
            expected = {(item, arm, seed) for item in items for arm in arms for seed in seeds}
            require(len(expected) == latency_rows, "Latency plan row count mismatch")
            keys = [(r["item_id"], r["condition"], r["seed"]) for r in records]
            require(all(r["kind"] == items[r["item_id"]]["kind"] and r["block"] == blocks[r["item_id"]]
                        and r["phase"] == ("B" if r["condition"].startswith("B13") else "H") for r in records),
                    "Latency kind/block/phase mismatch")
        else:
            ids = expected_ids(inputs, expected_count, split, schedule)
            expected = {(item, condition, seed) for item in ids for seed in seeds}
            keys = [(r["problem_id"], r["condition"], r["seed"]) for r in records]
            metrics = jsonl(member("metrics.jsonl"))
            resume = [r["resume_key"] for r in records]
            require(len(set(resume)) == len(resume), "Duplicate resume keys")
            for row in records + metrics:
                key = sha256(f"{digest}|{row['problem_id']}|{row['seed']}|{condition}|{cfg['profile']}".encode())
                require(row["resume_key"] == key and row["condition"] == condition and row["config_hash"] == digest,
                        "Case/resume identity mismatch")
            require(Counter(r["resume_key"] for r in metrics) == Counter(resume), "Metrics coverage mismatch")
            if condition in ("JFINAL", "JFINAL_G4"):
                candidates = jsonl(member("candidates.jsonl"))
                decisions = jsonl(member("decisions.jsonl"))
                grouped = defaultdict(list)
                for row in candidates:
                    grouped[row["resume_key"]].append(row)
                require(set(grouped) == set(resume) and Counter(d["resume_key"] for d in decisions) == Counter(resume),
                        "JFINAL candidate/decision coverage mismatch")
                for decision in decisions:
                    cs = grouped[decision["resume_key"]]
                    require(len(cs) == 4 and {c["branch"] for c in cs} == {0, 1, 2, 3}, "JFINAL needs four branches per case")
                    for row in cs + [decision]:
                        auxiliary_key = sha256(f"{digest}|{row['problem_id']}|{row['seed']}|{condition}|{cfg['profile']}".encode())
                        require(row["round"] == 0 and row["config_hash"] == digest and row["condition"] == condition
                                and row["resume_key"] == auxiliary_key
                                and (row["problem_id"], condition, row["seed"]) in expected,
                                "JFINAL auxiliary identity mismatch")
                    require(sorted(decision["permutation"]) == [0, 1, 2, 3], "Invalid selector permutation")
                    require(decision["status"] in ("ok", "selector_context_limit"), "Invalid selector execution status")
                    if decision["status"] == "ok":
                        winner = decision["winner_branch"]
                        require(decision["permutation"][decision["winner_position"]] == winner
                                and [c["branch"] for c in cs if c["chosen"]] == [winner], "Selector winner mismatch")
                # eos_invalid/truncated are measured outcomes, not execution exceptions.
        require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected,
                "Exact expected ID/condition/seed coverage mismatch")
        progress = json.loads(member("progress.json"))
        require(progress["done"] == progress["total"] == len(expected), "Incomplete final progress")
        return {"ok": True, "condition": condition, "records": len(records), "config_hash": digest,
                "archive_sha256": sha256(path.read_bytes()), "archive": str(path),
                "statuses": dict(Counter(r["status"] for r in records))}


def verify_run(directory, *, conditions, **kwargs):
    directory = Path(directory)
    aliases = json.loads(Path(kwargs["config"]).read_text(encoding="utf-8")).get("condition_aliases", {})
    conditions = [aliases.get(condition, condition) for condition in conditions]
    require(len(conditions) == len(set(conditions)), "Duplicate requested conditions")
    results = {}
    for condition in conditions:
        try:
            paths = list(directory.glob(condition + "_" + kwargs["run_tag"] + "_*_final.zip"))
            require(len(paths) == 1, f"Expected one final archive for {condition}, found {len(paths)}")
            info = verify_archive(paths[0], condition, **kwargs)
            prefix = {"G_SINGLE": "01_", "G4_SINGLE": "01_", "B13": "02_", "B13_GREEDY": "03_",
                      "J64": "04_", "J64_G4": "04_", "JSTEP": "05_", "JSTEP_G4": "05_",
                      "JFINAL": "06_", "JFINAL_G4": "06_", "LATENCY": "08_"}[condition]
            notebooks = [p for p in directory.glob(prefix + "*.out.*.ipynb") if not p.name.endswith(".canonical.ipynb")]
            require(len(notebooks) == 1, "Expected one executed notebook, not retries")
            check_notebook(notebooks[0])
            info["notebook_sha256"] = sha256(notebooks[0].read_bytes())
            results[condition] = info
        except Exception as error:
            results[condition] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"ok": bool(results) and all(r["ok"] for r in results.values()), "runs": results,
            "note": "Execution integrity only; no gold grading or pilot go/no-go decision."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True, help="1 smoke, 40 pilot, actual N test")
    parser.add_argument("--split", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--latency-rows", type=int, default=192)
    parser.add_argument("--latency-dev-count", type=int, default=20)
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts")
    parser.add_argument("--code-version")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = verify_run(**args)
    text = json.dumps(report, indent=2) + "\n"
    if output:
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(output)
    print(text, end="")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
