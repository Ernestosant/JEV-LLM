"""Build the isolated, immutable second-series dataset, before any inference.

Run: python data/build_dataset_v2.py. A completed data/v2 cannot be overwritten
without an explicit one-off pre-inference repair that archives the rejected candidate.
The previous builder supplies pinned source loaders and unchanged difficulty rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

import build_dataset as previous

ROOT = previous.ROOT
OUT = ROOT / "data" / "v2"
INITIAL_SEED = 20261001
SEED = 20261001
PILOT_QUOTA = {"easy": 4, "medium": 4}
G4 = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
META_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CURRENCY_UNITS = re.compile(previous.COMP_MONEY.pattern +
    r"|[\u00a3\u20ac\u00a5\u20b9\u20bd\u00a2]|\b(?:euros?|rupees?|yen|yuan|pesos?|"
    r"francs?|pounds?\s+sterling|sterling|USD|EUR|GBP|JPY|CNY|INR|RUB|CAD|AUD)\b", re.I)
GOLD_UNITS = re.compile(CURRENCY_UNITS.pattern +
    r"|\$|percent|%|\b(?:pounds?|penn(?:y|ies)|nickels?|dimes?|quarters?|currency)\b", re.I)
MONETARY_WORDS = re.compile(
    r"\b(?:money|cash|currency|cost(?:s|ing)?|prices?|paid|pay(?:s|ing|ments?)?|"
    r"spend(?:s|ing)?|spent|salar(?:y|ies)|wages?|income|revenue|profit|"
    r"charges?|charged|charging|budget|purchase(?:s|d)?|refund(?:s|ed)?)\b", re.I)
NONMONETARY_SPEND = re.compile(
    r"\b(?:spend(?:s|ing)?|spent)\b(?:\s+(?:about|exactly|only|half|all|most|some|the|"
    r"his|her|their|that|this|of|\d+(?:\.\d+)?%?)){0,8}\s+"
    r"(?:time|hours?|minutes?|seconds?|days?|weeks?|months?|years?|energy)\b|"
    r"\b(?:time|hours?|minutes?|seconds?|days?|weeks?|months?|years?|energy)\b"
    r"(?:\s+\w+){0,4}\s+(?:spend(?:s|ing)?|spent)\b", re.I)
COIN_WORDS = re.compile(r"\b(?:coins?|penn(?:y|ies)|nickels?|dimes?|quarters?)\b", re.I)
COIN_VALUE = re.compile(r"\b(?:(?:total|monetary|face)\s+value|worth|value\s+of)\b", re.I)
COUNT_QUESTION = re.compile(r"\b(?:how many|number of ways|probability|probabilities)\b", re.I)
AMOUNT_QUESTION = re.compile(
    r"\b(?:how much|money|cash|currency|cost(?:s|ing)?|prices?|paid|salary|wages?|"
    r"(?:total|monetary|face)\s+value|worth|spend(?:s|ing)?|spent)\b", re.I)
HARD_PREF = {**previous.HARD_PREF, "ratios_percentages": ["competition", "math", "gsm8k"]}
SPLITS = ("test", "pilot", "dev")
GENERATED_FILES = ({f"{split}_{kind}.jsonl" for split in SPLITS for kind in ("inputs", "gold")}
                   | {"schedule.json", "dataset_manifest.json", "blind_review_ids.json",
                      "review_sheet.csv", "SHA256SUMS"}
                   | {"model_meta/G4/" + name for name in META_FILES})
REJECTION_REASON = ("Pre-inference statement review found jev-v2-t-028 "
                    "(openai/gsm8k::test::814) asks 'How much money is used?' with quarter costs. "
                    "The original COMP_MONEY regex misses 'money', and bare numeric gold 11 "
                    "does not reveal the monetary unit. Initial candidate rejected; correct "
                    "the monetary filter before selection, with no inference or quality-based tuning.")
SEED_RESTORATION_REASON = ("The agent incorrectly changed the required construction seed from "
                           "20261001 to 20261002 during the monetary-filter repair and incorrectly "
                           "called that change preregistered. The change was not authorized. "
                           "The user explicitly requires restoring the original seed 20261001, "
                           "preserving the wrong-seed candidate byte-for-byte, and retaining the "
                           "existing corrected monetary filter. No inference or quality-based tuning.")


def monetary_answer(problem):
    if CURRENCY_UNITS.search(problem):
        return True
    question = re.split(r"(?<=[.!?])\s+", problem.strip())[-1]
    if COIN_WORDS.search(problem) and COIN_VALUE.search(question):
        return True
    # Denomination names alone are not monetary answers: keep coin-count combinatorics.
    if COIN_WORDS.search(problem) and COUNT_QUESTION.search(question) and not AMOUNT_QUESTION.search(question):
        return False
    return bool(MONETARY_WORDS.search(NONMONETARY_SPEND.sub(" ", problem)))


def correction_policy():
    return {"reason": REJECTION_REASON, "filter_applied_before_selection": True,
            "monetary_words_regex": MONETARY_WORDS.pattern,
            "currency_units_regex": CURRENCY_UNITS.pattern, "gold_units_regex": GOLD_UNITS.pattern,
            "nonmonetary_spend_regex": NONMONETARY_SPEND.pattern,
            "coin_words_regex": COIN_WORDS.pattern, "count_question_regex": COUNT_QUESTION.pattern,
            "coin_value_regex": COIN_VALUE.pattern,
            "amount_question_regex": AMOUNT_QUESTION.pattern, "case_insensitive": True,
            "coin_counting_exempt_unless_asks_monetary_amount": True,
            "original_filters_retained": True, "no_result_based_tuning": True,
            "preregistered_seed": INITIAL_SEED, "construction_seed": SEED,
            "seed_policy": "Keep the original required seed 20261001 during monetary-filter repairs."}


def no_runtime_execution():
    results = ROOT / "results" / "v2"
    if results.is_symlink() or (results.exists() and not results.is_dir()):
        raise ValueError("Cannot verify results/v2: not a plain directory")
    for path in sorted(results.rglob("*")):
        if path.is_symlink() or (path.is_file() and path.relative_to(results).parts[0] != "preparation"):
            raise ValueError(f"Repair forbidden: runtime execution artifact outside preparation: {path}")
    return {"scope": "results/v2", "allowed_subdirectory": "preparation",
            "runtime_execution_files": [], "checked_before_archive_and_unseal": True}


def repair_before_inference(*, restore_seed=False):
    """Preserve and verify a specifically rejected freeze BEFORE unsealing generated files."""
    guard = no_runtime_execution()
    preparation = ROOT / "results" / "v2" / "preparation"
    stem = "dataset_wrong_seed_rejected" if restore_seed else "dataset_initial_rejected"
    archive = preparation / f"{stem}.zip"
    record = preparation / f"{stem}.json"
    if archive.exists() or record.exists():
        raise ValueError("Rejection archive/audit already exists; refusing a second repair of this candidate")
    if OUT.is_symlink() or not (OUT / "SHA256SUMS").is_file():
        raise ValueError("Repair requires the initial sealed dataset")
    initial = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Cannot archive a symlink: {path}")
        if path.is_file():
            initial[path.relative_to(OUT).as_posix()] = path.read_bytes()
    if set(initial) != GENERATED_FILES:
        raise ValueError("Initial dataset has unexpected/missing files; unseal only known generated files")
    inventory = {name: hashlib.sha256(content).hexdigest() for name, content in initial.items()}
    sealed = {}
    for line in initial["SHA256SUMS"].decode("utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in sealed:
            raise ValueError("Duplicate initial SHA256SUMS entry")
        sealed[name] = digest
    if sealed != {name: digest for name, digest in inventory.items() if name != "SHA256SUMS"}:
        raise ValueError("Initial candidate does not match its complete sealed hash inventory")
    initial_manifest = json.loads(initial["dataset_manifest.json"])
    expected_seed = 20261002 if restore_seed else INITIAL_SEED
    if initial_manifest.get("seed") != expected_seed or initial_manifest.get("dataset_version") != "v2":
        raise ValueError("Repair is limited to the explicitly rejected candidate, not a seed/quality retry")
    if restore_seed:
        initial_record = preparation / "dataset_initial_rejected.json"
        money_audit = json.loads(initial_record.read_text(encoding="utf-8"))
        if (initial_manifest.get("initial_candidate_audit") != money_audit
                or initial_manifest.get("initial_candidate_audit_sha256") != previous.sha256_file(initial_record)
                or money_audit.get("archive") != "results/v2/preparation/dataset_initial_rejected.zip"
                or previous.sha256_file(preparation / "dataset_initial_rejected.zip") != money_audit["archive_sha256"]):
            raise ValueError("Wrong-seed candidate is not bound to the preserved initial money rejection")
        for key, value in correction_policy().items():
            if key not in ("preregistered_seed", "construction_seed", "seed_policy"):
                if initial_manifest.get("monetary_filter_correction", {}).get(key) != value:
                    raise ValueError(f"Seed restoration must retain the existing monetary filter: {key}")
        audit = {"event": "wrong_seed_candidate_rejected_before_inference", "reason": SEED_RESTORATION_REASON,
                 "wrong_seed": expected_seed, "restored_seed": SEED,
                 "seed_change_was_authorized": False, "prior_preregistration_claim_was_false": True,
                 "wrong_seed_files_sha256": inventory,
                 "wrong_seed_manifest_sha256": inventory["dataset_manifest.json"],
                 "wrong_seed_sha256sums_sha256": inventory["SHA256SUMS"],
                 "wrong_seed_builder_sha256": initial_manifest["builder_sha256"],
                 "wrong_seed_counts": initial_manifest["actual_n"],
                 "initial_money_rejection_audit": money_audit,
                 "initial_money_rejection_audit_sha256": previous.sha256_file(initial_record),
                 "monetary_filter_unchanged": True, "correction": correction_policy(),
                 "pre_inference_guard": guard}
    else:
        initial_inputs = [json.loads(line) for line in initial["test_inputs.jsonl"].decode("utf-8").splitlines()]
        initial_gold = [json.loads(line) for line in initial["test_gold.jsonl"].decode("utf-8").splitlines()]
        item = next((i for i in initial_inputs if i["id"] == "jev-v2-t-028"), None)
        gold = next((g for g in initial_gold if g["id"] == "jev-v2-t-028"), None)
        if not item or not gold or gold["template_group"] != "openai/gsm8k::test::814" or not monetary_answer(item["problem"]):
            raise ValueError("Initial candidate does not contain the known monetary-filter defect")
        audit = {"event": "initial_candidate_rejected_before_inference", "reason": REJECTION_REASON,
                 "initial_seed": INITIAL_SEED, "replacement_seed": SEED,
                 "initial_files_sha256": inventory,
                 "initial_manifest_sha256": inventory["dataset_manifest.json"],
                 "initial_sha256sums_sha256": inventory["SHA256SUMS"],
                 "initial_builder_sha256": initial_manifest["builder_sha256"],
                 "initial_counts": initial_manifest["actual_n"],
                 "rejected_item": {**item, "template_group": gold["template_group"], "gold_answer": gold["gold_answer"]},
                 "correction": correction_policy(), "pre_inference_guard": guard}
    preparation.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in initial.items():
            handle.writestr("data/v2/" + name, content)
        handle.writestr("hash_inventory.json", json.dumps(inventory, sort_keys=True, indent=2) + "\n")
    # Verify archived bytes, not just successful ZIP creation, before any original is writable.
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None or any(handle.read("data/v2/" + name) != content for name, content in initial.items()):
            raise ValueError("Rejected-candidate archive verification failed; initial files remain sealed")
    audit["archive"] = str(archive.relative_to(ROOT)).replace("\\", "/")
    audit["archive_sha256"] = previous.sha256_file(archive)
    with record.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    archive.chmod(0o444)
    record.chmod(0o444)
    no_runtime_execution()
    if any(previous.sha256_file(OUT / name) != digest for name, digest in inventory.items()):
        raise ValueError("Initial dataset changed during archival; refusing to unseal")
    for name in sorted(GENERATED_FILES):
        (OUT / name).chmod(0o644)
    (OUT / "SHA256SUMS").unlink()
    return audit


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_id(item):
    return f"{item['source']}::{item['source_split']}::{item['source_index']}"


def assign(item):
    # The old parser strips outer dollars, and competition overrides bypass unit filters.
    # Reject money/percent gold explicitly, including those otherwise hidden by parsing.
    if GOLD_UNITS.search(item.get("gold_raw") or ""):
        return None, None, "money_or_percent_gold"
    if monetary_answer(item["problem"]):
        return None, None, "money"
    if previous.ASKS_PERCENT.search(item["problem"]):
        return None, None, "asks_percent"
    return previous.assign(item)


def load_pool():
    paths = {}
    for key, (repo, revision, names) in previous.SOURCES.items():
        base = previous.RAW / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision
        paths[key] = [base / name for name in names]
        for path in paths[key]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing pinned local source (raw is read-only): {path}")
    old_manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, digest in old_manifest["aime_local_files"].items():
        if previous.sha256_file(previous.RAW / name) != digest:
            raise ValueError(f"Previous pinned AIME hash changed: {name}")
    pool = (previous.gsm8k_items(paths["gsm8k"]) + previous.math_items(paths["math"])
            + previous.competition_items(paths["amc"]))
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): previous.sha256_file(path)
              for group in paths.values() for path in group}
    hashes.update({"data/raw/" + name: previous.sha256_file(previous.RAW / name)
                   for name in previous.AIME_FILES})
    return pool, hashes


def load_previous():
    inputs, gold = [], []
    for split, expected in (("test", 100), ("dev", 20)):
        ii = read_jsonl(ROOT / "data" / f"{split}_inputs.jsonl")
        gg = read_jsonl(ROOT / "data" / f"{split}_gold.jsonl")
        if len(ii) != expected or len(gg) != expected or [i["id"] for i in ii] != [g["id"] for g in gg]:
            raise ValueError(f"Previous {split} must contain {expected} aligned records")
        inputs.extend(ii)
        gold.extend(gg)
    if len({g["template_group"] for g in gold}) != 120 or len({i["id"] for i in inputs}) != 120:
        raise ValueError("Previous 120 records are not unique")
    return inputs, gold


def token_counter():
    from jevlab.prompts import load_fast_tokenizer_file, render_generator_prompt

    system = previous.load_prompt(ROOT / "prompts" / "generador.txt")
    tokenizers = {"G4": load_fast_tokenizer_file(str(OUT / "model_meta" / "G4")),
                  "B": load_fast_tokenizer_file(str(previous.RAW / "model_meta" / "B"))}

    def count(problem):
        return {key: len(tok.encode(render_generator_prompt(tok, system, problem),
                                    add_special_tokens=False)) for key, tok in tokenizers.items()}

    return count


def select(pool, old_inputs, old_gold, count_tokens):
    """Pure deterministic selection, also replayed by tests against every source row."""
    rng = random.Random(SEED)
    blocked = {g["template_group"] for g in old_gold}
    excluded = Counter()
    available_before_length = Counter()
    buckets = defaultdict(list)
    for original in sorted(pool, key=source_id):
        it = dict(original)
        if source_id(it) in blocked:
            excluded["previous_source_id"] += 1
            continue
        it["gold"] = previous.gold_value(it)
        dom, diff, rule = assign(it)
        if dom is None:
            excluded[rule] += 1
            continue
        available_before_length[f"{dom}/{diff}/{previous.family(it)}"] += 1
        if any(previous.near_duplicate(it["problem"], old["problem"]) for old in old_inputs):
            excluded["near_duplicate_previous120"] += 1
            continue
        tc = count_tokens(it["problem"])
        if max(tc.values()) > previous.MAX_PROMPT_TOKENS:
            excluded["prompt_gt_512_tokens"] += 1
            continue
        it.update(domain=dom, difficulty=diff, rule=rule, prompt_tokens=tc)
        buckets[(dom, diff, previous.family(it))].append(it)
    available = {"/".join(key): len(value) for key, value in sorted(buckets.items())}
    for key in sorted(buckets):
        rng.shuffle(buckets[key])

    chosen = []
    selections = []

    def take(dom, diff, n, split):
        families = (HARD_PREF[dom] if diff == "hard" else ["gsm8k", "math"]
                    if dom == "ratios_percentages" else ["gsm8k"]
                    if dom == "arithmetic" else ["math"])
        queues = [buckets[(dom, diff, family)] for family in families]
        got, qi = 0, 0
        while got < n and any(queues):
            if diff == "hard":
                queue = next(q for q in queues if q)
            else:
                queue = queues[qi % len(queues)]
                qi += 1
                if not queue:
                    continue
            it = queue.pop(0)
            if any(previous.near_duplicate(it["problem"], other["problem"]) for other in chosen):
                excluded[f"near_duplicate_selected_{split}"] += 1
                continue
            it["split"] = split
            chosen.append(it)
            got += 1
        selections.append({"split": split, "domain": dom, "difficulty": diff,
                           "requested": n, "selected": got})
        return got

    # Reserve every test stratum before consuming even one pilot candidate.
    for dom in previous.DOMAINS:
        for diff, n in previous.TEST_QUOTA.items():
            take(dom, diff, n, "test")
    for dom in previous.DOMAINS:
        easy = take(dom, "easy", 4, "pilot")
        medium = take(dom, "medium", 4, "pilot")
        if easy + medium < 8:
            take(dom, "easy", 8 - easy - medium, "pilot")
    for split in ("test", "pilot"):
        items = [it for it in chosen if it["split"] == split]
        rng.shuffle(items)
        for number, it in enumerate(items, 1):
            it["id"] = f"jev-v2-{split[0]}-{number:03d}"
    return chosen, {"pool_size": len(pool), "exclusions": dict(excluded),
                    "available_before_length_and_near_duplicates": dict(available_before_length),
                    "available_after_filters": available, "selection_attempts": selections}


def gold_record(it):
    value = it["gold"]
    return {"id": it["id"], "gold_numerator": value.numerator,
            "gold_denominator": value.denominator, "gold_answer": previous.fraction_to_str(value),
            "gold_raw": it["gold_raw"], "reference_solution": it["solution"],
            "domain": it["domain"], "difficulty": it["difficulty"], "difficulty_rule": it["rule"],
            "template_group": source_id(it), "source": it["source"],
            "source_split": it["source_split"], "source_index": it["source_index"],
            "prompt_tokens": it["prompt_tokens"], "reviewed": False}


def schedule_for(test, pilot):
    rng = random.Random(SEED)
    by_domain = defaultdict(list)
    for it in sorted(test, key=lambda it: it["id"]):
        by_domain[it["domain"]].append(it["id"])
    blocks = [[] for _ in range(10)]
    for dom in previous.DOMAINS:
        rng.shuffle(by_domain[dom])
        for index, pid in enumerate(by_domain[dom]):
            blocks[index % 10].append(pid)
    for block in blocks:
        rng.shuffle(block)
    rng.shuffle(blocks)
    pilot_order = sorted(it["id"] for it in pilot)
    rng.shuffle(pilot_order)
    return {"planning_seed": SEED, "blocks": blocks,
            "order": [pid for block in blocks for pid in block], "pilot_order": pilot_order}


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repairs = parser.add_mutually_exclusive_group()
    repairs.add_argument("--repair-before-inference", action="store_true",
                        help="archive and repair the rejected initial candidate only if results/v2 has no runtime files")
    repairs.add_argument("--restore-preregistered-seed", action="store_true",
                        help="one-off: archive the unauthorized seed20261002 candidate and restore required seed20261001")
    args = parser.parse_args()
    initial_audit = None
    seed_audit = None
    if args.restore_preregistered_seed:
        seed_audit = repair_before_inference(restore_seed=True)
        initial_audit = seed_audit["initial_money_rejection_audit"]
    elif args.repair_before_inference:
        initial_audit = repair_before_inference()
    elif (OUT / "SHA256SUMS").exists():
        raise SystemExit("data/v2 is frozen; refusing to overwrite immutable artifacts")
    elif (ROOT / "results" / "v2" / "preparation" / "dataset_initial_rejected.json").exists():
        raise SystemExit("Incomplete pre-inference repair; recover from the preserved archive, never rebuild without audit")
    old_inputs, old_gold = load_previous()
    # Check the old freeze before reading it; never call the old downloader or builder.
    for line in (ROOT / "data" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        if previous.sha256_file(ROOT / "data" / name.strip()) != digest:
            raise ValueError(f"Previous frozen artifact changed: {name}")
    pool, source_hashes = load_pool()
    meta = OUT / "model_meta" / "G4"
    meta.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        path = meta / name
        if not path.exists():
            with urlopen(f"https://huggingface.co/{G4[0]}/resolve/{G4[1]}/{name}", timeout=120) as response:
                path.write_bytes(response.read())
    count_tokens = token_counter()
    chosen, audit = select(pool, old_inputs, old_gold, count_tokens)
    inputs, gold = {}, {}
    for split in ("test", "pilot"):
        items = sorted((it for it in chosen if it["split"] == split), key=lambda it: it["id"])
        inputs[split] = [{"id": it["id"], "problem": it["problem"], "language": "en"} for it in items]
        gold[split] = [gold_record(it) for it in items]
    # Preserve dev IDs, statements, gold, and order. Only refresh G4/B token counts.
    inputs["dev"] = old_inputs[100:]
    gold["dev"] = [dict(g, prompt_tokens=count_tokens(i["problem"]), reviewed=False)
                   for i, g in zip(inputs["dev"], old_gold[100:])]
    for split in SPLITS:
        write_jsonl(OUT / f"{split}_inputs.jsonl", inputs[split])
        write_jsonl(OUT / f"{split}_gold.jsonl", gold[split])
    previous.write_json(OUT / "schedule.json", schedule_for(gold["test"], gold["pilot"]))
    rrng = random.Random(SEED + 11)
    review_ids = []
    for dom in previous.DOMAINS:
        ids = sorted(g["id"] for g in gold["test"] if g["domain"] == dom)
        review_ids.extend(sorted(rrng.sample(ids, min(4, len(ids)))))
    previous.write_json(OUT / "blind_review_ids.json", {
        "seed": SEED + 11, "ids": review_ids, "reviewed": False,
        "note": "Preregistered before inference; human review waived by user, not completed."})
    with (OUT / "review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "split", "domain", "difficulty", "source", "problem", "gold_answer",
                         "reviewed", "unique_answer_ok", "statement_ok", "solution_ok", "difficulty_ok",
                         "reviewer", "notes"])
        for split in SPLITS:
            for i, g in zip(inputs[split], gold[split]):
                writer.writerow([g["id"], split, g["domain"], g["difficulty"], g["source"], i["problem"],
                                 g["gold_answer"], "false", "", "", "", "", "", "Human review waived"])
    old_names = ("test_inputs.jsonl", "test_gold.jsonl", "dev_inputs.jsonl", "dev_gold.jsonl",
                 "schedule.json", "dataset_manifest.json", "SHA256SUMS", "review_sheet.csv",
                 "blind_review_ids.json", "build_dataset.py")
    manifest = {
        "dataset_version": "v2", "seed": SEED, **audit,
        "sources": {key: {"repo": repo, "revision": rev, "files": names}
                    for key, (repo, rev, names) in previous.SOURCES.items()},
        "source_sha256": source_hashes,
        "builder_sha256": previous.sha256_file(Path(__file__)),
        "previous_artifact_sha256": {name: previous.sha256_file(ROOT / "data" / name) for name in old_names},
        "previous_exclusion": {"count": 120, "ids": sorted(i["id"] for i in old_inputs),
                               "template_groups": sorted(g["template_group"] for g in old_gold),
                               "dev_reuse_does_not_remove_exclusions": True},
        "near_duplicate": {"normalization": "lowercase ASCII alphanumeric words", "n": 8,
                           "threshold": 0.5, "denominator": "minimum unique 8-gram count",
                           "shorter_than_8_words": "no grams: false, same as previous builder"},
        "domains": previous.DOMAINS, "quota": {"test": previous.TEST_QUOTA, "pilot": PILOT_QUOTA},
        "hard_source_preference": HARD_PREF,
        "ratios_hard_fallback": "competition first, then MATH L4-5, then GSM8K >=7 steps; "
                                "GSM8K explicitly allowed, no reclassification",
        "pilot_policy": "After all test strata: per domain take 4 easy FIRST, then 4 medium, "
                        "then remaining easy to reach 8 if medium is scarce; never reclassify difficulty. "
                        "Preregistered before inference; publish reduced N if easy fallback is insufficient.",
        "dev_policy": "Reuse previous 20 only for debugging/warmup/latency; excluded from new selection.",
        "max_prompt_tokens": 512, "max_abs_answer": 999,
        "tokenizers": {"G4": {"repo": G4[0], "revision": G4[1], "sha256": {
            name: previous.sha256_file(meta / name) for name in META_FILES}},
            "B": {"repo": "DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking",
                  "revision": "717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe", "sha256": {
                      name: previous.sha256_file(previous.RAW / "model_meta" / "B" / name) for name in META_FILES}}},
        "generator_prompt_sha256": previous.sha256_file(ROOT / "prompts" / "generador.txt"),
        "enable_thinking": False, "gold_money_percent_explicitly_excluded": True,
        "monetary_filter_correction": correction_policy(),
        "initial_candidate_audit": initial_audit,
        "initial_candidate_audit_sha256": previous.sha256_file(
            ROOT / "results" / "v2" / "preparation" / "dataset_initial_rejected.json") if initial_audit else None,
        "seed_restoration_audit": seed_audit,
        "seed_restoration_audit_sha256": previous.sha256_file(
            ROOT / "results" / "v2" / "preparation" / "dataset_wrong_seed_rejected.json") if seed_audit else None,
        "actual_n": {split: len(gold[split]) for split in SPLITS},
        "counts": {split: dict(Counter(f"{g['domain']}/{g['difficulty']}/"
                                       f"{'competition' if 'AMC' in g['source'] or 'AIME' in g['source'] or 'aimo' in g['source'] else 'gsm8k' if g['source'] == 'openai/gsm8k' else 'math'}"
                                       for g in gold[split])) for split in SPLITS},
        "strata": {split: {dom: {diff: sum(g["domain"] == dom and g["difficulty"] == diff
                                           for g in gold[split]) for diff in previous.TEST_QUOTA}
                           for dom in previous.DOMAINS} for split in SPLITS},
        "shortfalls": [a for a in audit["selection_attempts"] if a["selected"] < a["requested"]],
        "prompt_tokens_max": {split: max((max(g["prompt_tokens"].values()) for g in gold[split]), default=0)
                              for split in SPLITS},
        "review": {"reviewed": False, "human_review_waived": True, "blind_ids_preregistered": True,
                   "limitation": "No completed human or blind secondary review; not claimed as reviewed."},
        "input_schema": ["id", "problem", "language"],
        "schedule_schema": "order and blocks cover test only; pilot_order covers pilot only",
        "immutable": "SHA256SUMS seals every generated artifact except itself; builder refuses a sealed rebuild; files read-only.",
    }
    if initial_audit:
        no_runtime_execution()
    previous.write_json(OUT / "dataset_manifest.json", manifest)
    artifacts = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (OUT / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for path in artifacts:
            handle.write(f"{previous.sha256_file(path)}  {path.relative_to(OUT).as_posix()}\n")
    for path in artifacts + [OUT / "SHA256SUMS"]:
        path.chmod(0o444)
    print(json.dumps({key: manifest[key] for key in ("actual_n", "strata", "shortfalls", "prompt_tokens_max")}, indent=2))


if __name__ == "__main__":
    main()
