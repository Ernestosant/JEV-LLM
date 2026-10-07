"""Build isolated v3 bundles; full publication requires actual sealed reviews."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/experiment_v3.json"
OFFICIAL_Q9_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OPERATIONAL_POLICY_ID = "jev-v3-balanced-residencies-25h-20261006"
REVIEW_POLICY_ID = "jev-v3-source-sampling-tier-agent-review-20261004"
REVIEW_CONTRACT_SHA256 = "4ad2d4bbae07425864bcaa3fa5c99f8aa140c8feeaf902955a6bc8041c5689cb"
PUBLIC_DATA = ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl",
               "latency_inputs.jsonl", "schedule.json", "latency_plan.json",
               "dataset_manifest.json", "REVIEW_CONTRACT.md", "review_policy.json", "ELIGIBILITY_POLICY.md", "SHA256SUMS")
GOLD_DATA = tuple(f"{s}_gold.jsonl" for s in ("test", "pilot", "dev"))
SHARED_MODULES = ("__init__", "algorithms", "analysis", "common", "engines", "evaluate", "hw", "hwprobe",
                  "latency_study", "logutil", "models", "native_ref", "parsing", "preflight", "prompts", "runner", "seeds", "selector")
TOOL_SOURCES = ("tools/build_notebooks_v3.py", "tools/make_bundle_v3.py", "tools/colab/run_v3.py",
                "tools/colab/launch_v3.sh", "tools/colab/operator_v3.sh", "tools/colab/session_guard.sh",
                "tools/colab/session_guard.py", "tools/colab/run_v2.py", "tools/verify_run.py",
                "tools/build_notebooks.py", "tools/pilot_decision_v3.py")
_DATASET_CERTIFICATES = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"Unsafe member/source path: {name!r}")
    return name


def source_bytes(root, name):
    safe_name(name)
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Unsafe source: {name}")
    if not path.is_file():
        raise ValueError(f"Missing required v3 source: {name}")
    return path.read_bytes()


def input_rows(raw, count, name):
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if (len(rows) != count or any(not isinstance(r, dict) or set(r) != {"id", "problem", "language"}
            or r["language"] != "en" or any(not isinstance(v, str) or not v.strip() for v in r.values()) for r in rows)
            or len({r["id"] for r in rows}) != count):
        raise ValueError(f"Invalid blind inputs (expected {count}): {name}")
    return rows


def validate_config(cfg):
    if (cfg.get("protocol_version") != "3" or cfg.get("code_version") != "0.3.0"
            or cfg.get("implementation_profile") != "1COPY-G4-VLLM-GRAPH-V3"):
        raise ValueError("Requires exact v3 config/code/profile")
    models = cfg.get("models", {})
    if (set(models) != {"G", "J", "B", "O"} or any(not m.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", m.get("revision", "")) for m in models.values())
            or models["O"].get("repo") != "Qwen/Qwen3.5-9B" or models["O"].get("revision") != OFFICIAL_Q9_REVISION):
        raise ValueError("Requires the exact pinned official Qwen3.5-9B revision and four frozen model roles")


def _validate_dataset_full(root=ROOT):
    """Verify original full seals and actual review bindings without fetching anything.

    Return the actual public manifest plus the original seal inventory. A tooling
    wrapper must not repair a publisher contract or invent a replacement seal.
    """
    root = Path(root).resolve()
    try:
        validate_config(json.loads(source_bytes(root, CONFIG)))
        seal_raw = source_bytes(root, "data/v3/SHA256SUMS")
        seal_names = set()
        for line in seal_raw.decode().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            safe_name(name)
            if name in seal_names or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Malformed/duplicate actual dataset seal")
            seal_names.add(name)
        builder_path = root / "data/build_dataset_v3.py"
        builder_raw = source_bytes(root, "data/build_dataset_v3.py")
        spec = importlib.util.spec_from_file_location("bundle_dataset_v3", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        builder.ROOT, builder.OUT = root, root / "data/v3"
        directory = builder.OUT
        seals = builder.verify_seal(directory)
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                  if p.is_file() and p != directory / "SHA256SUMS"}
        if actual != set(seals) or not seals:
            raise ValueError("Full dataset contains unsealed/missing members")
        for name in seals:
            source_bytes(root, "data/v3/" + name)
        original = builder.read_json(directory / "dataset_manifest.json")
        if (original.get("dataset_version") != "v3" or original.get("sealed") is not True
                or original.get("actual_n") != {"test": 500, "pilot": 50, "dev": 20, "latency": 100}
                or original.get("review", {}).get("agent_reviewed_all") is not True
                or original["review"].get("human_reviewed") is not False):
            raise ValueError("Requires actual runtime-compatible sealed public v3 manifest and agent reviews; no schema projection")
        public_raw = {"data/v3/" + n: source_bytes(root, "data/v3/" + n) for n in PUBLIC_DATA}
        validate_public(public_raw, False)
        cfg = json.loads(source_bytes(root, CONFIG))
        bound = cfg.get("dataset_review_policy")
        if bound and (bound.get("id") != original["review"]["policy_id"]
                      or bound.get("sha256") != original["review"]["policy_sha256"]):
            raise ValueError("Actual sealed review policy differs from pinned configuration")
        stage = directory / "staging"
        builder.verify_seal(stage)
        metadata = builder.read_json(stage / "preparation_manifest.json")
        protected = builder.verify_preparation_lineage(stage, metadata)
        if (metadata["builder_sha256"] != sha(builder_raw)
                or metadata["construction_seed"] != builder.SEED or metadata["latency_seed"] != builder.LATENCY_SEED):
            raise ValueError("Dataset builder/seed binding changed")
        policy = json.loads(source_bytes(root, "data/v3/review_policy.json"))
        if (source_bytes(root, "data/v3/review_policy.json") != source_bytes(root, "data/v3/staging/review_policy.json")
                or policy != builder.review_policy() or sha(source_bytes(root, "data/v3/review_policy.json")) != metadata["review_policy_json_sha256"]):
            raise ValueError("Declared pre-review policy changed")
        lineage_inventory = {**protected, **metadata["source_sha256"], metadata["source_lock_path"]: metadata["source_lock_sha256"]}
        lineage_inventory["data/build_dataset_v3.py"] = sha(builder_raw)
        for inventory in (protected, metadata["source_sha256"]):
            if not inventory:
                raise ValueError("Missing original protected/source inventory")
            for name, expected in inventory.items():
                if sha(source_bytes(root, name)) != expected:
                    raise ValueError(f"Original dataset evidence changed: {name}")
        for tok in metadata["tokenizers"].values():
            for name, expected in tok["sha256"].items():
                source = tok["directory"] + "/" + name
                if sha(source_bytes(root, source)) != expected:
                    raise ValueError("Tokenizer evidence changed")
                lineage_inventory[source] = expected
        if sha(source_bytes(root, metadata["source_lock_path"])) != metadata["source_lock_sha256"]:
            raise ValueError("Original source lock evidence changed")
        if sha(source_bytes(root, "prompts/generador.txt")) != metadata["generator_prompt_sha256"]:
            raise ValueError("Dataset generator prompt changed")
        rows = builder.read_jsonl(stage / "candidate_pool.jsonl")
        if any(r["candidate_sha256"] != builder.candidate_sha(r) for r in rows):
            raise ValueError("Candidate hash changed")
        packet_rows = []
        for packet in metadata["packets"]:
            blind = source_bytes(root, packet["blind"])
            reference = source_bytes(root, packet["reference"])
            if sha(blind) != packet["blind_sha256"] or sha(reference) != packet["reference_sha256"]:
                raise ValueError("Original review packet binding changed")
            refs = [json.loads(l) for l in reference.decode().splitlines() if l.strip()]
            if ([r["id"] for r in refs] != packet["candidate_ids"]
                    or [json.loads(l) for l in blind.decode().splitlines() if l.strip()] !=
                    [{k: r[k] for k in ("id", "problem", "language")} for r in refs]):
                raise ValueError("Original blind/reference packet content changed")
            packet_rows.extend(refs)
        if packet_rows != rows:
            raise ValueError("Review packet pool differs from original candidates")
        approved, decisions, evidence = builder.load_reviews(rows, stage, metadata["packets"])
        for name, expected in evidence.items():
            if not name.startswith("data/v3/") or seals.get(name.removeprefix("data/v3/")) != expected:
                raise ValueError("Actual review evidence not bound by the original dataset seal")
        selected = builder.select_approved(rows, approved, decisions)
        inputs, gold = {}, {}
        for split, count in (("test", 500), ("pilot", 50), ("dev", 20), ("latency", 100)):
            inputs[split] = input_rows(source_bytes(root, f"data/v3/{split}_inputs.jsonl"), count, split)
            if split in ("test", "pilot", "dev"):
                gold[split] = builder.read_jsonl(directory / f"{split}_gold.jsonl")
                if [r["id"] for r in gold[split]] != [r["id"] for r in inputs[split]]:
                    raise ValueError("Gold/input alignment changed")
            if split in selected:
                expected = {r["id"]: r for r in selected[split]}
                if {r["candidate_id"] for r in gold[split]} != set(expected):
                    raise ValueError("Frozen selection differs from actual review decisions")
                for i, g in zip(inputs[split], gold[split]):
                    candidate = expected[g["candidate_id"]]
                    decision = decisions[candidate["id"]]
                    if (g.get("agent_reviews") != decision["reviews"]
                            or g.get("review_decision") != decision
                            or g.get("accepted_via_adjudication") != decision["accepted_via_adjudication"]
                            or i["problem"] != candidate["problem"]
                            or any(g.get(k) != v for k, v in candidate.items() if k != "id")
                            or g.get("domain") != decision["final_domain"]
                            or g.get("difficulty") != decision["final_difficulty"]
                            or g.get("sampling_tier") != candidate["provisional_difficulty"]
                            or decision.get("sampling_tier") != candidate["provisional_difficulty"]
                            or g.get("agent_reviewed") is not True or g.get("human_reviewed") is not False):
                        raise ValueError("Frozen gold differs from reviewed candidate")
        intrinsic_counts = {s: {label: sum(g["difficulty"] == label for g in gold[s])
                                for label in ("easy", "medium", "hard")} for s in ("test", "pilot")}
        if original["review"]["intrinsic_difficulty_counts"] != intrinsic_counts:
            raise ValueError("Public intrinsic difficulty counts differ from actual selected gold")
        if inputs["dev"] != input_rows(source_bytes(root, "data/dev_inputs.jsonl"), 20, "original dev"):
            raise ValueError("Development inputs differ from original20")
        test = {r["id"]: r for r in inputs["test"]}
        if any(test.get(r["id"]) != r for r in inputs["latency"]):
            raise ValueError("Latency subset differs from test")
        if [r["id"] for r in inputs["latency"]] != builder.latency_subset(gold["test"]):
            raise ValueError("Latency subset differs from frozen seed/strata")
        schedule = builder.read_json(directory / "schedule.json")
        if schedule != builder.schedule_for(gold["test"], gold["pilot"]):
            raise ValueError("Frozen balanced schedule changed")
        if builder.read_json(directory / "latency_plan.json") != builder.latency_plan_for(gold["test"], [r["id"] for r in inputs["latency"]], schedule):
            raise ValueError("Frozen schedule-bound latency plan changed")
        if original.get("analysis_manifest_sha256") != sha(source_bytes(root, "data/v3/review_manifest.json")):
            raise ValueError("Sealed private review provenance commitment changed")
        return {"manifest": original, "seals": seals, "lineage_inventory": lineage_inventory}
    except (OSError, KeyError, TypeError, ImportError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot validate actual sealed v3 dataset: {exc}") from exc


def dataset_fingerprint(root, sources):
    """Content hashes and complete dataset path inventory, never mtime/size trust."""
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Symlink certificate root forbidden")
    names, directories = set(sources), set()
    names.update(p.relative_to(root).as_posix() for p in (root / "src/jevlab").rglob("*.py"))
    names.update(p.relative_to(root).as_posix() for p in (root / "data").glob("*.py"))
    for directory, dirs, files in os.walk(root / "data/v3", followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Symlink dataset directory forbidden")
            directories.add(path.relative_to(root).as_posix())
        names.update((Path(directory) / name).relative_to(root).as_posix() for name in files)
    parents = {p for n in names for p in (root / safe_name(n)).parents if p != root and p.is_relative_to(root)}
    if any(p.is_symlink() for p in parents):
        raise ValueError("Symlink certificate source parent forbidden")
    def fingerprint(name):
        path = root / name
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise ValueError("Non-regular certificate source: " + name)
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return name, value.hexdigest()
    with ThreadPoolExecutor(max_workers=4) as workers:
        hashes = dict(workers.map(fingerprint, sorted(names)))
    return hashes, directories


def validate_dataset(root=ROOT):
    """Full proof once per process; every reuse rehashes all proof inputs and paths."""
    root = Path(root).absolute()
    key = str(root)
    certificate = _DATASET_CERTIFICATES.get(key)
    if certificate is not None:
        if dataset_fingerprint(root, certificate["sources"]) != certificate["fingerprint"]:
            raise ValueError("Dataset certificate bytes/path inventory changed; no cached approval")
        return copy.deepcopy(certificate["validated"])
    # Read only the preparation descriptor to enumerate external proof inputs.
    source_bytes(root, "data/v3/SHA256SUMS")
    if not (root / "data/v3/staging/preparation_manifest.json").is_file():
        return _validate_dataset_full(root)  # Preserve fail-closed diagnostics for incomplete/unsealed roots.
    metadata = json.loads(source_bytes(root, "data/v3/staging/preparation_manifest.json"))
    sources = {CONFIG, "data/build_dataset_v3.py", "data/build_dataset.py", "data/dev_inputs.jsonl", "prompts/generador.txt"}
    sources.update(TOOL_SOURCES)
    sources.update(metadata["source_sha256"])
    sources.update(metadata["previous_artifact_sha256"])
    sources.add(metadata["source_lock_path"])
    sources.update(p.relative_to(root).as_posix() for p in (root / "src/jevlab").rglob("*.py"))
    sources.update(p.relative_to(root).as_posix() for p in (root / "data").glob("*.py"))
    for tokenizer in metadata["tokenizers"].values():
        sources.update(tokenizer["directory"] + "/" + name for name in tokenizer["sha256"])
    before = dataset_fingerprint(root, sources)
    validated = _validate_dataset_full(root)
    after = dataset_fingerprint(root, sources)
    expected = {"data/v3/" + n: h for n, h in validated["seals"].items()}
    expected.update(validated["lineage_inventory"])
    if before != after or any(after[0].get(n) != h for n, h in expected.items()):
        raise ValueError("Dataset proof changed during full certification")
    _DATASET_CERTIFICATES[key] = {"sources": sources, "fingerprint": after, "validated": copy.deepcopy(validated)}
    return validated


def _payload(root, gold, dev):
    cfg_raw = source_bytes(root, CONFIG)
    cfg = json.loads(cfg_raw)
    validate_config(cfg)
    shared = sorted((root / "src/jevlab").glob("*.py"))
    v3 = sorted((root / "src/jevlab/v3").rglob("*.py"))
    if not shared or not v3:
        raise ValueError("Missing shared parent/v3 source")
    names = [p.relative_to(root).as_posix() for p in shared + v3]
    names += [CONFIG, "tools/verify_run_v3.py"] + [f"prompts/{n}.txt" for n in
                                                          ("generador", "criterio_paso", "criterio_final")]
    payload = {n: source_bytes(root, n) for n in names}
    mapping = {n: n for n in names}
    if dev:
        if gold:
            raise ValueError("Development bundles never contain gold")
        original = next((n for n in ("data/v3/staging/dev_reuse_inputs.jsonl", "data/v3/staging/dev_inputs.jsonl")
                         if (root / n).is_file()), "data/dev_inputs.jsonl")
        raw = source_bytes(root, original)
        input_rows(raw, 20, original)
        if input_rows(raw, 20, original) != input_rows(source_bytes(root, "data/dev_inputs.jsonl"), 20, "original dev"):
            raise ValueError("Development staging inputs differ from the original20")
        payload["data/v3/dev_inputs.jsonl"] = raw
        mapping["data/v3/dev_inputs.jsonl"] = original
        payload["data/v3/dataset_manifest.json"] = encoded({"dataset_version": "v3-dev", "sealed": False,
            "development_only": True, "actual_n": {"dev": 20}, "review": {"agent_reviewed_all": False,
            "human_reviewed": False}, "input_sha256": {"dev_inputs.jsonl": sha(raw)}})
    else:
        validated = validate_dataset(root)
        for name in PUBLIC_DATA:
            member = "data/v3/" + name
            payload[member] = source_bytes(root, member)
            mapping[member] = member
        for name in GOLD_DATA:
            if gold:
                payload["data/v3/" + name] = source_bytes(root, "data/v3/" + name)
                mapping["data/v3/" + name] = "data/v3/" + name
    sources = set(mapping.values())
    sources.update(TOOL_SOURCES)
    if not dev:
        sources.update("data/v3/" + n for n in validated["seals"])
        sources.add("data/v3/SHA256SUMS")
        sources.update(validated["lineage_inventory"])
    manifest = {"series": "v3", "code_version": "0.3.0", "legacy_code_version": "0.2.1",
                "config_sha256": sha(cfg_raw), "implementation_profile": cfg["implementation_profile"],
                "models": cfg["models"], "jevk5_runtime": cfg["jevk5_runtime"], "software_pins": cfg["software_pins"],
                "development_only": dev, "contains_gold": gold, "source_mapping": mapping,
                "source_sha256": {n: sha(source_bytes(root, n)) for n in sorted(sources)},
                 "files": {n: sha(v) for n, v in sorted(payload.items())}}
    operational = "src/jevlab/v3/operational.py"
    if operational in payload:
        manifest.update(operational_policy_id=OPERATIONAL_POLICY_ID, operational_module_sha256=sha(payload[operational]))
    return payload, manifest


def validate_public(raw, dev):
    """Check the actual public dataset contract and sealed blind strata."""
    metadata = json.loads(raw["data/v3/dataset_manifest.json"])
    allowed = {"dataset_version", "sealed", "status", "actual_n", "construction_seed", "latency_seed",
               "review", "input_sha256", "original_manifest_sha256", "original_seal_sha256", "development_only",
               "requested_n", "quota", "builder_sha256", "analysis_manifest_sha256", "policy_artifact_sha256",
                "source_revisions", "test_reserved_before_pilot", "latency_test_subset_only", "max_prompt_tokens", "max_abs_answer", "quota_axis"}
    if not set(metadata) <= allowed or not set(metadata.get("review", {})) <= {
            "agent_reviewed_all", "human_reviewed", "dev_agent_reviewed", "required_independent_actual_agents",
            "policy_id", "policy_sha256", "reviewed", "scope", "intrinsic_difficulty_counts", "source_sampling_tier_is_intrinsic_difficulty"}:
        raise ValueError("Private/unapproved fields in public dataset manifest")
    for name in ("builder_sha256", "analysis_manifest_sha256", "original_manifest_sha256", "original_seal_sha256"):
        if name in metadata and (not isinstance(metadata[name], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[name])):
            raise ValueError("Public provenance commitments must be SHA-256 strings")
    for source in metadata.get("source_revisions", {}).values():
        if not isinstance(source, dict) or not set(source) <= {"repo", "revision", "license"} or any(not isinstance(v, str) for v in source.values()):
            raise ValueError("Private/unapproved source revision payload")
    if ("requested_n" in metadata and metadata["requested_n"] != {"test": 500, "pilot": 50, "dev": 20, "latency": 100}
            or "quota" in metadata and metadata["quota"] != {"test": {"easy": 30, "medium": 40, "hard": 30},
                "pilot": {"easy": 3, "medium": 4, "hard": 3}, "latency": {"easy": 6, "medium": 8, "hard": 6}}):
        raise ValueError("Public quota/count contract changed")
    if dev:
        if (set(metadata) != {"dataset_version", "sealed", "development_only", "actual_n", "review", "input_sha256"}
                or set(metadata.get("review", {})) != {"agent_reviewed_all", "human_reviewed"}
                or metadata.get("sealed") is not False or metadata.get("development_only") is not True
                or metadata.get("dataset_version") != "v3-dev" or metadata.get("actual_n") != {"dev": 20}
                or metadata.get("review", {}).get("agent_reviewed_all") is not False):
            raise ValueError("Development manifest must not claim full/sealed data")
    else:
        review = metadata.get("review", {})
        if (metadata.get("sealed") is not True or metadata.get("dataset_version") != "v3"
                or metadata.get("actual_n") != {"test": 500, "pilot": 50, "dev": 20, "latency": 100}
                or review.get("agent_reviewed_all") is not True or review.get("human_reviewed") is not False
                or review.get("policy_id") != REVIEW_POLICY_ID or review.get("policy_sha256") != REVIEW_CONTRACT_SHA256
                or review["policy_sha256"] != sha(raw["data/v3/REVIEW_CONTRACT.md"])
                or metadata.get("quota_axis") != "source_sampling_tier"
                or review.get("source_sampling_tier_is_intrinsic_difficulty") is not False):
            raise ValueError("Invalid sealed public v3 dataset/review contract")
        counts = review.get("intrinsic_difficulty_counts")
        if (not isinstance(counts, dict) or set(counts) != {"test", "pilot"} or any(
                not isinstance(counts[s], dict) or set(counts[s]) != {"easy", "medium", "hard"}
                or any(type(n) is not int or n < 0 for n in counts[s].values())
                or sum(counts[s].values()) != count for s, count in (("test", 500), ("pilot", 50)))):
            raise ValueError("Public publication requires actual intrinsic difficulty counts, not source quotas")
        if ("reviewed" in review and review["reviewed"] is not False or "scope" in review and review["scope"] != ["test", "pilot"]
                or "dev_agent_reviewed" in review and review["dev_agent_reviewed"] is not False):
            raise ValueError("Public review scope must not claim human/dev review")
        policy = json.loads(raw["data/v3/review_policy.json"])
        if not isinstance(policy, dict) or set(policy) != {"policy_id", "contract", "contract_sha256", "eligibility_document_sha256", "method",
                "required_independent_reviewers", "human_review_required", "reviewed_field_means_human_review",
                "dev_reuse_not_new_agent_review", "domain_policy", "difficulty_policy", "adjudication", "eligibility", "seeds", "quota", "quota_axis", "sampling_tier", "scope_integrity"}:
            raise ValueError("Private/unapproved policy payload")
        # Match the publisher's nested public shape, not arbitrary dictionaries.
        shapes = {
            "scope_integrity": {"rule_id", "incident_ledger", "threshold", "reason", "original_verdicts_and_registrations_immutable", "replacement_reviews_cannot_reverse_disqualification"},
            "domain_policy": {"version", "applies_to", "precedence", "ratio_boundaries", "geometry_boundaries", "arithmetic", "regex_evidence"},
            "difficulty_policy": {"gsm8k", "math", "asdiv", "competition", "openstax", "disagreement"},
            "adjudication": {"required_actual_independent_agents", "original_reviews_immutable", "final_labels", "units", "gold", "vetoes"},
            "eligibility": {"language", "exact_integer_fraction_finite_decimal", "max_abs_answer", "max_rendered_tokens_each_pinned_G4_B13_O9",
                            "money_and_percent_allowed", "dollar_math_delimiters_not_currency_exclusions", "units", "non_scalar", "publisher_extraction"},
            "seeds": {"construction", "latency"}, "quota": {"test", "pilot", "latency"}}
        for key, keys in shapes.items():
            if not isinstance(policy.get(key), dict) or set(policy[key]) != keys:
                raise ValueError("Private/unapproved nested policy payload: " + key)
        for key, value in policy.items():
            if key in shapes:
                continue
            expected_type = (bool if key in {"human_review_required", "reviewed_field_means_human_review",
                "dev_reuse_not_new_agent_review"} else int if key == "required_independent_reviewers" else str)
            if type(value) is not expected_type:
                raise ValueError("Private/unapproved policy value")
        regex = policy["domain_policy"]["regex_evidence"]
        if (not isinstance(regex, dict) or set(regex) != {"counting", "number_theory", "formal_algebra", "quantities"}
                or any(not isinstance(v, str) for v in regex.values())
                or not isinstance(policy["domain_policy"]["precedence"], list)
                or any(not isinstance(v, str) for v in policy["domain_policy"]["precedence"])):
            raise ValueError("Private/unapproved nested domain policy payload")
        for key in ("domain_policy", "difficulty_policy", "adjudication", "eligibility", "seeds", "scope_integrity"):
            for name, value in policy[key].items():
                if key == "domain_policy" and name in {"regex_evidence", "precedence"}:
                    continue
                expected_type = (bool if name in {"original_reviews_immutable", "exact_integer_fraction_finite_decimal",
                    "money_and_percent_allowed", "dollar_math_delimiters_not_currency_exclusions",
                    "original_verdicts_and_registrations_immutable", "replacement_reviews_cannot_reverse_disqualification"} else
                    int if name in {"required_actual_independent_agents", "max_abs_answer", "max_rendered_tokens_each_pinned_G4_B13_O9",
                                    "construction", "latency"} else str)
                if type(value) is not expected_type:
                    raise ValueError("Private/unapproved nested policy value")
        if (policy["quota_axis"] != "source_sampling_tier" or not policy["sampling_tier"].strip()
                or policy["scope_integrity"]["rule_id"] != "assigned-input-integrity-v1"
                or policy["scope_integrity"]["incident_ledger"] != "reviews/review_scope_incidents.jsonl"
                or policy["scope_integrity"]["reason"] != "review_scope_breach"
                or not policy["scope_integrity"]["threshold"].strip()
                or policy["scope_integrity"]["original_verdicts_and_registrations_immutable"] is not True
                or policy["scope_integrity"]["replacement_reviews_cannot_reverse_disqualification"] is not True
                or policy["quota"] != {"test": {"easy": 30, "medium": 40, "hard": 30},
                "pilot": {"easy": 3, "medium": 4, "hard": 3}, "latency": {"easy": 6, "medium": 8, "hard": 6}}
                or policy["seeds"] != {"construction": 20261002, "latency": 20261003}
                or policy["adjudication"]["required_actual_independent_agents"] != 1
                or policy["adjudication"]["original_reviews_immutable"] is not True
                or policy["required_independent_reviewers"] != 2 or policy["human_review_required"] is not False
                or policy["reviewed_field_means_human_review"] is not True or policy["dev_reuse_not_new_agent_review"] is not True
                or policy["eligibility"]["language"] != "en" or policy["eligibility"]["max_abs_answer"] != 10**9
                or policy["eligibility"]["max_rendered_tokens_each_pinned_G4_B13_O9"] != 1024
                or any(policy["eligibility"][k] is not True for k in ("exact_integer_fraction_finite_decimal",
                    "money_and_percent_allowed", "dollar_math_delimiters_not_currency_exclusions"))):
            raise ValueError("Current public policy controls changed")
        if (policy.get("policy_id") != review["policy_id"] or policy.get("contract") != "REVIEW_CONTRACT.md"
                or policy.get("contract_sha256") != review["policy_sha256"]
                or policy.get("eligibility_document_sha256") != sha(raw["data/v3/ELIGIBILITY_POLICY.md"])):
            raise ValueError("Public policy identity/hash mismatch")
        policy_hashes = metadata.get("policy_artifact_sha256", {})
        if policy_hashes and (set(policy_hashes) != {"review_policy.json", "ELIGIBILITY_POLICY.md"}
                or any(h != sha(raw["data/v3/" + n]) for n, h in policy_hashes.items())):
            raise ValueError("Public policy artifact commitments changed")
        seals = {}
        for line in raw["data/v3/SHA256SUMS"].decode().splitlines():
            digest, name = line.split(maxsplit=1)
            safe_name(name)
            if name in seals or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid public seal entry")
            seals[name] = digest
        expected = set(PUBLIC_DATA) - {"SHA256SUMS"} | set(GOLD_DATA)
        if not expected <= set(seals):
            raise ValueError("Unexpected/missing public seal entries")
        for name, digest in seals.items():
            member = "data/v3/" + name
            if member in raw and sha(raw[member]) != digest:
                raise ValueError(f"Public seal mismatch: {name}")
        test = input_rows(raw["data/v3/test_inputs.jsonl"], 500, "test")
        pilot = input_rows(raw["data/v3/pilot_inputs.jsonl"], 50, "pilot")
        dev_rows = input_rows(raw["data/v3/dev_inputs.jsonl"], 20, "dev")
        groups = [{r["id"] for r in rows} for rows in (test, pilot, dev_rows)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Overlapping public split IDs")
        schedule = json.loads(raw["data/v3/schedule.json"])
        if not set(schedule) <= {"planning_seed", "order", "pilot_order", "blocks", "pilot_blocks"}:
            raise ValueError("Private/unapproved schedule fields")
        for rows, key, blocks_key, block_count in ((test, "order", "blocks", 50), (pilot, "pilot_order", "pilot_blocks", 5)):
            order = schedule.get(key, [])
            if len(order) != len(rows) or len(set(order)) != len(order) or set(order) != {r["id"] for r in rows}:
                raise ValueError("Public schedule mismatch")
            if blocks_key in schedule and (len(schedule[blocks_key]) != block_count or any(len(b) != 10 for b in schedule[blocks_key])
                    or [pid for block in schedule[blocks_key] for pid in block] != order):
                raise ValueError("Public balanced schedule blocks mismatch")
        subset = input_rows(raw["data/v3/latency_inputs.jsonl"], 100, "latency")
        test_by_id = {r["id"]: r for r in test}
        if any(test_by_id.get(r["id"]) != r for r in subset):
            raise ValueError("Latency/test content mismatch")
        plan = json.loads(raw["data/v3/latency_plan.json"])
        if set(plan) != {"items", "dataset_version", "sealed", "selection_seed", "source_split", "split", "blocks",
                         "seeds", "repetitions", "conditions", "generation", "expected_rows", "quota_per_domain", "quota_axis", "intrinsic_difficulty_counts"}:
            raise ValueError("Missing or unexpected/private latency plan fields")
        for key, expected_value in {"dataset_version": "v3", "sealed": True, "selection_seed": 20261003, "source_split": "test",
                "split": "test", "seeds": [17, 29, 43], "repetitions": 3, "conditions": ["JFINAL", "B13_GREEDY"],
                "generation": 0, "expected_rows": 1800, "quota_per_domain": {"easy": 6, "medium": 8, "hard": 6},
                "quota_axis": "source_sampling_tier"}.items():
            if plan[key] != expected_value:
                raise ValueError("Frozen latency plan control mismatch: " + key)
        items = plan["items"]
        if len(items) != 100 or {i["item_id"] for i in items} != {r["id"] for r in subset}:
            raise ValueError("Latency plan coverage mismatch")
        for item in items:
            if (set(item) != {"item_id", "problem", "domain", "difficulty", "sampling_tier"}
                    or item["problem"] != test_by_id[item["item_id"]]["problem"]
                    or item["difficulty"] not in {"easy", "medium", "hard"}
                    or item["sampling_tier"] not in {"easy", "medium", "hard"}):
                raise ValueError("Private fields or changed content in latency plan")
        expected_order = [pid for pid in schedule["order"] if pid in {r["id"] for r in subset}]
        if [i["item_id"] for i in items] != expected_order or plan["blocks"] != [expected_order[i:i + 10] for i in range(0, 100, 10)]:
            raise ValueError("Frozen latency plan block order mismatch")
        domains = {"arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"}
        if {i["domain"] for i in items} != domains or any(
                Counter(i["sampling_tier"] for i in items if i["domain"] == d) != {"easy": 6, "medium": 8, "hard": 6}
                for d in domains):
            raise ValueError("Invalid blind latency source sampling strata")
        latency_counts = plan["intrinsic_difficulty_counts"]
        actual_counts = {label: sum(i["difficulty"] == label for i in items) for label in ("easy", "medium", "hard")}
        if (not isinstance(latency_counts, dict) or set(latency_counts) != set(actual_counts)
                or any(type(n) is not int or n < 0 for n in latency_counts.values()) or latency_counts != actual_counts):
            raise ValueError("Latency intrinsic difficulty counts differ from actual item labels")
        for split, rows in (("test", test), ("pilot", pilot)):
            member = f"data/v3/{split}_gold.jsonl"
            if member not in raw:
                continue
            gold = [json.loads(line) for line in raw[member].decode().splitlines() if line.strip()]
            if (len(gold) != len(rows) or [r["id"] for r in gold] != [r["id"] for r in rows]
                    or any(r.get("difficulty") not in {"easy", "medium", "hard"} for r in gold)
                    or counts[split] != {label: sum(g["difficulty"] == label for g in gold) for label in ("easy", "medium", "hard")}):
                raise ValueError("Public intrinsic difficulty counts differ from analysis gold")
            if split == "test":
                indexed = {r["id"]: r for r in gold}
                if any(any(item[k] != indexed[item["item_id"]].get(k) for k in ("domain", "difficulty", "sampling_tier")) for item in items):
                    raise ValueError("Latency reviewed labels/source tiers differ from analysis gold")
    hashes = metadata.get("input_sha256", {})
    expected = {"dev_inputs.jsonl"} if dev else {"test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl",
                                               "latency_inputs.jsonl", "schedule.json", "latency_plan.json"}
    if set(hashes) != expected or any(sha(raw["data/v3/" + n]) != h for n, h in hashes.items()):
        raise ValueError("Public input hashes mismatch")


def inspect_bundle(path, root=ROOT, check_local=True):
    """Validate ZIP inventory, visibility, pinned metadata and optionally local sources."""
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > 512 * 1024 * 1024:
                raise ValueError("Bundle uncompressed size exceeds 512 MiB")
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP members")
            for info in archive.infolist():
                safe_name(info.orig_filename)
                safe_name(info.filename)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Directory/symlink ZIP member forbidden")
            manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
            if (manifest.get("series") != "v3" or manifest.get("code_version") != "0.3.0"
                    or manifest.get("legacy_code_version") != "0.2.1"):
                raise ValueError("Unknown bundle series/code version")
            files = manifest["files"]
            if not isinstance(files, dict) or set(names) != set(files) | {"BUNDLE_MANIFEST.json"}:
                raise ValueError("Missing hashes or extra/missing ZIP members")
            raw = {}
            for name, digest in files.items():
                safe_name(name)
                raw[name] = archive.read(name)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha(raw[name]) != digest:
                    raise ValueError(f"Bundle hash mismatch: {name}")
            dev, gold = manifest["development_only"], manifest["contains_gold"]
            if type(dev) is not bool or type(gold) is not bool or (dev and gold):
                raise ValueError("Invalid bundle visibility flags")
            allowed_data = {"dev_inputs.jsonl", "dataset_manifest.json"} if dev else set(PUBLIC_DATA) | (set(GOLD_DATA) if gold else set())
            required = {CONFIG, "tools/verify_run_v3.py", "src/jevlab/__init__.py", "src/jevlab/v3/__init__.py"}
            required |= {"src/jevlab/" + n + ".py" for n in SHARED_MODULES}
            required |= {"data/v3/" + n for n in allowed_data}
            required |= {f"src/jevlab/v3/{n}.py" for n in ("runner", "analysis", "algorithms", "latency_study", "preflight")}
            required |= {f"prompts/{n}.txt" for n in ("generador", "criterio_paso", "criterio_final")}
            if not required <= set(files):
                raise ValueError("Missing required v3 bundle members")
            for name in files:
                source = name.startswith("src/jevlab/") and name.endswith(".py") and (
                    name.count("/") == 2 or name.startswith("src/jevlab/v3/"))
                if name not in required and not source:
                    raise ValueError(f"Unapproved bundle member: {name}")
                if name.endswith("_inputs.jsonl"):
                    count = {"test": 500, "pilot": 50, "dev": 20, "latency": 100}[Path(name).stem.removesuffix("_inputs")]
                    input_rows(raw[name], count, name)
            for member, version in (("src/jevlab/__init__.py", "0.2.1"), ("src/jevlab/v3/__init__.py", "0.3.0")):
                versions = [ast.literal_eval(n.value) for n in ast.parse(raw[member]).body
                            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
                if versions != [version]:
                    raise ValueError("Incorrect parent/v3 code version")
            mapping, sources = manifest["source_mapping"], manifest["source_sha256"]
            if not isinstance(mapping, dict) or not isinstance(sources, dict) or not set(mapping) <= set(files):
                raise ValueError("Invalid source mapping/inventory")
            if not set(TOOL_SOURCES) <= set(sources):
                raise ValueError("Missing immutable tooling source hashes")
            for name, digest in sources.items():
                safe_name(name)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid local source hash")
            for member, original in mapping.items():
                safe_name(original)
                if sources.get(original) != files[member]:
                    raise ValueError("Source mapping hash mismatch")
            if not required - ({"data/v3/dataset_manifest.json"} if dev else set()) <= set(mapping):
                raise ValueError("Missing source mapping")
            validate_public(raw, dev)
            if ("src/jevlab/v3/operational.py" in raw or "operational_policy_id" in manifest
                    or "operational_module_sha256" in manifest):
                operational = "src/jevlab/v3/operational.py"
                if (manifest.get("operational_policy_id") != OPERATIONAL_POLICY_ID or operational not in raw
                        or manifest.get("operational_module_sha256") != sha(raw[operational])
                        or mapping.get(operational) != operational):
                    raise ValueError("Operational policy/module source binding mismatch")
            cfg = json.loads(raw[CONFIG])
            validate_config(cfg)
            if not dev:
                review = json.loads(raw["data/v3/dataset_manifest.json"])["review"]
                if cfg.get("dataset_review_policy") != {"id": review["policy_id"], "sha256": review["policy_sha256"]}:
                    raise ValueError("Bundle/config review policy pin mismatch")
            for key, value in (("config_sha256", sha(raw[CONFIG])), ("models", cfg["models"]),
                               ("implementation_profile", cfg["implementation_profile"]),
                               ("jevk5_runtime", cfg["jevk5_runtime"]), ("software_pins", cfg["software_pins"])):
                if manifest.get(key) != value:
                    raise ValueError(f"Bundle/config pin mismatch: {key}")
            if cfg.get("code_version") != "0.3.0" or cfg.get("protocol_version") != "3":
                raise ValueError("Invalid v3 configuration")
            if check_local:
                payload, expected = _payload(Path(root).resolve(), gold, dev)
                if manifest != expected or raw != payload:
                    raise ValueError("Bundle differs from local source integrity/actual public artifacts")
            return manifest
    except (OSError, KeyError, TypeError, UnicodeError, SyntaxError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid v3 bundle: {exc}") from exc


def build(name=None, gold=False, dev=False, root=ROOT, output_dir=None, archive_existing=False):
    root = Path(root).resolve()
    payload, manifest = _payload(root, gold, dev)
    name = name or ("jev_llm_v3_dev_bundle.zip" if dev else
                    "jev_llm_v3_analysis_bundle.zip" if gold else "jev_llm_v3_bundle.zip")
    safe_name(name)
    if "/" in name:
        raise ValueError("Bundle name must be a basename")
    dest = Path(output_dir) if output_dir is not None else root / "dist/v3"
    out = dest / name
    refresh_dev = False
    if out.exists():
        existing = inspect_bundle(out, root, check_local=False)
        if existing != manifest:
            if archive_existing:
                if existing["files"] != manifest["files"]:
                    raise ValueError("Archival tooling rebuild must preserve every packaged experiment byte")
                previous = dest / "archive" / (out.stem + "_" + sha(out.read_bytes()) + ".zip")
                previous.parent.mkdir(exist_ok=True)
                if previous.exists():
                    raise ValueError(f"Prior bundle archive already exists: {previous}")
                out.rename(previous)
            elif not dev or existing.get("development_only") is not True:
                raise ValueError(f"Refusing to overwrite frozen bundle with differing contents: {out}")
            else:
                refresh_dev = True  # Explicit --dev regenerates only this unsealed development artifact.
        else:
            return out
    dest.mkdir(parents=True, exist_ok=True)
    target = out.with_name(out.name + ".pending") if refresh_dev else out
    with target.open("xb") as handle, zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, raw in sorted(payload.items()):
            archive.writestr(member, raw)
        archive.writestr("BUNDLE_MANIFEST.json", encoded(manifest))
    if inspect_bundle(target, root, check_local=False) != manifest:
        raise ValueError("Written bundle differs from the validated payload")
    # Recheck every frozen source after writing; do not replay the identical
    # immutable review/lineage validation a second time in this build.
    if any(sha(source_bytes(root, n)) != h for n, h in manifest["source_sha256"].items()):
        raise ValueError("Source changed while building the frozen bundle")
    if refresh_dev:
        target.replace(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Explicit unsealed original20 dev inputs only")
    parser.add_argument("--gold", action="store_true", help="Offline analysis bundle")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive-existing", action="store_true", help="Archive superseded tooling metadata; packaged experiment bytes must remain identical")
    args = parser.parse_args(argv)
    try:
        print(build(gold=args.gold, dev=args.dev, root=args.root, output_dir=args.output_dir, archive_existing=args.archive_existing))
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "blocked_bundle" if args.dev else "blocked_dataset_review", "error": str(error)[:500]}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
