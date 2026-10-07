import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "v2"
sys.path.insert(0, str(ROOT / "data"))
spec = importlib.util.spec_from_file_location("dataset_v2_builder", ROOT / "data" / "build_dataset_v2.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
old = builder.previous


def jl(split, kind, directory=D):
    return builder.read_jsonl(directory / f"{split}_{kind}.jsonl")


def manifest():
    return json.loads((D / "dataset_manifest.json").read_text(encoding="utf-8"))


def test_inputs_blind_and_ids_disjoint():
    ids = []
    for split in builder.SPLITS:
        inputs, gold = jl(split, "inputs"), jl(split, "gold")
        assert [i["id"] for i in inputs] == [g["id"] for g in gold]
        assert all(set(i) == {"id", "problem", "language"} and i["language"] == "en" for i in inputs)
        ids.extend(i["id"] for i in inputs)
    assert len(ids) == len(set(ids))
    old_ids = {i["id"] for s in ("test", "dev") for i in jl(s, "inputs", ROOT / "data")}
    for split, letter in (("test", "t"), ("pilot", "p")):
        assert all(re.fullmatch(rf"jev-v2-{letter}-\d{{3}}", i["id"]) for i in jl(split, "inputs"))
        assert not old_ids & {i["id"] for i in jl(split, "inputs")}


def test_exclude_all_previous120_even_reused_dev_and_near_duplicates():
    old_inputs, old_gold = builder.load_previous()
    blocked = {g["template_group"] for g in old_gold}
    new_inputs = jl("test", "inputs") + jl("pilot", "inputs")
    new_gold = jl("test", "gold") + jl("pilot", "gold")
    assert not blocked & {g["template_group"] for g in new_gold}
    assert len(new_gold) == len({g["template_group"] for g in new_gold})
    assert len(new_inputs) == len({i["problem"] for i in new_inputs})
    for item in new_inputs:
        assert not any(old.near_duplicate(item["problem"], previous["problem"]) for previous in old_inputs)
    for a, b in combinations(new_inputs, 2):
        assert not old.near_duplicate(a["problem"], b["problem"]), (a["id"], b["id"])
    exclusion = manifest()["previous_exclusion"]
    assert exclusion["count"] == len(exclusion["ids"]) == len(exclusion["template_groups"]) == 120
    assert set(exclusion["template_groups"]) == blocked
    assert exclusion["dev_reuse_does_not_remove_exclusions"] is True


def test_dev_reused_only_for_debugging():
    assert jl("dev", "inputs") == jl("dev", "inputs", ROOT / "data")
    assert len(jl("dev", "gold")) == 20
    for current, previous in zip(jl("dev", "gold"), jl("dev", "gold", ROOT / "data")):
        assert {k: v for k, v in current.items() if k != "prompt_tokens"} == {
            k: v for k, v in previous.items() if k != "prompt_tokens"}


def test_real_counts_quotas_and_no_difficulty_reclassification():
    m = manifest()
    assert m["seed"] == builder.SEED == builder.INITIAL_SEED == 20261001
    assert m["actual_n"]["test"] <= 100 and m["actual_n"]["pilot"] <= 40 and m["actual_n"]["dev"] == 20
    for split in builder.SPLITS:
        records = jl(split, "gold")
        assert len(records) == m["actual_n"][split]
        for dom in old.DOMAINS:
            counts = Counter(g["difficulty"] for g in records if g["domain"] == dom)
            assert dict(counts) == {diff: n for diff, n in m["strata"][split][dom].items() if n}
            if split == "test":
                for diff, quota in old.TEST_QUOTA.items():
                    assert counts[diff] <= quota
                    if counts[diff] < quota:
                        assert {"split": "test", "domain": dom, "difficulty": diff,
                                "requested": quota, "selected": counts[diff]} in m["shortfalls"]
            if split == "pilot":
                assert sum(counts.values()) <= 8 and counts["hard"] == 0
                assert counts["medium"] <= 4
    assert builder.HARD_PREF["ratios_percentages"] == ["competition", "math", "gsm8k"]
    ratios_hard = [g for g in jl("test", "gold") if g["domain"] == "ratios_percentages" and g["difficulty"] == "hard"]
    assert Counter(g["source"] for g in ratios_hard)["openai/gsm8k"] <= 2
    assert all(g["source"] in {"openai/gsm8k", "EleutherAI/hendrycks_math"} for g in ratios_hard)
    assert m["review"]["reviewed"] is False and m["review"]["human_review_waived"] is True


def test_gold_filters_and_actual_rendered_tokens():
    count = builder.token_counter()
    for split in builder.SPLITS:
        for i, g in zip(jl(split, "inputs"), jl(split, "gold")):
            value = Fraction(g["gold_numerator"], g["gold_denominator"])
            assert g["gold_denominator"] > 0 and abs(value) <= 999
            assert old.parse_latex_gold(g["gold_raw"]) == value
            assert Fraction(g["gold_answer"]) == value
            assert not builder.GOLD_UNITS.search(g["gold_raw"])
            assert not builder.monetary_answer(i["problem"])
            assert not old.ASKS_PERCENT.search(i["problem"])
            assert g["reviewed"] is False
            assert g["prompt_tokens"] == count(i["problem"])
            assert max(g["prompt_tokens"].values()) <= 512
    assert manifest()["enable_thinking"] is False


@pytest.mark.parametrize("raw", ["$5", "\\$5", "5 dollars", "5 cents", "5%", "5\\%", "5 percent",
                                 "5 EUR", "5 euros", "5 pounds", "5 yen", "5 rupees", "5 quarters",
                                 "\u00a35", "\u20ac5", "\u00a55", "\u20b95"])
def test_explicit_money_and_percent_gold_rejected(raw):
    item = {"gold_raw": raw, "gold": Fraction(5), "problem": "Compute the value.",
            "subject": "gsm8k", "steps": 2}
    assert builder.assign(item) == (None, None, "money_or_percent_gold")
    selected, _ = builder.select([dict(item, source="openai/gsm8k", source_split="test", source_index=999)],
                                 [], [], lambda _: {"G4": 10, "B": 10})
    assert selected == []


@pytest.mark.parametrize("question", [
    "A game uses a quarter every twenty minutes. How much money is used?",
    "How MUCH MONEY is used?", "How much was paid?", "What is the total cost?",
    "What does it cost?", "How much did she spend?", "What amount was spent?",
    "What is the monthly salary?", "How many dollars were used?", "What is the price in cents?",
    "What is the amount in euros?", "She has 5 GBP. How much remains?",
    "The purse has quarters and dimes. What is their total value?",
    "Money was used to buy games. Find the amount used.",
])
def test_monetary_answer_never_selected(question):
    item = {"source": "openai/gsm8k", "source_split": "test", "source_index": 814,
            "subject": "gsm8k", "steps": 8, "problem": question, "gold_raw": "11",
            "solution": "", "level": None, "gold": Fraction(11)}
    assert builder.assign(item) == (None, None, "money")
    selected, audit = builder.select([item], [], [], lambda _: {"G4": 10, "B": 10})
    assert not selected and audit["exclusions"]["money"] == 1


@pytest.mark.parametrize("question", [
    "In how many different ways can 12 dimes be divided into three piles with an odd number of dimes in each pile?",
    "There is money in the form of nickels and quarters. How many ways can the coins be arranged?",
    "A fair coin is tossed three times. What is the probability of three heads?",
])
def test_coin_counting_is_not_rejected_as_money(question):
    item = {"source": "EleutherAI/hendrycks_math", "source_split": "counting_and_probability/test", "source_index": 0,
            "subject": "counting_and_probability", "problem": question, "gold_raw": "5", "gold": Fraction(5),
            "solution": "", "level": 3, "steps": None}
    assert not builder.monetary_answer(question)
    assert builder.assign(item)[:2] == ("counting_probability", "medium")
    selected, _ = builder.select([item], [], [], lambda _: {"G4": 10, "B": 10})
    assert len(selected) == 1


def test_spending_time_and_fractional_quarter_not_money():
    question = ("He spends 6 hours boating. This was 30% of the time he spent. "
                "He spent 40% of his time sightseeing. How much time did he spend sightseeing?")
    assert not builder.monetary_answer(question)
    assert not builder.monetary_answer("He eats a quarter of a sandwich. How much is left?")


def test_schedule_and_blind_review_are_gold_free():
    schedule = json.loads((D / "schedule.json").read_text(encoding="utf-8"))
    assert set(schedule) == {"planning_seed", "blocks", "order", "pilot_order"}
    assert schedule["planning_seed"] == builder.SEED
    assert schedule["order"] == [pid for block in schedule["blocks"] for pid in block]
    assert len(schedule["blocks"]) == 10
    assert sorted(schedule["order"]) == sorted(i["id"] for i in jl("test", "inputs"))
    assert sorted(schedule["pilot_order"]) == sorted(i["id"] for i in jl("pilot", "inputs"))
    domains = {g["id"]: g["domain"] for g in jl("test", "gold")}
    assert all(len(block) <= 10 and max(Counter(domains[pid] for pid in block).values(), default=0) <= 2
               for block in schedule["blocks"])
    blind = json.loads((D / "blind_review_ids.json").read_text(encoding="utf-8"))
    assert blind["reviewed"] is False and blind["seed"] == builder.SEED + 11
    assert len(blind["ids"]) == len(set(blind["ids"])) == 20
    assert Counter(domains[pid] for pid in blind["ids"]) == {dom: 4 for dom in old.DOMAINS}
    with (D / "review_sheet.csv").open(encoding="utf-8", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    assert len(reviews) == sum(manifest()["actual_n"].values())
    assert all(r["reviewed"] == "false" and not r["reviewer"] for r in reviews)


def test_hashes_cover_every_artifact_and_originals_unchanged():
    hashes = {}
    for line in (D / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        assert name not in hashes and ".." not in Path(name).parts and not Path(name).is_absolute()
        hashes[name] = digest
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest, name
    actual = {p.relative_to(D).as_posix() for p in D.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert set(hashes) == actual
    m = manifest()
    for name, digest in m["previous_artifact_sha256"].items():
        assert old.sha256_file(ROOT / "data" / name) == digest
    for name, digest in m["source_sha256"].items():
        assert old.sha256_file(ROOT / name) == digest
    assert m["builder_sha256"] == old.sha256_file(ROOT / "data" / "build_dataset_v2.py")
    for key, source in m["sources"].items():
        assert (source["repo"], source["revision"], source["files"]) == old.SOURCES[key]
    for key, meta in m["tokenizers"].items():
        base = D / "model_meta" / key if key == "G4" else old.RAW / "model_meta" / key
        for name, digest in meta["sha256"].items():
            assert old.sha256_file(base / name) == digest
    assert m["generator_prompt_sha256"] == old.sha256_file(ROOT / "prompts" / "generador.txt")


def test_full_selection_replays_before_inference():
    pool, _ = builder.load_pool()
    old_inputs, old_gold = builder.load_previous()
    chosen, audit = builder.select(pool, old_inputs, old_gold, builder.token_counter())
    m = manifest()
    for key, value in audit.items():
        assert m[key] == value
    for split in ("test", "pilot"):
        selected = sorted((it for it in chosen if it["split"] == split), key=lambda it: it["id"])
        assert [builder.gold_record(it) for it in selected] == jl(split, "gold")
        assert [{"id": it["id"], "problem": it["problem"], "language": "en"} for it in selected] == jl(split, "inputs")
    assert builder.schedule_for(jl("test", "gold"), jl("pilot", "gold")) == json.loads(
        (D / "schedule.json").read_text(encoding="utf-8"))


def test_shortage_never_relaxes_filters_and_test_reserved_first():
    pool = [{"source": "openai/gsm8k", "source_split": "test", "source_index": i,
             "subject": "gsm8k", "steps": 2, "problem": f"Calculate {i} plus one.",
             "gold_raw": str(i + 1), "solution": "", "level": None} for i in range(16)]
    pool += [{**pool[0], "source_index": 99, "gold_raw": "1000"},
             {**pool[0], "source_index": 100, "problem": "How many dollars?"},
             {**pool[0], "source_index": 101, "problem": "This input is too long."}]
    selected, audit = builder.select(pool, [], [], lambda text: {"G4": 513 if "too long" in text else 10, "B": 10})
    assert Counter(it["split"] for it in selected) == {"test": 8, "pilot": 8}
    assert all(it["difficulty"] == "easy" and it["source_index"] < 16 for it in selected)
    assert audit["selection_attempts"][:15] == [
        {"split": "test", "domain": dom, "difficulty": diff, "requested": n,
         "selected": 8 if dom == "arithmetic" and diff == "easy" else 0}
        for dom in old.DOMAINS for diff, n in old.TEST_QUOTA.items()]
    assert audit["exclusions"]["gold_ge_1000"] == 1
    assert audit["exclusions"]["money"] == 1
    assert audit["exclusions"]["prompt_gt_512_tokens"] == 1


def test_frozen_builder_refuses_rebuild_without_writing():
    before = {p: old.sha256_file(p) for p in D.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "data" / "build_dataset_v2.py")],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "frozen" in result.stderr
    assert {p: old.sha256_file(p) for p in D.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("split", builder.SPLITS)
def test_existing_runner_loads_isolated_gold_free_records(split):
    from jevlab.runner import Experiment

    # Exercise data loading only: never initialize engines, results, or inference.
    experiment = Experiment.__new__(Experiment)
    experiment.root = ROOT
    experiment.data_dir = D
    experiment.p = {"DATA_SUBDIR": "data/v2", "SPLIT": split, "N_PROBLEMS": 1000}
    loaded = experiment.problems()
    schedule = json.loads((D / "schedule.json").read_text(encoding="utf-8"))
    expected = (schedule["order"] if split == "test" else schedule["pilot_order"]
                if split == "pilot" else sorted(i["id"] for i in jl("dev", "inputs")))
    assert [i["id"] for i in loaded] == expected
    assert all(set(i) == {"id", "problem", "language"} for i in loaded)
    assert experiment.dev_problems() == jl("dev", "inputs")
    assert experiment._data_hashes() == {
        name: old.sha256_file(D / name) for name in
        ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "schedule.json")}


def test_initial_candidate_archive_and_exact_audit_binding():
    m = manifest()
    audit_path = ROOT / "results" / "v2" / "preparation" / "dataset_initial_rejected.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert m["initial_candidate_audit"] == audit
    assert m["initial_candidate_audit_sha256"] == old.sha256_file(audit_path)
    assert m["monetary_filter_correction"] == builder.correction_policy()
    for key, value in builder.correction_policy().items():
        if key not in ("preregistered_seed", "construction_seed", "seed_policy"):
            assert audit["correction"][key] == value
    assert audit["initial_seed"] == 20261001 and audit["replacement_seed"] == 20261002
    assert audit["initial_manifest_sha256"] == "9bfe4096aeb4346ac4824fd2a42329d69d98663012435cf4978f313f74c08ee1"
    assert audit["initial_sha256sums_sha256"] == "11ef44a8b31c04823fdedae75666ae03eba4635ff9e10800a5d5c57d22aa0dca"
    assert audit["pre_inference_guard"]["runtime_execution_files"] == []
    archive = ROOT / audit["archive"]
    assert old.sha256_file(archive) == audit["archive_sha256"]
    with zipfile.ZipFile(archive) as handle:
        inventory = audit["initial_files_sha256"]
        assert set(handle.namelist()) == {"data/v2/" + name for name in inventory} | {"hash_inventory.json"}
        assert json.loads(handle.read("hash_inventory.json")) == inventory
        for name, digest in inventory.items():
            assert hashlib.sha256(handle.read("data/v2/" + name)).hexdigest() == digest
        archived_sums = dict((name.strip(), digest) for digest, name in
                             (line.split(None, 1) for line in handle.read("data/v2/SHA256SUMS").decode().splitlines()))
        assert archived_sums == {name: digest for name, digest in inventory.items() if name != "SHA256SUMS"}
    assert audit["rejected_item"]["template_group"] == "openai/gsm8k::test::814"
    for split in ("test", "pilot"):
        assert all(g["template_group"] != audit["rejected_item"]["template_group"] for g in jl(split, "gold"))
    assert set(inventory) == builder.GENERATED_FILES


def test_wrong_seed_candidate_archive_and_truthful_restoration_audit():
    m = manifest()
    audit_path = ROOT / "results" / "v2" / "preparation" / "dataset_wrong_seed_rejected.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert m["seed_restoration_audit"] == audit
    assert m["seed_restoration_audit_sha256"] == old.sha256_file(audit_path)
    assert audit["wrong_seed"] == 20261002 and audit["restored_seed"] == m["seed"] == 20261001
    assert audit["seed_change_was_authorized"] is False
    assert audit["prior_preregistration_claim_was_false"] is True
    assert audit["monetary_filter_unchanged"] is True
    assert audit["initial_money_rejection_audit"] == m["initial_candidate_audit"]
    assert audit["initial_money_rejection_audit_sha256"] == m["initial_candidate_audit_sha256"]
    assert audit["wrong_seed_manifest_sha256"] == "424c1c727d03314ace48bf4316caeb84421778d1166c8fb3b7229124f21a587a"
    assert audit["wrong_seed_sha256sums_sha256"] == "9ece345fdce47ae64cf28cb3be1cb4b14f4cfcd813b80740089793b22e81e20e"
    assert audit["pre_inference_guard"]["runtime_execution_files"] == []
    assert audit["correction"] == m["monetary_filter_correction"]
    archive = ROOT / audit["archive"]
    assert old.sha256_file(archive) == audit["archive_sha256"]
    with zipfile.ZipFile(archive) as handle:
        inventory = audit["wrong_seed_files_sha256"]
        assert set(handle.namelist()) == {"data/v2/" + name for name in inventory} | {"hash_inventory.json"}
        assert json.loads(handle.read("hash_inventory.json")) == inventory
        for name, digest in inventory.items():
            assert hashlib.sha256(handle.read("data/v2/" + name)).hexdigest() == digest
        wrong_manifest = json.loads(handle.read("data/v2/dataset_manifest.json"))
        assert wrong_manifest["seed"] == 20261002
        assert wrong_manifest["initial_candidate_audit"] == m["initial_candidate_audit"]
    assert set(inventory) == builder.GENERATED_FILES


def test_seed_restoration_is_one_off_without_writes():
    paths = [p for directory in (D, ROOT / "results" / "v2" / "preparation")
             for p in directory.rglob("*") if p.is_file()]
    before = {p: old.sha256_file(p) for p in paths}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "data" / "build_dataset_v2.py"),
                             "--restore-preregistered-seed"], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and any(reason in result.stderr for reason in
                                         ("second repair", "runtime execution artifact"))
    assert before == {p: old.sha256_file(p) for p in paths}


@pytest.mark.parametrize("relative", ["pilot/JFINAL/predictions.jsonl", "dev/run/manifest.json",
                                     "session_guard.log", "some_other_file.txt"])
@pytest.mark.parametrize("restore_seed", [False, True])
def test_repair_forbidden_after_runtime_artifact(tmp_path, monkeypatch, relative, restore_seed):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    runtime = tmp_path / "results" / "v2" / relative
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"execution")
    monkeypatch.setattr(builder, "OUT", tmp_path / "data" / "v2")
    with pytest.raises(ValueError, match="runtime execution artifact"):
        builder.repair_before_inference(restore_seed=restore_seed)
    assert not (tmp_path / "results" / "v2" / "preparation").exists()


def test_preparation_is_allowed_but_second_repair_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    preparation = tmp_path / "results" / "v2" / "preparation"
    preparation.mkdir(parents=True)
    (preparation / "dataset_initial_rejected.json").write_bytes(b"preserved audit")
    assert builder.no_runtime_execution()["runtime_execution_files"] == []
    before = {p: old.sha256_file(p) for p in D.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="second repair"):
        builder.repair_before_inference()
    assert before == {p: old.sha256_file(p) for p in D.rglob("*") if p.is_file()}


@pytest.fixture
def initial_candidate(tmp_path, monkeypatch):
    directory = tmp_path / "data" / "v2"
    contents = {name: b"initial file" for name in builder.GENERATED_FILES if name != "SHA256SUMS"}
    contents["test_inputs.jsonl"] = (json.dumps({"id": "jev-v2-t-028", "problem": "How much money is used?",
                                               "language": "en"}) + "\n").encode()
    contents["test_gold.jsonl"] = (json.dumps({"id": "jev-v2-t-028", "template_group": "openai/gsm8k::test::814",
                                             "gold_answer": "11"}) + "\n").encode()
    contents["dataset_manifest.json"] = json.dumps({"seed": builder.INITIAL_SEED, "dataset_version": "v2",
        "builder_sha256": "initial builder hash", "actual_n": {"test": 1, "pilot": 0, "dev": 0}}).encode()
    contents["SHA256SUMS"] = "".join(f"{hashlib.sha256(contents[name]).hexdigest()}  {name}\n"
                                    for name in sorted(contents)).encode()
    for name, content in contents.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o444)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "OUT", directory)
    return directory, contents


def test_repair_preserves_every_initial_byte_before_unsealing(initial_candidate, tmp_path):
    directory, contents = initial_candidate
    unowned = tmp_path / "data" / "unowned.txt"
    unowned.write_bytes(b"do not change")
    audit = builder.repair_before_inference()
    with zipfile.ZipFile(tmp_path / audit["archive"]) as handle:
        for name, content in contents.items():
            assert handle.read("data/v2/" + name) == content
    assert not (directory / "SHA256SUMS").exists()
    for name, content in contents.items():
        if name != "SHA256SUMS":
            assert (directory / name).read_bytes() == content
            assert (directory / name).stat().st_mode & 0o200
    assert unowned.read_bytes() == b"do not change"
    assert audit["initial_files_sha256"] == {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}


@pytest.mark.parametrize("defect", ["changed", "unexpected", "late_runtime"])
def test_archive_guard_failure_never_unseals_initial(initial_candidate, tmp_path, monkeypatch, defect):
    directory, contents = initial_candidate
    if defect == "changed":
        path = directory / "test_gold.jsonl"
        path.chmod(0o644)
        path.write_bytes(b"changed")
        path.chmod(0o444)
    elif defect == "unexpected":
        (directory / "unowned.txt").write_bytes(b"must not unseal")
    else:
        original_guard = builder.no_runtime_execution
        calls = []

        def late_runtime():
            calls.append(True)
            if len(calls) > 1:
                raise ValueError("runtime execution artifact appeared before unseal")
            return original_guard()

        monkeypatch.setattr(builder, "no_runtime_execution", late_runtime)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        builder.repair_before_inference()
    assert before == {p: (p.read_bytes(), p.stat().st_mode) for p in directory.rglob("*") if p.is_file()}


@pytest.fixture
def wrong_seed_candidate(initial_candidate, tmp_path):
    directory, contents = initial_candidate
    initial_audit = builder.repair_before_inference()
    prior_record = tmp_path / "results" / "v2" / "preparation" / "dataset_initial_rejected.json"
    wrong_manifest = {"seed": 20261002, "dataset_version": "v2", "builder_sha256": "wrong seed builder hash",
                      "actual_n": {"test": 1, "pilot": 0, "dev": 0},
                      "initial_candidate_audit": initial_audit,
                      "initial_candidate_audit_sha256": old.sha256_file(prior_record),
                      "monetary_filter_correction": builder.correction_policy()}
    contents = dict(contents, **{"dataset_manifest.json": json.dumps(wrong_manifest).encode()})
    contents["SHA256SUMS"] = "".join(f"{hashlib.sha256(contents[name]).hexdigest()}  {name}\n"
                                    for name in sorted(contents) if name != "SHA256SUMS").encode()
    for name, content in contents.items():
        path = directory / name
        path.write_bytes(content)
        path.chmod(0o444)
    return directory, contents


def test_seed_restoration_preserves_wrong_candidate_and_prior_archive(wrong_seed_candidate, tmp_path):
    directory, contents = wrong_seed_candidate
    preparation = tmp_path / "results" / "v2" / "preparation"
    prior = {p: p.read_bytes() for p in preparation.iterdir()}
    audit = builder.repair_before_inference(restore_seed=True)
    with zipfile.ZipFile(tmp_path / audit["archive"]) as handle:
        for name, content in contents.items():
            assert handle.read("data/v2/" + name) == content
    assert prior == {p: p.read_bytes() for p in prior}
    assert not (directory / "SHA256SUMS").exists()
    assert audit["wrong_seed_files_sha256"] == {name: hashlib.sha256(content).hexdigest()
                                              for name, content in contents.items()}
    assert audit["seed_change_was_authorized"] is False and audit["restored_seed"] == 20261001


def test_seed_restoration_rejects_changed_prior_audit_without_unsealing(wrong_seed_candidate, tmp_path):
    directory, _ = wrong_seed_candidate
    prior = tmp_path / "results" / "v2" / "preparation" / "dataset_initial_rejected.json"
    prior.chmod(0o644)
    prior.write_text("{}", encoding="utf-8")
    prior.chmod(0o444)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="bound to the preserved initial money rejection"):
        builder.repair_before_inference(restore_seed=True)
    assert before == {p: (p.read_bytes(), p.stat().st_mode) for p in directory.rglob("*") if p.is_file()}
    assert not (prior.parent / "dataset_wrong_seed_rejected.zip").exists()
