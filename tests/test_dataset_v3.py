"""Local policy fixtures never create or claim actual completed dataset reviews."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
spec = importlib.util.spec_from_file_location("dataset_v3_builder", ROOT / "data" / "build_dataset_v3.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def item(**changes):
    return {"source": "openai/gsm8k", "source_split": "test", "source_index": 999999,
            "subject": "gsm8k", "steps": 4, "problem": "Five boxes each have two toys. How many toys altogether?",
            "gold_raw": "10", "solution": "Synthetic test fixture only, never an actual agent review.",
            "level": None, **changes}


def candidate(domain="arithmetic", difficulty="medium", identifier="fixture-only"):
    return {"id": identifier, "candidate_sha256": "a" * 64, "provisional_domain": domain,
            "provisional_difficulty": difficulty, "gold_numerator": 10, "gold_denominator": 1,
            "source_family": "fixture"}


def review_pair(c, rid="fixture-only"):
    blind = {"id": c["id"], "candidate_sha256": c["candidate_sha256"], "reviewer_id": rid,
             "phase": "blind", "domain": c["provisional_domain"], "difficulty": c["provisional_difficulty"],
             "independent_solution": "Synthetic fixture only: five boxes each contain two toys, so multiplication of five by two yields ten toys in total. This is not an actual agent output.",
             "answer": {"numerator": 10, "denominator": 1},
             "domain_reason": "The requested operation is ordinary arithmetic multiplication without variables, number theoretic constraints or probability.",
             "difficulty_reason": "This synthetic example exercises schema validation only and does not establish a real benchmark difficulty judgment.",
             **{k: True for k in ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")}}
    reference = {"id": c["id"], "candidate_sha256": c["candidate_sha256"], "reviewer_id": rid,
                 "phase": "reference", "blind_record_sha256": hashlib.sha256(b.canonical(blind).encode()).hexdigest(),
                 "reference_comparison": "Synthetic fixture only: the supplied rational answer equals the independently computed product in this schema test, not an actual agent comparison.",
                 "gold_matches": True, "solution_consistent": True, "approve": True}
    return blind, reference


@pytest.mark.parametrize("raw, expected, unit", [
    ("1000000000", Fraction(10**9), None), ("-1000000000", Fraction(-10**9), None),
    ("1/3", Fraction(1, 3), None), ("0.125", Fraction(1, 8), None),
    (r"\frac{2}{3}", Fraction(2, 3), None), ("$5$", Fraction(5), None),
    ("$5", Fraction(5), "dollars"), (r"\$5", Fraction(5), "dollars"),
    ("5 dollars", Fraction(5), "dollars"), ("5 percent", Fraction(5), "percent"),
    (r"5\%", Fraction(5), r"\%"), ("5%", Fraction(5), "%"),
    ("1,250", Fraction(1250), None), (r"5\text{ cents}", Fraction(5), "cents"),
])
def test_corrected_numeric_policy(raw, expected, unit):
    assert b.numeric_gold(raw) == (expected, unit)


@pytest.mark.parametrize("raw", ["1/0", r"\sqrt{2}", "2+2", "1,2", "1e2", "3 or 4", "(3,4)"])
def test_non_scalar_or_inexact_gold_fails_closed(raw):
    assert b.numeric_gold(raw)[0] is None


def test_money_percent_and_latex_delimiters_not_literal_exclusions():
    for question, raw in [("A ticket costs $5 and she buys two. Find the total cost in dollars.", "10 dollars"),
                          ("A value rises from 20 to 30. What is the percentage increase?", "50%"),
                          ("Compute $2+3+5$.", "10")]:
        assigned, reason = b.provisional(item(problem=question, gold_raw=raw))
        assert assigned is not None and reason is None
    assert b.provisional(item(gold_raw="1000000001"))[1] == "gold_abs_gt_1e9"


@pytest.mark.parametrize("first, second, reason", [
    ("Find 2 plus 3.", "find 2 plus 3!", "exact_normalized"),
    ("Find 2 plus 3.", "Find 9 plus 4.", "numeric_template_exact"),
    ("A farmer has 12 red hens and buys 4 more red hens today how many hens now",
     "A farmer has 15 red hens and buys 7 more red hens today how many hens now", "numeric_template_exact"),
    ("A farmer has 12 red hens and buys 4 more red hens today how many hens now",
     "A farmer has 15 red hens and purchases 7 more red hens today how many hens now", "numeric_template_approximate"),
    ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
     "alpha beta gamma delta epsilon zeta eta theta iota kappa mu", "shared_8gram"),
])
def test_duplicates_including_short_and_numeric_templates(first, second, reason):
    assert b.duplicate_reason(first, second) == reason
    assert b.DuplicateIndex([{"problem": first}]).match(second) == reason


def test_unrelated_short_questions_do_not_collide():
    assert b.duplicate_reason("Add 2 and 3.", "Multiply 2 and 3.") is None


def test_prealgebra_arithmetic_requires_genuine_domain():
    base = item(subject="prealgebra", level=4, steps=None)
    assert b.provisional(base)[0]["provisional_domain"] == "arithmetic"
    assert b.provisional({**base, "problem": "Solve for $x$ when $x+2=4$."})[0] is None
    assert b.provisional({**base, "problem": "Find the area of a rectangle."})[0] is None
    assert b.provisional({**base, "problem": "Find the greatest common divisor of 12 and 18."})[0]["provisional_domain"] == "number_theory"
    assert b.provisional({**base, "problem": "A shadow is six feet long. How tall is the tree?"})[0] is None
    assert b.provisional({**base, "problem": "Find the value of a ratio of 12 to 18."})[0]["provisional_domain"] == "ratios_percentages"


def test_asdiv_grade_alone_cannot_pad_hard():
    assigned, _ = b.provisional(item(subject="asdiv", level="6", steps=1,
                                    problem="A value rises by 10 percent. Find its percentage increase."))
    assert assigned["provisional_difficulty"] == "easy"
    assert b.provisional(item(subject="asdiv", level="6", steps=1))[1] == "arithmetic_domain_requires_multistep_not_single_operation"
    assert b.provisional(item(subject="asdiv", level="1", steps=7))[0]["provisional_difficulty"] == "hard"


def test_filter_boundaries_and_required_three_tokenizers():
    pool = [item()]
    rows, audit = b.discover(pool, [], [], lambda _: {k: 1024 for k in b.MODELS})
    assert len(rows) == 1 and audit["candidate_pool_count"] == 1
    assert rows[0]["candidate_sha256"] == b.candidate_sha(rows[0])
    assert rows[0]["raw_problem"] == rows[0]["problem"]
    rows, audit = b.discover(pool, [], [], lambda _: {"G4": 1024, "B13": 1025, "O9": 1})
    assert not rows and audit["exclusions"]["rendered_prompt_gt_1024"] == 1
    with pytest.raises(ValueError, match="all three"):
        b.discover(pool, [], [], lambda _: {"G4": 1, "B13": 1})
    rows, audit = b.discover(pool, [{"source_id": b.source_id(pool[0]), "problem": pool[0]["problem"]}],
                             [], lambda _: {k: 1 for k in b.MODELS})
    assert not rows and audit["exclusions"]["used_source_id"] == 1


def test_archive_and_short_text_exclusion_independent_from_used_union():
    pool = [item(problem="Find 2 plus 3.")]
    rows, audit = b.discover(pool, [], [{"source_id": "different", "problem": "FIND 2 plus 3!"}],
                             lambda _: {k: 1 for k in b.MODELS})
    assert not rows and audit["exclusions"]["previous_or_archive_exact_normalized"] == 1


def test_candidate_hash_binds_source_gold_problem_labels_and_tokens():
    rows, _ = b.discover([item()], [], [], lambda _: {k: 1 for k in b.MODELS})
    row = rows[0]
    for field, value in [("problem", "Changed statement"), ("gold_numerator", 99),
                         ("provisional_domain", "algebra"), ("source_index", 0),
                         ("prompt_tokens", {k: 2 for k in b.MODELS})]:
        assert b.candidate_sha({**row, field: value}) != row["candidate_sha256"]


def test_full_evidence_core_binds_all_judgments_and_only_ignores_parent_hashes():
    c = candidate()
    blind, _ = review_pair(c)
    core = b.review_core(blind)
    assert core == b.review_core({**blind, "packet_sha256": "parent-added"})
    for field, changed in [("answer", {"numerator": 11, "denominator": 1}),
                           ("correctness", False), ("domain_ok", False), ("difficulty", "hard")]:
        assert b.review_core({**blind, field: changed}) != core


def test_cached_tokenizer_bytes_must_match_authoritative_pin(tmp_path):
    path = tmp_path / "tokenizer_config.json"
    content = b'{"fixture":true}'
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    b.verify_metadata_bytes(tmp_path, {path.name: ("sha256", sha)})
    b.verify_metadata_bytes(tmp_path, {path.name: ("git_blob", blob)})
    path.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="authoritative revision"):
        b.verify_metadata_bytes(tmp_path, {path.name: ("sha256", sha)})


def test_seal_marker_not_published_if_fsync_interrupted(tmp_path, monkeypatch):
    artifact = tmp_path / "fixture.json"
    artifact.write_text("{}", encoding="utf-8")

    def interrupted(_):
        raise OSError("fixture interruption")

    monkeypatch.setattr(b.os, "fsync", interrupted)
    with pytest.raises(OSError, match="fixture interruption"):
        b.seal(tmp_path, [artifact])
    assert not (tmp_path / "SHA256SUMS").exists()


def test_seal_atomic_marker_verifies_entire_inventory(tmp_path):
    artifact = tmp_path / "fixture.json"
    artifact.write_text("{}", encoding="utf-8")
    b.seal(tmp_path, [artifact])
    assert not (tmp_path / "SHA256SUMS.pending").exists()
    assert b.verify_seal(tmp_path) == {artifact.name: b.digest(artifact)}


def test_all_emitted_rejections_must_be_consumed(tmp_path, monkeypatch):
    # Only a local rejected schema fixture. No approved record or final publication.
    monkeypatch.setattr(b, "ROOT", tmp_path)
    c = candidate()
    stage = tmp_path / "data" / "v3" / "staging"
    directory = stage / "reviews"
    output = {"id": c["id"], "reviewer_id": "fixture-only", "phase": "blind", "correctness": False}
    evidence = directory / "evidence" / "schema.jsonl"
    b.write_jsonl(evidence, [output])
    b.write_json(directory / "reviewers.json", {"reviewers": [{"reviewer_id": "fixture-only",
        "agent_run_id": "local-schema-fixture", "kind": "agent", "independent_blind_review": True,
        "evidence": [{"path": "evidence/schema.jsonl", "sha256": b.digest(evidence)}]}]})
    with pytest.raises(ValueError, match="Omitted emitted agent judgments"):
        b.load_reviews([c], stage, [{"candidate_ids": [c["id"]]}])


def test_synthetic_schema_checks_do_not_publish_or_authenticate_reviews():
    c = candidate()
    blind, reference = review_pair(c)
    assert b.validate_review(c, blind, reference) is True  # schema only, not an actual-review claim
    for field in ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit"):
        rejected = {**blind, field: False}
        comparison = {**reference, "approve": False,
                      "blind_record_sha256": hashlib.sha256(b.canonical(rejected).encode()).hexdigest()}
        assert b.validate_review(c, rejected, comparison) is False
    with pytest.raises(ValueError, match="bound"):
        b.validate_review(c, {**blind, "candidate_sha256": "b" * 64}, reference)
    with pytest.raises(ValueError, match="detailed independent"):
        b.validate_review(c, {**blind, "independent_solution": "approved"}, reference)
    with pytest.raises(ValueError, match="bind"):
        b.validate_review(c, blind, {**reference, "blind_record_sha256": "b" * 64})
    with pytest.raises(ValueError, match="disagrees"):
        b.validate_review({**c, "gold_numerator": 11}, blind, reference)
    with pytest.raises(ValueError, match="different reviewers"):
        b.validate_review(c, blind, {**reference, "reviewer_id": "fixture-two"})


def test_missing_actual_registry_and_dummy_evidence_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    stage = tmp_path / "data" / "v3" / "staging"
    directory = stage / "reviews"
    b.write_json(directory / "reviewers.json", {"reviewers": []})
    with pytest.raises(ValueError, match="Missing actual"):
        b.load_reviews([], stage, [])
    evidence = directory / "evidence" / "fixture.txt"
    evidence.parent.mkdir()
    evidence.write_text("Synthetic fixture only, never actual agent execution.", encoding="utf-8")
    (directory / "reviewers.json").unlink()
    b.write_json(directory / "reviewers.json", {"reviewers": [{"reviewer_id": "fixture-only",
        "agent_run_id": "synthetic-fixture", "kind": "agent", "independent_blind_review": True,
        "evidence": [{"path": "evidence/fixture.txt", "sha256": b.digest(evidence)}]}]})
    with pytest.raises(ValueError, match="Fixture/dummy"):
        b.load_reviews([], stage, [])
    assert not list((tmp_path / "data" / "v3").glob("test_*.jsonl"))


def test_test_reserved_first_without_quota_relaxation():
    rows = []
    for domain in b.DOMAINS:
        for diff in b.TEST_QUOTA:
            for number in range(b.TEST_QUOTA[diff] + b.PILOT_QUOTA[diff]):
                row = candidate(domain, diff, f"fixture-{domain}-{diff}-{number:03}")
                row["source_family"] = "fixture-a" if number % 2 else "fixture-b"
                rows.append(row)
    approved_ids = {r["id"] for r in rows}  # pure selection fixture, no review claims or output
    decisions = {r["id"]: {"accepted": True, "final_domain": r["provisional_domain"],
                           "final_difficulty": r["provisional_difficulty"]} for r in rows}
    selected = b.select_approved(rows, approved_ids, decisions)
    assert {s: len(rr) for s, rr in selected.items()} == {"test": 500, "pilot": 50}
    for domain in b.DOMAINS:
        for diff in b.TEST_QUOTA:
            first = [r for r in rows if r["provisional_domain"] == domain and r["provisional_difficulty"] == diff]
            actual = [r for r in selected["test"] if r["provisional_domain"] == domain and r["provisional_difficulty"] == diff]
            assert actual == first[:b.TEST_QUOTA[diff]]
    approved_ids.remove(rows[-1]["id"])
    with pytest.raises(ValueError, match="Insufficient twice-reviewed"):
        b.select_approved(rows, approved_ids, decisions)


def test_reviewed_domain_and_original_source_tier_drive_selection_not_intrinsic_difficulty():
    # Pure selection fixture, no emitted events, actual registration or publication.
    rows, decisions = [], {}
    for domain in b.DOMAINS:
        for diff in b.TEST_QUOTA:
            for number in range(b.TEST_QUOTA[diff] + b.PILOT_QUOTA[diff]):
                row = candidate("algebra", diff, f"fixture-final-{domain}-{diff}-{number}")
                row.update(source_family="fixture-a" if number % 2 else "fixture-b", source_id=row["id"], gold_raw="10")
                row["candidate_sha256"] = b.candidate_sha(row)
                rows.append(row)
                decisions[row["id"]] = {"accepted": True, "final_domain": domain, "final_difficulty": "easy" if number % 2 else "medium",
                                        "accepted_via_adjudication": (domain, diff) != ("algebra", "hard")}
    before = b.canonical(rows)
    approved = set(decisions)
    selected = b.select_approved(rows, approved, decisions)
    assert {s: len(rr) for s, rr in selected.items()} == {"test": 500, "pilot": 50}
    assert b.reviewed_capacity(selected["test"], approved, decisions, axis=b.QUOTA_AXIS) == {d: b.TEST_QUOTA for d in b.DOMAINS}
    assert b.reviewed_capacity(selected["pilot"], approved, decisions, axis=b.QUOTA_AXIS) == {d: b.PILOT_QUOTA for d in b.DOMAINS}
    assert all(tiers["hard"] == 0 for tiers in b.reviewed_capacity(rows, approved, decisions).values())
    assert b.canonical(rows) == before and all(b.candidate_sha(r) == r["candidate_sha256"] for r in rows)
    with pytest.raises(KeyError):
        b.select_approved(rows, approved, {})  # Never fall back to provisional source labels.
    decisions[rows[0]["id"]]["status"] = "awaiting_adjudication"
    with pytest.raises(ValueError, match="ALL eligible"):
        b.select_approved(rows, approved, decisions)


def test_real_source_replay_and_reviewed_label_selection_preserve_original_packets():
    # Hypothetical selection schema only: no actual-review claim or final files.
    stage = b.OUT / "staging"
    if not (stage / "preparation_manifest.json").exists():
        pytest.skip("Pinned preparation sources unavailable")
    original = b.read_jsonl(stage / "candidate_pool.jsonl")
    used, extra, _, _ = b.load_exclusions()
    pool, _, _ = b.load_sources()
    counter, _ = b.tokenizer_counter()
    replayed, _ = b.discover(pool, used, extra, counter)
    replayed, _, _ = b.review_priority(replayed)
    assert replayed == original
    decisions = {}
    for domain in b.DOMAINS:
        for diff in b.TEST_QUOTA:
            ids = [r["id"] for r in original if (r["provisional_domain"], r["provisional_difficulty"]) == (domain, diff)]
            for pid in ids[:b.TEST_QUOTA[diff] + b.PILOT_QUOTA[diff]]:
                decisions[pid] = {"accepted": True, "final_domain": domain, "final_difficulty": "easy"}
    ids = list(decisions)
    first = next(pid for pid in ids if decisions[pid]["final_domain"] == "arithmetic")
    second = next(pid for pid in ids if decisions[pid]["final_domain"] == "algebra")
    decisions[first], decisions[second] = decisions[second], decisions[first]
    selected = b.select_approved(original, set(decisions), decisions)
    assert selected == b.select_approved(replayed, set(decisions), decisions)
    for split, rows in selected.items():
        final = [{**r, "domain": decisions[r["id"]]["final_domain"], "difficulty": decisions[r["id"]]["final_difficulty"],
                  "sampling_tier": r["provisional_difficulty"]} for r in rows]
        assert Counter((r["domain"], r["sampling_tier"]) for r in final) == {
            (d, f): n for d in b.DOMAINS for f, n in (b.TEST_QUOTA if split == "test" else b.PILOT_QUOTA).items()}
        assert all({k: v for k, v in r.items() if k not in ("domain", "difficulty", "sampling_tier")} == source
                   for r, source in zip(final, rows))


def test_priority_reserves_primary550_then_weak_cell_backups_without_changing_labels():
    rows = []
    for domain in b.DOMAINS:
        for difficulty in b.TEST_QUOTA:
            count = b.TEST_QUOTA[difficulty] + b.PILOT_QUOTA[difficulty] + 50
            rows.extend(candidate(domain, difficulty, f"fixture-{domain}-{difficulty}-{n}") for n in range(count))
    original = {r["id"]: (r["provisional_domain"], r["provisional_difficulty"]) for r in rows}
    ordered, primary_count, shortfalls = b.review_priority(rows)
    assert primary_count == 550 and not shortfalls
    for domain in b.DOMAINS:
        for difficulty in b.TEST_QUOTA:
            assert sum(r["provisional_domain"] == domain and r["provisional_difficulty"] == difficulty
                       for r in ordered[:550]) == b.TEST_QUOTA[difficulty] + b.PILOT_QUOTA[difficulty]
    assert {r["id"]: (r["provisional_domain"], r["provisional_difficulty"]) for r in ordered} == original
    assert len({r["id"] for r in ordered}) == len(rows)


def test_latency_is_fixed_seed_balanced_test_subset():
    test = [{"id": f"fixture-{domain}-{diff}-{n}", "domain": domain, "sampling_tier": diff,
             "difficulty": "easy" if n % 2 else "medium"}
            for domain in b.DOMAINS for diff, quota in b.TEST_QUOTA.items() for n in range(quota)]
    selected = b.latency_subset(test)
    assert len(selected) == len(set(selected)) == 100
    assert set(selected) <= {r["id"] for r in test}
    mapping = {r["id"]: r for r in test}
    assert Counter((mapping[pid]["domain"], mapping[pid]["sampling_tier"]) for pid in selected) == {
        (domain, diff): quota for domain in b.DOMAINS for diff, quota in b.LATENCY_QUOTA.items()}
    assert selected == b.latency_subset(list(reversed(test)))
    assert b.SEED == 20261002 and b.LATENCY_SEED == 20261003


@pytest.mark.parametrize("defect", ["duplicate_selected_id", "unresolved_intrinsic_label"])
def test_latency_source_tier_metadata_does_not_hide_duplicate_ids_or_unresolved_intrinsic_labels(defect):
    rows = [{"id": f"schema-{d}-{f}-{n}", "problem": "Local schema-only latency item, never actual execution",
             "domain": d, "sampling_tier": f, "difficulty": "easy"}
            for d in b.DOMAINS for f, quota in b.TEST_QUOTA.items() for n in range(quota)]
    selected = b.latency_subset(rows)
    schedule = b.schedule_for(rows, [])
    if defect == "duplicate_selected_id":
        same_cell = [r["id"] for r in rows if r["id"] in selected and r["domain"] == b.DOMAINS[0] and r["sampling_tier"] == "easy"]
        schedule["order"][schedule["order"].index(same_cell[1])] = same_cell[0]
    else:
        next(r for r in rows if r["id"] == selected[0])["difficulty"] = "unresolved"
    with pytest.raises(ValueError, match="unique test ID|resolved intrinsic"):
        b.latency_plan_for(rows, selected, schedule)


@pytest.mark.parametrize("subject", ["gsm8k", "asdiv", "algebra", "number_theory", "counting_and_probability", "prealgebra"])
def test_content_ratio_rule_is_global_and_preserves_source_identity(subject):
    original = item(subject=subject, level=4, steps=7,
                    problem="A person's weight is 40 percent more than B's and 30 percent less than C's. Find the ratio of B's weight to C's weight.")
    before = dict(original)
    assigned, reason = b.provisional(original)
    assert reason is None and assigned["provisional_domain"] == "ratios_percentages"
    assert original == before
    assert assigned["classification_rule"] == "central_quantitative_shares_percentage_unit_rate_or_variation"


@pytest.mark.parametrize("problem, expected", [
    ("Find the ratio of the roots of a quadratic polynomial.", "algebra"),
    ("Find the ratio of the greatest common divisor to the least common multiple.", "number_theory"),
    ("Find the percentage of integers whose remainder on division by seven is three.", "number_theory"),
    ("A weighted die has different face probabilities. Find its expected value.", "counting_probability"),
    ("Two cyclists travel 13 miles at different speeds in two hours. Find the ratio of their travel times.", "ratios_percentages"),
    ("Find the ratio of the areas of two circles with different radii.", None),
    ("Find the value of a function inverse involving an exact fraction.", "algebra"),
    ("Find the 4037th decimal digit of 1/17.", "number_theory"),
    ("An unfair coin is flipped many times, and heads are twice as likely as tails.", "counting_probability"),
])
def test_ratio_negative_controls_not_rational_format_or_incidental_keywords(problem, expected):
    domain, _ = b.content_domain(item(subject="algebra", problem=problem))
    assert domain == expected


def test_geometry_homonyms_and_scalar_ordered_pair_counts_are_not_blanket_excluded():
    for question in ("The score rose by five points. How many points remain?",
                     "Find a perfect square integer.",
                     "A runner travels 12 miles in two hours. Find the average speed.",
                     "How many integer points satisfy x^2-y^2=17?"):
        assert not b.geometric_reasoning(question)
    assigned, reason = b.provisional(item(subject="counting_and_probability", level=3,
        problem="How many ordered pairs of positive integers have sum 7?", gold_raw="6"))
    assert assigned and not reason and assigned["provisional_domain"] == "counting_probability"
    assert b.provisional(item(problem="Find the ordered pair satisfying both equations."))[0] is None
    for question in ("A square is enlarged to have greater area. Find the new side length.",
                     "Find the area covered by rectangular pictures.",
                     "A cube has a surface-area cost. Find its volume.",
                     "Find the angle between the sides."):
        assert b.geometric_reasoning(question)


def test_publisher_mathml_structure_and_authoritative_endpoints():
    xml = b.ElementTree.fromstring(f'<solution xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"><para id="key"><m:math><m:mrow><m:mi>P</m:mi><m:mo>=</m:mo><m:mfrac><m:mn>175</m:mn><m:mn>2162</m:mn></m:mfrac></m:mrow></m:math></para></solution>')
    assert b.publisher_gold(xml)[0] == "175/2162"
    assert r"\frac{175}{2162}" in b.publisher_text(xml)
    xml = b.ElementTree.fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>151</mn>\n<mo>,</mo>\n<mn>200</mn></mrow></math>')
    assert b.publisher_math(xml) == "151,200"
    for source, expected in [
        ("<msqrt><mi>x</mi><mo>+</mo><mn>1</mn></msqrt>", r"\sqrt{x+1}"),
        ("<msup><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mn>2</mn></msup>", "(x+1)^{2}"),
        ("<munderover><mo>&#8721;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mn>7</mn></munderover>", r"\sum _{k=1}^{7}"),
        ("<mover><mi>x</mi><mo>&#175;</mo></mover>", r"\bar{x}"),
    ]:
        node = b.ElementTree.fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML">' + source + '</math>')
        assert b.publisher_math(node) == expected
    node = b.ElementTree.fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML"><maction><mn>7</mn></maction></math>')
    with pytest.raises(ValueError, match="Unsupported MathML"):
        b.publisher_math(node)


def test_actual_publisher_scalar_keys_and_original_shared_context():
    base = b.RAW / "openstax" / b.OPENSTAX_REVISION
    if not (base / "modules" / "m49448" / "index.cnxml").exists():
        pytest.skip("Pinned publisher sources not acquired")
    rows = [row for module in b.OPENSTAX_MODULES
            for row in b.publisher_items(base / "modules" / module / "index.cnxml")]
    indexed = {r["source_index"]: r for r in rows}
    for eid, gold in {"fs-id1165137772252": "8640", "fs-id1165137735343": "7200",
                      "fs-id1446532": "151200", "fs-id1137436": "175/2162",
                      "fs-id1429776": "5/12"}.items():
        row = indexed[eid]
        assert row["gold_raw"] == gold
        assert row["source_revision"] == b.OPENSTAX_REVISION
        assert row["source_problem_xml"] and row["source_solution_xml"] and row["source_gold_locator"]
        assert b.provisional(row)[0]["provisional_difficulty"] == "medium"
    assert indexed["fs-id1137436"]["source_context_ids"] == ["fs-id962975"]
    assert "12" in indexed["fs-id1137436"]["problem"] and "5" in indexed["fs-id1137436"]["problem"]
    assert "fs-id1425116" not in indexed  # Multipart original, not a generated scalar variant.
    assert not {"fs-id1646993", "ti_11_07_01", "fs-id1704961", "fs-id953004"} & set(indexed)


def test_actual_source_negative_controls_do_not_inflate_ratio_capacity():
    paths = [b.previous.RAW / ("datasets--" + b.previous.SOURCES["math"][0].replace("/", "--"))
             / "snapshots" / b.previous.SOURCES["math"][1] / name for name in b.previous.SOURCES["math"][2]]
    by_id = {b.source_id(row): row for row in b.previous.math_items(paths)}
    for subject, index, expected in [("algebra", 99, "algebra"), ("algebra", 978, "algebra"),
                                    ("number_theory", 128, "number_theory"),
                                    ("counting_and_probability", 214, "counting_probability")]:
        row = by_id[f"EleutherAI/hendrycks_math::{subject}/test::{index}"]
        assert b.content_domain(row)[0] == expected
    for subject, index in [("prealgebra", 42), ("prealgebra", 861), ("algebra", 884),
                           ("algebra", 854), ("prealgebra", 97), ("prealgebra", 809)]:
        row = by_id[f"EleutherAI/hendrycks_math::{subject}/test::{index}"]
        assert b.content_domain(row)[0] is None


def test_public_manifest_whitelist_and_false_preparation_flags():
    metadata = {"builder_sha256": "a" * 64, "review_evidence_sha256": {"secret": "private"},
                "sources": {"fixture": {"repo": "fixture-only", "revision": "b" * 40,
                                         "license": "fixture", "answers": "PRIVATE GOLD 385"}},
                "rejected_item": {"problem": "PRIVATE PAYLOAD", "gold": 385}}
    policy = {"policy_id": "fixture-only", "contract_sha256": "c" * 64}
    manifest = b.public_manifest(metadata, {}, policy)
    assert manifest["sealed"] is False and manifest["review"]["agent_reviewed_all"] is False
    assert manifest["review"]["human_reviewed"] is False and manifest["review"]["reviewed"] is False
    assert "PRIVATE" not in json.dumps(manifest) and "review_evidence_sha256" not in manifest
    assert manifest["input_sha256"] == {} and manifest["review"]["policy_sha256"] == "c" * 64


def test_public_source_tier_schema_preserves_blind_inputs_and_intrinsic_counts(tmp_path, monkeypatch):
    # Hypothetical schema fixture only. Never invoke freeze or register these as actual reviews.
    from jevlab.v3.runner import Experiment

    data = tmp_path / "data" / "v3"
    inputs, labels = {}, {}
    for split, quota in (("test", b.TEST_QUOTA), ("pilot", b.PILOT_QUOTA)):
        rows = [{"id": f"fixture-{split}-{domain}-{difficulty}-{n}",
                 "problem": f"Schema fixture {split} {domain} {difficulty} {n}",
                  "domain": domain, "sampling_tier": difficulty, "difficulty": "easy"}
                for domain in b.DOMAINS for difficulty, count in quota.items() for n in range(count)]
        labels[split] = rows
        inputs[split] = [{"id": r["id"], "problem": r["problem"], "language": "en"} for r in rows]
    inputs["dev"] = [{"id": f"fixture-dev-{n}", "problem": f"Schema fixture dev {n}", "language": "en"} for n in range(20)]
    schedule = b.schedule_for(labels["test"], labels["pilot"])
    subset = b.latency_subset(labels["test"])
    indexed = {r["id"]: r for r in inputs["test"]}
    inputs["latency"] = [indexed[pid] for pid in subset]
    plan = b.latency_plan_for(labels["test"], subset, schedule)
    assert plan["seeds"] == [17, 29, 43] and plan["repetitions"] == 3 and plan["expected_rows"] == 1800
    assert all(set(r) == {"item_id", "problem", "domain", "difficulty", "sampling_tier"} for r in plan["items"])
    assert plan["quota_axis"] == b.QUOTA_AXIS
    assert plan["intrinsic_difficulty_counts"] == {"easy": 100, "medium": 0, "hard": 0}
    for split, rows in inputs.items():
        b.write_jsonl(data / f"{split}_inputs.jsonl", rows)
    b.write_json(data / "schedule.json", schedule)
    b.write_json(data / "latency_plan.json", plan)
    b.write_json(data / "review_manifest.json", {"fixture_only": True, "secret": "not readable by runtime"})
    b.write_json(data / "test_gold.jsonl", {"fixture_only": True, "secret": "not even valid gold"})
    policy = {"policy_id": "fixture-only", "contract_sha256": "c" * 64}
    hashes = {p.name: b.digest(p) for p in data.iterdir() if "inputs" in p.name or p.name in ("schedule.json", "latency_plan.json")}
    counts = {s: {"easy": len(rr), "medium": 0, "hard": 0} for s, rr in labels.items()}
    manifest = b.public_manifest({"builder_sha256": "a" * 64, "sources": {}}, hashes, policy, complete=True, intrinsic_counts=counts)
    assert manifest["review"]["intrinsic_difficulty_counts"] == counts
    assert manifest["quota_axis"] == b.QUOTA_AXIS
    manifest["fixture_only"] = True
    b.write_json(data / "dataset_manifest.json", manifest)
    b.seal(data, list(data.iterdir()))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "gold" in path.name or path.name == "review_manifest.json":
            raise AssertionError("Actual runtime must not read analysis-only artifacts")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    # The sibling latency metadata validator is owned elsewhere; the new schema above is explicit.
    for cls in (Experiment,):
        obj = cls.__new__(cls)
        obj.data_dir, obj.root, obj.cfg = data, tmp_path, {}
        obj.p = {"SPLIT": "test", "N_PROBLEMS": 500}
        obj.config = {"data_sha256": obj._data_hashes()}
        result = obj.problems() if cls is Experiment else obj.latency_items()
        assert len(result) == (500 if cls is Experiment else 100)
        assert all(set(r) == {"id", "problem", "language"} for r in result)


def test_public_publication_cannot_invent_intrinsic_counts_from_source_quotas():
    with pytest.raises(ValueError, match="actual selected intrinsic"):
        b.public_manifest({"builder_sha256": "a" * 64, "sources": {}}, {},
                          {"policy_id": "fixture", "contract_sha256": "b" * 64}, complete=True)


def test_real_exclusion_counts_and_original_dev_preserved():
    used, extra, dev, archives = b.load_exclusions()
    assert len(used) == 260 and len(extra) == 206
    assert len({r["source_id"] for r in extra} - {r["source_id"] for r in used}) == 113
    assert len(archives) == 2 and all(a["candidate_count"] == 140 for a in archives)
    assert dev[0] == b.read_jsonl(ROOT / "data" / "dev_inputs.jsonl")
    assert len(dev[0]) == 20


def test_real_staging_blind_packets_and_no_completed_review_claims():
    stage = b.OUT / "staging"
    if not (stage / "preparation_manifest.json").exists():
        pytest.skip("Preparation not available; policy fixtures still run offline")
    metadata = b.read_json(stage / "preparation_manifest.json")
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    assert len(rows) == metadata["candidate_pool_count"]
    assert b.capacity(rows) == metadata["per_stratum"]
    assert all(r["candidate_sha256"] == b.candidate_sha(r) for r in rows)
    assert all(r["raw_problem"] == r["problem"] for r in rows)
    assert all(abs(Fraction(r["gold_numerator"], r["gold_denominator"])) <= 10**9 for r in rows)
    assert all(set(r["prompt_tokens"]) == set(b.MODELS) and max(r["prompt_tokens"].values()) <= 1024 for r in rows)
    packet_ids = []
    for packet in metadata["packets"]:
        blind = b.read_jsonl(ROOT / packet["blind"])
        reference = b.read_jsonl(ROOT / packet["reference"])
        assert 1 <= len(blind) <= 25
        assert [r["id"] for r in blind] == [r["id"] for r in reference] == packet["candidate_ids"]
        assert all(set(r) == {"id", "problem", "language"} for r in blind)
        assert b.digest(ROOT / packet["blind"]) == packet["blind_sha256"]
        assert b.digest(ROOT / packet["reference"]) == packet["reference_sha256"]
        packet_ids.extend(r["id"] for r in blind)
    assert packet_ids == [r["id"] for r in rows]
    assert all(metadata["review"][field] is False for field in ("agent_reviewed", "agent_reviewed_all", "human_reviewed", "reviewed"))
    preview = b.read_json(stage / "public_manifest_preview.json")
    assert preview["sealed"] is False and preview["review"]["agent_reviewed_all"] is False
    assert preview["review"]["human_reviewed"] is False
    assert metadata["review"]["policy_sha256"] == b.read_json(stage / "review_policy.json")["contract_sha256"]
    registry = b.read_json(stage / "reviews" / "reviewers.json")["reviewers"]
    assert all(r["kind"] == "agent" and r["independent_blind_review"] is True for r in registry)
    assert len({r["reviewer_id"] for r in registry}) == len(registry)
    assert not (b.OUT / "SHA256SUMS").exists()
    assert not list(b.OUT.glob("test_*.jsonl")) and not list(b.OUT.glob("pilot_*.jsonl"))
    policy_inputs = None
    if (metadata["builder_sha256"] != b.digest(Path(b.__file__))
            or metadata["review"]["policy_sha256"] != b.digest(b.OUT / "REVIEW_CONTRACT.md")):
        # The explicitly pending amendment must be tested BEFORE touching the live wrapper.
        snapshots = [p.parent for p in (b.OUT / "policy_amendments").glob("*/snapshot.json")
                     if b.read_json(p)["sha256"].get("data/v3/staging/preparation_manifest.json") == b.digest(stage / "preparation_manifest.json")]
        assert len(snapshots) == 1
        policy_inputs = snapshots[0]
    protected = b.verify_preparation_lineage(stage, metadata, policy_inputs=policy_inputs)
    b.verify_hashes(protected)


def test_real_freeze_missing_reviews_does_not_write_final_files():
    if not (b.OUT / "staging" / "SHA256SUMS").exists():
        pytest.skip("Preparation not available")
    before = {p: b.digest(p) for p in b.OUT.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "data" / "build_dataset_v3.py"), "--freeze"],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and any(reason in result.stderr for reason in
                                         ("Missing actual reviewer registry", "Insufficient twice-reviewed strata", "Staged builder/seed changed"))
    assert before == {p: b.digest(p) for p in b.OUT.rglob("*") if p.is_file()}


def test_real_closed_initial100_preserves_all76_adjudications_and_truthful_two_axes():
    stage = b.OUT / "staging"
    if not (stage / "reviews/adjudications/adjudicators.json").exists():
        pytest.skip("Actual initial adjudications not registered")
    metadata = b.read_json(stage / "preparation_manifest.json")
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    initial = {pid for p in metadata["packets"][:4] for pid in p["candidate_ids"]}
    approved, decisions, _ = b.load_reviews(rows, stage, metadata["packets"])
    initial_rows = [r for r in rows if r["id"] in initial]
    assert len(initial) == 100 and len(approved & initial) == 87
    assert sum(decisions[pid]["status"] == "resolved" for pid in initial) == 76
    assert sum(decisions[pid]["accepted_via_adjudication"] for pid in initial) == 75
    assert sum(decisions[pid]["status"] == "resolved" and not decisions[pid]["accepted"] for pid in initial) == 1
    assert sum(decisions[pid]["status"] == "awaiting_adjudication" for pid in initial) == 0
    intrinsic = b.reviewed_capacity(initial_rows, approved, decisions)
    sampling = b.reviewed_capacity(initial_rows, approved, decisions, axis=b.QUOTA_AXIS)
    assert {f: sum(t[f] for t in intrinsic.values()) for f in b.TEST_QUOTA} == {"easy": 52, "medium": 33, "hard": 2}
    assert {f: sum(t[f] for t in sampling.values()) for f in b.TEST_QUOTA} == {"easy": 23, "medium": 39, "hard": 25}
    assert sum(not review["approved"] for pid in initial for review in decisions[pid]["reviews"]) == 170
