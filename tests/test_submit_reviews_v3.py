"""Temporary schema-only unit fixtures; never consume or fabricate actual reviews."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submit_reviews_v3", ROOT / "tools/submit_reviews_v3.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
b = s.b


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    stage = tmp_path / "data/v3/staging"
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "OUT", stage.parent)
    for name in ("REVIEW_CONTRACT.md", "ELIGIBILITY_POLICY.md"):
        target = stage.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "data/v3" / name).read_bytes())
    b.write_json(tmp_path / "results/v3/implementation_status.json", {
        "v3_model_inference_executed": False, "pilot_executed": False, "dataset_frozen": False})
    rows = []
    for pid in ("schema-approve", "schema-veto", "schema-key-disagreement", "schema-missing"):
        row = {"id": pid, "problem": "Five boxes each contain two toys. How many toys are there?", "language": "en",
               "provisional_domain": "arithmetic", "provisional_difficulty": "medium", "gold_numerator": 10,
               "gold_denominator": 1, "source_family": "schema-only", "fixture_only": True,
               "source": "schema-source", "source_split": "schema-test", "source_index": pid,
               "source_id": "schema-source::schema-test::" + pid, "gold_raw": "10", "solution": "Schema-only source solution"}
        row["candidate_sha256"] = b.candidate_sha(row)
        rows.append(row)
    b.write_jsonl(stage / "candidate_pool.jsonl", rows)
    packet_rows = rows[:3]
    blind_path = stage / "packets/batch_001.blind.jsonl"
    reference_path = stage / "packets/batch_001.reference.jsonl"
    b.write_jsonl(blind_path, [{key: row[key] for key in ("id", "problem", "language")} for row in packet_rows])
    b.write_jsonl(reference_path, packet_rows)
    packet = {"batch": "batch_001", "candidate_ids": [row["id"] for row in packet_rows],
              "blind": blind_path.relative_to(tmp_path).as_posix(), "blind_sha256": b.digest(blind_path),
              "reference": reference_path.relative_to(tmp_path).as_posix(), "reference_sha256": b.digest(reference_path)}
    b.write_json(tmp_path / "source-lock.json", {"fixture_only": True})
    policy = b.review_policy()
    b.write_json(stage / "review_policy.json", policy)
    metadata = {"packets": [packet], "construction_seed": b.SEED, "latency_seed": b.LATENCY_SEED,
                "quota_axis": b.QUOTA_AXIS,
                "builder_sha256": b.digest(Path(b.__file__)), "previous_artifact_sha256": {}, "source_sha256": {},
                "source_lock_path": "source-lock.json", "source_lock_sha256": b.digest(tmp_path / "source-lock.json"),
                "review_policy_json_sha256": b.digest(stage / "review_policy.json"), "sources": {},
                "review": {"policy_id": policy["policy_id"], "policy_sha256": policy["contract_sha256"],
                           "agent_reviewed": False, "agent_reviewed_all": False, "human_reviewed": False, "reviewed": False}}
    b.write_json(stage / "preparation_manifest.json", metadata)
    b.write_json(stage / "public_manifest_preview.json", b.public_manifest(metadata, {}, policy))
    b.write_json(stage / "reviews/reviewers.json", {"reviewers": []})
    b.seal(stage, [p for p in stage.rglob("*") if p.is_file() and not p.is_relative_to(stage / "reviews")])
    return tmp_path, stage, rows, packet


def emitted_pair(pid, reviewer, rejected=False, key_disagreement=False):
    blind = {"id": pid, "reviewer_id": reviewer, "phase": "blind", "fixture_only": True,
             "independent_solution": "Local schema-only unit test: each of five boxes contains two toys, and multiplying five by two gives ten toys altogether, with no extra toys outside these boxes.",
             "answer": {"numerator": 11 if key_disagreement else 10, "denominator": 1},
             "domain": "outside_scope" if rejected else "arithmetic", "difficulty": "easy" if rejected else "medium",
             "domain_reason": "Local schema-only unit test checks a domain mismatch without changing the independent judgment or pretending this is an actual mathematical review.",
             "difficulty_reason": "Local schema-only unit test checks a difficulty mismatch while preserving the original boolean verdict and complete explanatory text for audit.",
             **{field: True for field in s.BLIND_VERDICTS}}
    if rejected:
        blind.update(domain_ok=False, difficulty_ok=False)
    reference = {"id": pid, "reviewer_id": reviewer, "phase": "reference", "fixture_only": True,
                 "blind_record_sha256": b.digest_bytes(b.canonical(blind).encode("utf-8")),
                 "reference_comparison": "Local schema-only unit test compares the independently returned rational answer against a hypothetical key and keeps every negative judgment rather than repairing its payload.",
                 "gold_matches": not key_disagreement, "solution_consistent": True, "approve": not (rejected or key_disagreement)}
    return blind, reference


def archive(sandbox, reviewer="schema-reviewer-a", reject=True):
    root, stage, rows, _ = sandbox
    pairs = [emitted_pair(row["id"], reviewer, rejected=reject and row["id"] == "schema-veto",
                          key_disagreement=reject and row["id"] == "schema-key-disagreement") for row in rows[:3]]
    paths = [stage / f"reviews/evidence/{reviewer}-{phase}.jsonl" for phase in ("blind", "reference")]
    # Deliberately different event ordering: submission must preserve each emitted file's order.
    b.write_jsonl(paths[0], [pair[0] for pair in pairs])
    b.write_jsonl(paths[1], [pair[1] for pair in reversed(pairs)])
    return paths


def deliver(sandbox, reviewer="schema-reviewer-a", run="local-unit-execution-a", paths=None):
    paths = paths or archive(sandbox, reviewer)
    return s.submit("batch_001", reviewer, run, *paths, root=sandbox[0])


def inventory(directory):
    return {path.relative_to(directory).as_posix(): b.digest(path) for path in directory.rglob("*") if path.is_file()}


def test_all_events_bound_without_mutating_raw_and_idempotent(sandbox):
    root, stage, rows, packet = sandbox
    paths = archive(sandbox)
    original = {path: path.read_bytes() for path in paths}
    result = deliver(sandbox, paths=paths)
    assert result["idempotent"] is False and result["records"] == 6
    assert result["approved_reviews"] == 1 and result["rejected_reviews"] == 2
    assert result["raw_unchanged"] is True
    output = root / result["bound_path"]
    bound = b.read_jsonl(output)
    raw = b.read_jsonl(paths[0]) + b.read_jsonl(paths[1])
    assert [b.review_core(event) for event in bound] == [b.review_core(event) for event in raw]
    assert all(set(event) - set(original_event) <= {"candidate_sha256", "packet_sha256", "blind_record_sha256"}
               for event, original_event in zip(bound, raw))
    blinds = {event["id"]: event for event in bound if event["phase"] == "blind"}
    for event in bound:
        assert event["packet_sha256"] == packet[event["phase"] + "_sha256"]
        if event["phase"] == "reference":
            assert event["blind_record_sha256"] == b.digest_bytes(b.canonical(blinds[event["id"]]).encode("utf-8"))
            assert event["blind_record_sha256"] != raw[3 + list(reversed(packet["candidate_ids"])).index(event["id"])]["blind_record_sha256"]
            assert b.validate_review(next(row for row in rows if row["id"] == event["id"]), blinds[event["id"]], event) is (event["id"] == "schema-approve")
    registration = b.read_json(stage / "reviews/reviewers.json")["reviewers"][0]
    assert registration["agent_run_id"] == "local-unit-execution-a"
    assert registration["evidence"] == [{"path": path.relative_to(stage / "reviews").as_posix(), "sha256": b.digest(path)} for path in paths]
    before = inventory(stage)
    repeat = deliver(sandbox, paths=paths)
    assert repeat["idempotent"] is True and repeat["status"] == "already_submitted"
    assert inventory(stage) == before
    assert {path: path.read_bytes() for path in paths} == original
    assert not list(root.glob("data/v3/test_*.jsonl"))


def test_progress_matches_builder_and_another_approval_cannot_erase_veto(sandbox):
    _, stage, rows, packet = sandbox
    deliver(sandbox)
    second = archive(sandbox, "schema-reviewer-b", reject=False)
    result = deliver(sandbox, "schema-reviewer-b", "local-unit-execution-b", second)
    approved, decisions, hashes = b.load_reviews(rows, stage, [packet])
    assert approved == {"schema-approve"}
    assert len(decisions["schema-approve"]["reviews"]) == 2
    progress = b.read_json(stage / "reviews/review_progress.json")
    assert progress["approved_candidate_count"] == result["approved_candidates"] == 1
    assert progress["rejected_candidate_count"] == result["rejected_candidates"] == 2
    assert progress["completed_review_count"] == 6 and progress["emitted_record_count"] == 12
    assert progress["approved_review_count"] == 4 and progress["rejected_review_count"] == 2
    assert progress["missing_candidate_count"] == 1 and progress["missing_review_count"] == 2
    assert progress["evidence_sha256"] == hashes
    assert progress["status"] == "blocked_dataset_review" and progress["sealed"] is False
    reasons = {event["id"]: event["reviews"][0]["reasons"] for event in progress["rejected_candidates"]}
    assert {"domain_ok", "difficulty_ok", "domain_mismatch", "difficulty_mismatch"} <= set(reasons["schema-veto"])
    assert {"gold_matches", "approve", "answer_mismatch"} <= set(reasons["schema-key-disagreement"])
    assert len(list((stage / "reviews").glob("*.jsonl"))) == 2


@pytest.mark.parametrize("change", ["unknown_id", "unknown_reviewer", "unknown_phase", "duplicate", "missing", "nonobject", "bad_text", "false_not_bool", "approval_mismatch", "answer_extra_key"])
def test_invalid_events_refused_before_writing_generated_data(sandbox, change):
    paths = archive(sandbox)
    events = b.read_jsonl(paths[0])
    if change == "unknown_id":
        events[0]["id"] = "not-staged"
    elif change == "unknown_reviewer":
        events[0]["reviewer_id"] = "another-unit-reviewer"
    elif change == "unknown_phase":
        events[0]["phase"] = "revised"
    elif change == "duplicate":
        events[1] = events[0]
    elif change == "missing":
        events.pop()
    elif change == "nonobject":
        events[0] = []
    elif change == "bad_text":
        events[0]["independent_solution"] = "approve"
    elif change == "false_not_bool":
        events[0]["correctness"] = 0
    elif change == "answer_extra_key":
        events[0]["answer"]["unit"] = "toys"
    else:
        events[0]["difficulty"] = "hard"
    paths[0].write_text("\n".join(b.canonical(event) for event in events) + "\n", encoding="utf-8")
    before = inventory(sandbox[1])
    with pytest.raises(ValueError):
        deliver(sandbox, paths=paths)
    assert inventory(sandbox[1]) == before


@pytest.mark.parametrize("collision", ["run", "raw", "registry", "bound", "duplicate_bound", "other_raw", "other_bound"])
def test_collisions_and_mutated_evidence_refused_without_repair(sandbox, collision):
    root, stage, _, _ = sandbox
    paths = archive(sandbox)
    result = deliver(sandbox, paths=paths)
    run = "local-unit-execution-a"
    registry_path = stage / "reviews/reviewers.json"
    output = root / result["bound_path"]
    if collision == "run":
        run = "changed-unit-execution"
    elif collision == "raw":
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
    elif collision == "registry":
        registry = b.read_json(registry_path)
        registry["reviewers"][0]["independent_blind_review"] = False
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
    elif collision == "bound":
        events = b.read_jsonl(output)
        events[0]["domain_reason"] += " Changed local test reason."
        output.write_text("\n".join(b.canonical(event) for event in events) + "\n", encoding="utf-8")
    elif collision == "duplicate_bound":
        b.write_jsonl(stage / "reviews/b001-duplicate.jsonl", b.read_jsonl(output))
    else:
        other = archive(sandbox, "schema-reviewer-b")
        other_result = deliver(sandbox, "schema-reviewer-b", "local-unit-execution-b", other)
        changed = other[0] if collision == "other_raw" else root / other_result["bound_path"]
        changed.write_bytes(changed.read_bytes() + (b"\n" if collision == "other_raw" else b'{}\n'))
    before = inventory(stage)
    with pytest.raises(ValueError):
        deliver(sandbox, run=run, paths=paths)
    assert inventory(stage) == before


def test_run_id_reuse_and_unarchived_paths_refused(sandbox):
    deliver(sandbox)
    paths = archive(sandbox, "schema-reviewer-b")
    before = inventory(sandbox[1])
    with pytest.raises(ValueError, match="distinct execution"):
        deliver(sandbox, "schema-reviewer-b", "local-unit-execution-a", paths)
    with pytest.raises(ValueError, match="archived"):
        deliver(sandbox, "schema-reviewer-b", "local-unit-execution-b", [sandbox[1] / "candidate_pool.jsonl", paths[1]])
    with pytest.raises(ValueError, match="packet"):
        s.submit("batch_999", "schema-reviewer-b", "local-unit-execution-b", *paths, root=sandbox[0])
    assert inventory(sandbox[1]) == before


def test_cli_local_submission_and_error(sandbox, capsys):
    paths = archive(sandbox)
    args = ["--packet", "batch_001", "--reviewer", "schema-reviewer-a", "--agent-run-id", "local-unit-execution-a",
            "--blind", str(paths[0]), "--reference", str(paths[1]), "--root", str(sandbox[0])]
    assert s.main(args) == 0
    assert json.loads(capsys.readouterr().out)["raw_unchanged"] is True
    assert s.main(args) == 0
    assert json.loads(capsys.readouterr().out)["idempotent"] is True
    args[1] = "batch_999"
    assert s.main(args) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "blocked_review_submission"


def register_classification_pairs(sandbox, unit_false=False, defect=None):
    # Persist the vetoes under a declared historical rule, then explicitly migrate.
    root, stage, _, _ = sandbox
    historical_policy = {**b.review_policy(), "policy_id": "fixture-historical-source-hint-contract"}
    historical_policy.pop("quota_axis")
    with patch.object(b, "review_policy", return_value=historical_policy):
        b.replace_generated(stage / "review_policy.json", historical_policy)
        metadata = b.read_json(stage / "preparation_manifest.json")
        metadata.pop("quota_axis")
        metadata["review"].update(policy_id=historical_policy["policy_id"], policy_sha256=historical_policy["contract_sha256"])
        metadata["review_policy_json_sha256"] = b.digest(stage / "review_policy.json")
        b.replace_generated(stage / "preparation_manifest.json", metadata)
        marker = stage / "SHA256SUMS"
        marker.chmod(0o644)
        marker.unlink()
        b.seal(stage, [p for p in stage.rglob("*") if p.is_file() and not p.is_relative_to(stage / "reviews")])
        _register_historical_pairs(sandbox, unit_false, defect)
    (root / "data/build_dataset_v3.py").write_bytes(Path(b.__file__).read_bytes())
    reason = "Explicit local schema-only source-tier amendment preserves every original historical false verdict, identity and gold"
    snapshot = s.archive_policy_inputs(reason, root)
    b.amend_review_policy_before_inference(reason, Path(snapshot["archive"]))


def _register_historical_pairs(sandbox, unit_false, defect):
    for number, reviewer in enumerate(("schema-reviewer-a", "schema-reviewer-b")):
        paths = archive(sandbox, reviewer, reject=False)
        blind, reference = (b.read_jsonl(p) for p in paths)
        for event in blind:
            event["difficulty"] = "easy" if number == 0 else "hard"
            event["unit_explicit"] = not unit_false
            if defect and event["id"] == "schema-approve" and number == 0:
                if defect in s.BLIND_VERDICTS:
                    event[defect] = False
                elif defect == "answer":
                    event["answer"]["numerator"] = 11
        for event in reference:
            event["approve"] = False
            source = next(e for e in blind if e["id"] == event["id"])
            event["blind_record_sha256"] = b.digest_bytes(b.canonical(source).encode("utf-8"))
            if defect in ("gold_matches", "solution_consistent", "answer") and event["id"] == "schema-approve" and number == 0:
                event["gold_matches" if defect == "answer" else defect] = False
        for path, events in zip(paths, (blind, reference)):
            path.write_text("\n".join(b.canonical(e) for e in events) + "\n", encoding="utf-8")
        deliver(sandbox, reviewer, f"local-unit-execution-{number}", paths)


def adjudication_event(sandbox, pid="schema-approve", reviewer="schema-adjudicator", approve=True, unit_false=False):
    _, stage, rows, packet = sandbox
    _, decisions, _ = b.load_reviews(rows, stage, [packet], include_adjudications=False)
    decision = decisions[pid]
    return {"id": pid, "reviewer_id": reviewer, "phase": "adjudication", "fixture_only": True,
            "original_review_refs": b.original_review_refs(decision["reviews"]),
            "source_math_validation": "Local schema-only unit test preserves the source question and original key: five boxes contain two toys each, so the exact product is ten. Both independent proofs use the same valid multiplication without introducing a new question, ambiguity, source identity or gold value.",
            "answer": {"numerator": 10, "denominator": 1},
            **{k: True for k in ("answer_correct", "gold_matches", "solution_consistent", "statement_ok", "unique_answer", "correctness")},
            "approve": approve, "final_domain": "arithmetic", "final_difficulty": "easy",
            "domain_reason": "Local schema-only test: ordinary multiplication is arithmetic, with no substantive algebra, ratios, integer structure or probabilistic event.",
            "difficulty_reason": "Local schema-only test: one direct operation is easy under the intrinsic rubric, regardless of source hints or requested sampling counts.",
            "resolution_reason": "Local schema-only test: the original mathematical solutions and source key agree, and the original false approval concerns only provisional difficulty and any explicitly recorded dimensionless interpretation; original false events remain preserved.",
            "unit_resolution": "inherently_dimensionless" if unit_false else "unchanged",
            "dimensionless_query": True, "dimensionless_kind": "count",
            "unit_reason": "Local schema-only test: the uniquely requested number of toys is a dimensionless count of discrete objects, with no monetary denomination or measurement conversion to resolve.",
            "query_quote": "How many toys are there?",
            "original_unit_concerns": [{"reviewer_id": r["reviewer_id"], "blind_record_sha256": r["blind_record_sha256"],
                "unit_explicit": False, "original_unit_reason": r["original_unit_reason"]}
                for r in decision["reviews"] if r["original_verdicts"]["blind"]["unit_explicit"] is False]}


def archive_adjudication(sandbox, event):
    path = sandbox[1] / "reviews/evidence/schema-adjudication.jsonl"
    b.write_jsonl(path, [event])
    return path


@pytest.mark.parametrize("unit_false", [False, True])
def test_third_actual_registry_resolves_labels_retains_every_original_veto(sandbox, unit_false):
    register_classification_pairs(sandbox, unit_false=unit_false)
    root, stage, rows, packet = sandbox
    original = {p: p.read_bytes() for p in stage.rglob("*.jsonl")}
    progress = s.refresh_progress(root)
    assert progress["awaiting_adjudication_candidate_count"] == 3
    assert progress["rejected_candidate_count"] == 0 and progress["preserved_veto_candidate_count"] == 3
    output = stage / "reviews/adjudication_packets/local-packet.jsonl"
    result = s.prepare_adjudication_packet(output, ["schema-approve"], root)
    assert result["quota_visibility"] is False and result["candidate_count"] == 1
    assert "quota" not in output.read_text(encoding="utf-8") and "priority" not in output.read_text(encoding="utf-8")
    event = adjudication_event(sandbox, unit_false=unit_false)
    path = archive_adjudication(sandbox, event)
    raw = path.read_bytes()
    result = s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), root)
    assert result["accepted"] == 1 and result["awaiting_adjudication_candidates"] == 2
    approved, decisions, hashes = b.load_reviews(rows, stage, [packet])
    assert approved == {"schema-approve"}
    resolved = decisions["schema-approve"]
    assert resolved["status"] == "resolved" and resolved["accepted_via_adjudication"] is True
    assert resolved["final_difficulty"] == "easy" and rows[0]["provisional_difficulty"] == "medium"
    assert all(r["original_verdicts"]["reference"]["approve"] is False for r in resolved["reviews"])
    assert all(r["original_verdicts"]["blind"]["unit_explicit"] is (not unit_false) for r in resolved["reviews"])
    assert all(p.read_bytes() == raw_bytes for p, raw_bytes in original.items()) and path.read_bytes() == raw
    assert b.reviewed_capacity(rows, approved, decisions)["arithmetic"]["easy"] == 1
    before = inventory(stage)
    repeat = s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), root)
    assert repeat["idempotent"] is True and before == inventory(stage)
    assert stage.joinpath("reviews/adjudications/adjudicators.json").relative_to(root).as_posix() in hashes
    assert not list(root.glob("data/v3/test_*.jsonl"))


@pytest.mark.parametrize("defect", ["correctness", "statement_ok", "unique_answer", "gold_matches", "solution_consistent", "answer"])
def test_no_adjudication_override_of_unsolved_ambiguous_wronggold_or_invalid_math(sandbox, defect):
    register_classification_pairs(sandbox, defect=defect)
    event = adjudication_event(sandbox)
    path = archive_adjudication(sandbox, event)
    before = inventory(sandbox[1])
    with pytest.raises(ValueError, match="cannot override"):
        s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), sandbox[0])
    assert before == inventory(sandbox[1])


@pytest.mark.parametrize("change", ["gold", "refs", "run", "run_whitespace", "false_math", "query", "units", "unit_reason", "quota", "raw_sha", "unknown_dimension", "answer_extra_key"])
def test_adjudication_false_units_hash_refs_original_gold_and_no_quota_guards(sandbox, change):
    register_classification_pairs(sandbox, unit_false=True)
    event = adjudication_event(sandbox, unit_false=True)
    run = "local-third-execution"
    if change == "gold":
        event["answer"]["numerator"] = 99
    elif change == "refs":
        event["original_review_refs"].pop()
    elif change == "run":
        run = "local-unit-execution-0"
    elif change == "run_whitespace":
        run = " local-unit-execution-0 "
    elif change == "false_math":
        event["correctness"] = False
    elif change == "query":
        event["query_quote"] = "Not the original statement"
    elif change == "units":
        event["unit_resolution"] = "unchanged"
    elif change == "unit_reason":
        event["original_unit_concerns"][0]["original_unit_reason"] = "Rewritten old reason"
    elif change == "quota":
        event["quota_target"] = "hard"
    elif change == "unknown_dimension":
        event["dimensionless_query"] = False
    elif change == "answer_extra_key":
        event["answer"]["unit"] = "toys"
    path = archive_adjudication(sandbox, event)
    before = inventory(sandbox[1])
    sha = "a" * 64 if change == "raw_sha" else b.digest(path)
    with pytest.raises(ValueError):
        s.submit_adjudication(event["reviewer_id"], run, path, sha, sandbox[0])
    assert before == inventory(sandbox[1])


def test_rejected_adjudication_is_registered_and_not_retried(sandbox):
    register_classification_pairs(sandbox)
    event = adjudication_event(sandbox, approve=False)
    path = archive_adjudication(sandbox, event)
    s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), sandbox[0])
    progress = s.refresh_progress(sandbox[0])
    assert progress["resolved_adjudication_candidate_count"] == progress["rejected_candidate_count"] == 1
    assert progress["approved_candidate_count"] == 0 and progress["preserved_veto_candidate_count"] == 3
    event["approve"] = True
    event["reviewer_id"] = "schema-adjudicator-other"
    other = sandbox[1] / "reviews/evidence/another-adjudicator.jsonl"
    b.write_jsonl(other, [event])
    before = inventory(sandbox[1])
    with pytest.raises(ValueError, match="already resolved"):
        s.submit_adjudication(event["reviewer_id"], "local-fourth-execution", other, b.digest(other), sandbox[0])
    assert before == inventory(sandbox[1])


def test_original_dimensionful_unit_cannot_be_overridden(sandbox):
    register_classification_pairs(sandbox, unit_false=True)
    _, stage, rows, packet = sandbox
    _, decisions, _ = b.load_reviews(rows, stage, [packet])
    event = adjudication_event(sandbox, unit_false=True)
    event["candidate_sha256"] = rows[0]["candidate_sha256"]
    with pytest.raises(ValueError, match="ONLY for inherently dimensionless"):
        b.validate_adjudication({**rows[0], "answer_unit": "dollars"}, decisions[rows[0]["id"]], event)


@pytest.mark.parametrize("query", ["How many dollars are there?", "Find the distance in meters.", "What percentage of the total is remaining?"])
def test_missing_source_unit_metadata_is_not_evidence_of_dimensionlessness(sandbox, query):
    register_classification_pairs(sandbox, unit_false=True)
    _, stage, rows, packet = sandbox
    _, decisions, _ = b.load_reviews(rows, stage, [packet])
    event = adjudication_event(sandbox, unit_false=True)
    event.update(query_quote=query, candidate_sha256=rows[0]["candidate_sha256"])
    with pytest.raises(ValueError, match="ONLY for inherently dimensionless"):
        b.validate_adjudication({**rows[0], "problem": query, "answer_unit": None}, decisions[rows[0]["id"]], event)


def test_stale_emitted_reference_blind_hash_is_not_silently_rebound(sandbox):
    paths = archive(sandbox)
    events = b.read_jsonl(paths[1])
    events[0]["blind_record_sha256"] = "a" * 64
    paths[1].write_text("\n".join(b.canonical(e) for e in events) + "\n", encoding="utf-8")
    before = inventory(sandbox[1])
    with pytest.raises(ValueError, match="Emitted reference hash"):
        deliver(sandbox, paths=paths)
    assert inventory(sandbox[1]) == before


@pytest.mark.parametrize("change", ["raw", "bound", "omit", "registry_run", "original_review"])
def test_registered_adjudication_evidence_tampering_fails_closed(sandbox, change):
    register_classification_pairs(sandbox)
    event = adjudication_event(sandbox)
    path = archive_adjudication(sandbox, event)
    result = s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), sandbox[0])
    output = sandbox[0] / result["bound_path"]
    if change == "raw":
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "bound":
        record = b.read_jsonl(output)[0]
        record["final_difficulty"] = "hard"
        output.write_text(b.canonical(record) + "\n", encoding="utf-8")
    elif change == "omit":
        output.write_text("", encoding="utf-8")
    elif change == "registry_run":
        registry_path = sandbox[1] / "reviews/adjudications/adjudicators.json"
        registry = b.read_json(registry_path)
        registry["adjudicators"][0]["agent_run_id"] = "local-unit-execution-0"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
    else:
        original = sandbox[1] / "reviews/b001-schema-reviewer-a.jsonl"
        original.write_bytes(original.read_bytes() + b"{}\n")
    before = inventory(sandbox[1])
    with pytest.raises(ValueError):
        s.refresh_progress(sandbox[0])
    assert before == inventory(sandbox[1])


def test_unexplained_approval_veto_without_label_or_unit_issue_is_not_adjudicable(sandbox):
    for number, rid in enumerate(("schema-reviewer-a", "schema-reviewer-b")):
        paths = archive(sandbox, rid, reject=False)
        reference = b.read_jsonl(paths[1])
        for event in reference:
            event["approve"] = False
        paths[1].write_text("\n".join(b.canonical(e) for e in reference) + "\n", encoding="utf-8")
        deliver(sandbox, rid, f"local-unit-execution-{number}", paths)
    progress = s.refresh_progress(sandbox[0])
    assert progress["awaiting_adjudication_candidate_count"] == 0 and progress["rejected_candidate_count"] == 3


def test_policy_migration_preserves_old_vetoes_source_identity_and_complete_snapshot(sandbox):
    register_classification_pairs(sandbox)
    root, stage, rows, packet = sandbox
    builder = root / "data/build_dataset_v3.py"
    builder.write_bytes(Path(b.__file__).read_bytes())
    original = inventory(stage)
    reason = "User-authorized pre-inference uniform policy amendment retains every original source identity, source gold and actual veto event"
    snapshot = s.archive_policy_inputs(reason, root)
    old_contract = (b.OUT / "REVIEW_CONTRACT.md").read_bytes()
    (b.OUT / "REVIEW_CONTRACT.md").write_bytes(old_contract + b"\nLocal schema-only policy amendment.\n")
    lineage = b.amend_review_policy_before_inference(reason, Path(snapshot["archive"]))
    archived = root / snapshot["archive"]
    assert inventory(archived / "data/v3/staging") == original
    assert (archived / "data/v3/REVIEW_CONTRACT.md").read_bytes() == old_contract
    assert (archived / "data/build_dataset_v3.py").read_bytes() == builder.read_bytes()
    assert b.read_json(stage / "preparation_manifest.json")["previous_preparation"] == lineage
    b.verify_preparation_lineage(stage)
    assert b.read_jsonl(stage / "candidate_pool.jsonl") == rows
    assert b.read_jsonl(root / packet["reference"]) == rows[:3]
    approved, decisions, _ = b.load_reviews(rows, stage, [packet])
    assert not approved and all(d["status"] == "awaiting_adjudication" for d in decisions.values())
    assert all(not r["approved"] for d in decisions.values() for r in d["reviews"])
    assert b.read_json(stage / "public_manifest_preview.json")["review"]["policy_sha256"] == b.digest(b.OUT / "REVIEW_CONTRACT.md")
    before = inventory(stage)
    with pytest.raises(ValueError, match="already applied"):
        b.amend_review_policy_before_inference(reason, Path(snapshot["archive"]))
    assert before == inventory(stage) and not (b.OUT / "SHA256SUMS").exists()
    altered = archived / "data/v3/staging/reviews/evidence/schema-reviewer-a-blind.jsonl"
    altered.chmod(0o644)
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Invalid seal"):
        b.verify_preparation_lineage(stage)


@pytest.mark.parametrize("artifact", ["results/v3/inference.jsonl", "data/v3/test_inputs.jsonl", "data/v3/SHA256SUMS", "data/v3/results"])
def test_policy_migration_refuses_inference_results_and_publication_without_writes(sandbox, artifact):
    root, stage, _, _ = sandbox
    target = root / artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    if artifact.endswith("/results"):
        target.mkdir()
    else:
        target.write_text("{}\n", encoding="utf-8")
    before = inventory(stage)
    with pytest.raises(ValueError, match="forbidden"):
        b.amend_review_policy_before_inference("User authorized original source preserving policy amendment before any inference and actual results", Path("missing"))
    assert before == inventory(stage)


@pytest.mark.parametrize("defect", ["missing_status", "missing_status_flag", "missing_lock", "unsealed_archive"])
def test_migration_preflight_failure_preserves_every_staging_byte(sandbox, defect):
    register_classification_pairs(sandbox)
    root, stage, _, _ = sandbox
    (root / "data/build_dataset_v3.py").write_bytes(Path(b.__file__).read_bytes())
    reason = "User-authorized pre-inference uniform policy amendment retains every original source identity and actual veto event"
    snapshot = s.archive_policy_inputs(reason, root)
    contract = b.OUT / "REVIEW_CONTRACT.md"
    contract.write_bytes(contract.read_bytes() + b"\nLocal schema-only amendment before preflight.\n")
    if defect == "missing_status":
        (root / "results/v3/implementation_status.json").unlink()
    elif defect == "missing_status_flag":
        (root / "results/v3/implementation_status.json").write_text("{}", encoding="utf-8")
    elif defect == "missing_lock":
        (root / "source-lock.json").unlink()
    else:
        (root / snapshot["archive"] / "unsealed.json").write_text("{}", encoding="utf-8")
    before = inventory(stage)
    with pytest.raises((ValueError, FileNotFoundError)):
        b.amend_review_policy_before_inference(reason, Path(snapshot["archive"]))
    assert before == inventory(stage)


def test_interrupted_policy_migration_restores_only_original_generated_descriptors(sandbox, monkeypatch):
    register_classification_pairs(sandbox)
    root, stage, _, _ = sandbox
    (root / "data/build_dataset_v3.py").write_bytes(Path(b.__file__).read_bytes())
    reason = "User-authorized pre-inference uniform policy amendment retains every original source identity and actual veto event"
    snapshot = s.archive_policy_inputs(reason, root)
    contract = b.OUT / "REVIEW_CONTRACT.md"
    contract.write_bytes(contract.read_bytes() + b"\nLocal schema-only amendment before interruption.\n")
    original = inventory(stage)
    def interrupted(_):
        raise OSError("Local schema-only fsync interruption")
    monkeypatch.setattr(b.os, "fsync", interrupted)
    with pytest.raises(OSError, match="fsync interruption"):
        b.amend_review_policy_before_inference(reason, Path(snapshot["archive"]))
    assert original == inventory(stage)
    assert b.read_json(stage / "preparation_manifest.json") == b.read_json(
        root / snapshot["archive"] / "data/v3/staging/preparation_manifest.json")
    b.verify_seal(stage)


def register_fresh_contrast_pairs(sandbox, disagree=False, old_veto=False):
    for number, rid in enumerate(("schema-reviewer-a", "schema-reviewer-b")):
        paths = archive(sandbox, rid, reject=False)
        blind, reference = (b.read_jsonl(p) for p in paths)
        for event in blind:
            event.update(domain="algebra", difficulty="hard" if disagree and number else "easy")
        for event in reference:
            event["approve"] = not old_veto
            original = next(e for e in blind if e["id"] == event["id"])
            event["blind_record_sha256"] = b.digest_bytes(b.canonical(original).encode("utf-8"))
            event["reference_comparison"] += " The source hint differs from these intrinsic labels but does not invalidate the exact independent result."
        for path, events in zip(paths, (blind, reference)):
            path.write_text("\n".join(b.canonical(e) for e in events) + "\n", encoding="utf-8")
        deliver(sandbox, rid, f"local-fresh-contrast-{number}", paths)


def test_fresh_agreeing_intrinsic_reviews_approve_without_source_hint_adjudication(sandbox):
    original_candidates = (sandbox[1] / "candidate_pool.jsonl").read_bytes()
    register_fresh_contrast_pairs(sandbox)
    root, stage, rows, packet = sandbox
    approved, decisions, _ = b.load_reviews(rows, stage, [packet])
    assert approved == {r["id"] for r in rows[:3]}
    assert all(d["decision_type"] == "source_hint_contrast" and d["status"] == "approved"
               and d["sampling_tier"] == "medium" and d["final_domain"] == "algebra" and d["final_difficulty"] == "easy"
               and d["adjudication"] is None for d in decisions.values())
    registry = b.read_json(stage / "reviews/reviewers.json")["reviewers"]
    assert all(r["review_policy"]["id"] == b.REVIEW_POLICY_ID for r in registry)
    for r in registry:
        raw = [e for entry in r["evidence"] for e in b.read_jsonl(stage / "reviews" / entry["path"])]
        bound = b.read_jsonl(stage / "reviews" / ("b001-" + r["reviewer_id"] + ".jsonl"))
        assert [b.review_core(e) for e in raw] == [b.review_core(e) for e in bound]
        assert all(e["domain"] == "algebra" and e["difficulty"] == "easy" for e in raw if e["phase"] == "blind")
    assert (stage / "candidate_pool.jsonl").read_bytes() == original_candidates
    progress = s.refresh_progress(root)
    assert progress["accepted_sampling_strata"]["algebra"]["medium"] == 3
    assert progress["accepted_reviewed_strata"]["algebra"]["easy"] == 3
    assert progress["accepted_via_adjudication_count"] == 0 and progress["awaiting_adjudication_candidate_count"] == 0


def test_fresh_reviewers_disagreeing_intrinsic_labels_require_actual_third_agent(sandbox):
    register_fresh_contrast_pairs(sandbox, disagree=True)
    root, stage, rows, packet = sandbox
    approved, decisions, _ = b.load_reviews(rows, stage, [packet])
    assert not approved and all(d["status"] == "awaiting_adjudication" for d in decisions.values())
    assert all(r["approved"] for d in decisions.values() for r in d["reviews"])
    event = adjudication_event(sandbox)
    path = archive_adjudication(sandbox, event)
    s.submit_adjudication(event["reviewer_id"], "local-actual-third-schema", path, b.digest(path), root)
    approved, decisions, _ = b.load_reviews(rows, stage, [packet])
    assert approved == {"schema-approve"}
    assert decisions["schema-approve"]["decision_type"] == "adjudication"


@pytest.mark.parametrize("false_field", ["domain_ok", "difficulty_ok", "unit_explicit", "approve"])
def test_new_false_verdicts_cannot_use_old_source_hint_or_unit_exceptions(sandbox, false_field):
    register_fresh_contrast_pairs(sandbox)
    _, stage, rows, packet = sandbox
    _, decisions, _ = b.load_reviews(rows, stage, [packet])
    raw = b.read_jsonl(stage / "reviews/evidence/schema-reviewer-a-blind.jsonl")
    refs = b.read_jsonl(stage / "reviews/evidence/schema-reviewer-a-reference.jsonl")
    blind = next(e for e in raw if e["id"] == "schema-approve")
    ref = next(e for e in refs if e["id"] == "schema-approve")
    if false_field == "approve":
        ref["approve"] = False
    else:
        blind[false_field] = False
    assert b.adjudicable_pair(rows[0], blind, ref, source_hint_agreement_required=False) is False
    assert decisions["schema-approve"]["accepted"] is True  # Original fixture remains untouched.


def test_removing_fresh_policy_binding_cannot_grant_historical_veto_exceptions(sandbox):
    register_fresh_contrast_pairs(sandbox, old_veto=True)
    root, stage, rows, packet = sandbox
    registry_path = stage / "reviews/reviewers.json"
    registry = b.read_json(registry_path)
    for r in registry["reviewers"]:
        r.pop("review_policy")  # Persisted historical registration fixture, not a new-policy claim.
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    before = inventory(stage)
    with pytest.raises(ValueError, match="Fresh reviewer requires explicit"):
        b.load_reviews(rows, stage, [packet])
    with pytest.raises(ValueError, match="Fresh reviewer requires explicit"):
        s.refresh_progress(root)
    assert before == inventory(stage)


def test_two_policy_migrations_verify_both_archives_and_preserve_resolved_adjudication(sandbox):
    register_classification_pairs(sandbox)
    event = adjudication_event(sandbox)
    path = archive_adjudication(sandbox, event)
    s.submit_adjudication(event["reviewer_id"], "local-third-execution", path, b.digest(path), sandbox[0])
    root, stage, rows, packet = sandbox
    (root / "data/build_dataset_v3.py").write_bytes(Path(b.__file__).read_bytes())
    before = b.load_reviews(rows, stage, [packet])
    reason = "Explicit user-authorized source-tier policy amendment preserves every previous review and adjudication, gold and identity"
    first = s.archive_policy_inputs(reason, root)
    contract = b.OUT / "REVIEW_CONTRACT.md"
    contract.write_bytes(contract.read_bytes() + b"\nFirst local schema-only authorized amendment.\n")
    first_lineage = b.amend_review_policy_before_inference(reason, Path(first["archive"]))
    second = s.archive_policy_inputs(reason, root)
    contract.write_bytes(contract.read_bytes() + b"\nSecond local schema-only authorized amendment.\n")
    second_lineage = b.amend_review_policy_before_inference(reason, Path(second["archive"]))
    b.verify_preparation_lineage(stage)
    current = b.read_json(stage / "preparation_manifest.json")
    assert current["previous_preparation"] == second_lineage
    old = b.read_json(root / second["archive"] / "data/v3/staging/preparation_manifest.json")
    assert old["previous_preparation"] == first_lineage
    after = b.load_reviews(rows, stage, [packet])
    assert before[:2] == after[:2] and (stage / "candidate_pool.jsonl").read_bytes() == (
        root / first["archive"] / "data/v3/staging/candidate_pool.jsonl").read_bytes()
    damaged = root / first["archive"] / "data/v3/ELIGIBILITY_POLICY.md"
    damaged.chmod(0o644)
    damaged.write_bytes(damaged.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Invalid seal"):
        b.verify_preparation_lineage(stage)


def mock_scope_report(sandbox, reviewer="schema-reviewer-a", adjudication=False):
    root, stage, rows, _ = sandbox
    registry_path = stage / ("reviews/adjudications/adjudicators.json" if adjudication else "reviews/reviewers.json")
    registration = next(r for r in b.read_json(registry_path)["adjudicators" if adjudication else "reviewers"] if r["reviewer_id"] == reviewer)
    entry = registration["evidence"][0]
    ids = list(dict.fromkeys(e["id"] for e in b.read_jsonl(stage / "reviews" / entry["path"])))
    report = stage / "reviews/evidence/local-scope-report.json"
    b.write_json(report, {"reviewer_id": reviewer, "agent_run_id": registration["agent_run_id"],
        "strict_assigned_input_boundary_satisfied": False, "candidate_ids": ids,
        "agent_report": "Local unit mock actor self-report: I accessed material outside the assigned packet. Those results were not used, but the assigned input boundary was breached.",
        "original_evidence_path": entry["path"], "original_evidence_sha256": entry["sha256"]})
    packet = stage / "packets/batch_001.blind.jsonl"
    if adjudication:
        packet = stage / "reviews/adjudication_packets/local-scope-assignment.jsonl"
        b.write_jsonl(packet, [{"candidate": next(r for r in rows if r["id"] == pid)} for pid in ids])
        content = b.read_json(report)
        content.update(assigned_packet_path=packet.relative_to(stage / "reviews").as_posix(), assigned_packet_sha256=b.digest(packet))
        report.write_text(json.dumps(content), encoding="utf-8")
    return report, packet


@pytest.mark.parametrize("adjudication", [False, True])
def test_scope_breach_excludes_ordinary_or_adjudicated_acceptance_without_rewriting_verdicts(sandbox, adjudication):
    root, stage, rows, packet = sandbox
    reviewer = "schema-reviewer-a"
    if adjudication:
        register_classification_pairs(sandbox)
        event = adjudication_event(sandbox)
        evidence = archive_adjudication(sandbox, event)
        s.submit_adjudication(event["reviewer_id"], "local-third-execution", evidence, b.digest(evidence), root)
        reviewer = event["reviewer_id"]
    else:
        register_fresh_contrast_pairs(sandbox)
    before = {p: p.read_bytes() for p in (stage / "reviews").rglob("*") if p.is_file() and p.name != "review_progress.json"}
    report, assigned = mock_scope_report(sandbox, reviewer, adjudication)
    receipt = s.register_scope_incident(report, b.digest(report), assigned, root=root)
    approved, decisions, hashes = b.load_reviews(rows, stage, [packet])
    assert not approved and receipt["effective_approved_candidates"] == 0
    for pid in receipt["candidate_ids"]:
        decision = decisions[pid]
        assert decision["scope_disqualified"] and decision["pre_scope_accepted"] is True
        assert decision["exclusion_reasons"] == ["review_scope_breach"]
        assert decision["status"] == "scope_disqualified"
        assert all(r["original_verdicts"]["blind"]["correctness"] for r in decision["reviews"])
        if adjudication:
            assert decision["adjudication"]["record"]["approve"] is True
            assert decision["accepted_via_adjudication"] is True
    assert all(p.read_bytes() == content for p, content in before.items())
    assert b.relative(stage / "reviews" / b.SCOPE_LEDGER) in hashes
    snapshot = inventory(stage)
    repeat = s.register_scope_incident(report, b.digest(report), assigned, root=root)
    assert repeat["status"] == "already_registered" and snapshot == inventory(stage)
    progress = s.refresh_progress(root)
    assert progress["scope_disqualified_candidate_count"] == len(receipt["candidate_ids"])
    assert progress["historically_accepted_candidate_count"] == len(receipt["candidate_ids"])
    if adjudication:
        assert progress["resolved_adjudication_candidate_count"] == 1
        assert progress["accepted_via_adjudication_count"] == 1 and progress["effective_accepted_via_adjudication_count"] == 0
    with pytest.raises(ValueError, match="retry scope-disqualified"):
        paths = archive(sandbox, "schema-replacement-reviewer", reject=False)
        deliver(sandbox, "schema-replacement-reviewer", "local-new-attempt-execution", paths)


@pytest.mark.parametrize("defect", ["run", "marker", "flag", "partial_ids", "report_sha", "packet_sha"])
def test_scope_evidence_must_match_actual_execution_assignment_and_self_report(sandbox, defect):
    register_fresh_contrast_pairs(sandbox)
    report, assigned = mock_scope_report(sandbox)
    value = b.read_json(report)
    if defect == "run":
        value["agent_run_id"] = "wrong-local-execution"
    elif defect == "flag":
        value["strict_assigned_input_boundary_satisfied"] = True
    elif defect == "partial_ids":
        value["candidate_ids"].pop()
    elif defect == "marker":
        value["agent_report"] = "No actual scope report"
    report.write_text(json.dumps(value), encoding="utf-8")
    if defect == "packet_sha":
        assigned = sandbox[1] / "reviews/adjudication_packets/wrong-scope-assignment.jsonl"
        b.write_jsonl(assigned, [{"id": "schema-approve"}])
    before = inventory(sandbox[1])
    with pytest.raises(ValueError):
        s.register_scope_incident(report, "0" * 64 if defect == "report_sha" else b.digest(report), assigned, root=sandbox[0])
    assert before == inventory(sandbox[1])


def test_scope_technical_attestation_keeps_contract_and_allows_only_review_manager_records(sandbox, monkeypatch):
    register_fresh_contrast_pairs(sandbox)
    root, stage, rows, packet = sandbox
    original_builder = root / "data/build_dataset_v3.py"
    original_builder.write_bytes(Path(b.__file__).read_bytes())
    reason = "Explicit local technical scope-integrity repair preserves original registry, every mathematical verdict and source policy before inference"
    first = s.archive_policy_inputs(reason, root)
    replacement = root / "data/local-replacement-builder.txt"
    replacement.write_text("Local code-only attestation fixture", encoding="utf-8")
    monkeypatch.setattr(b, "__file__", str(replacement))
    report, assigned = mock_scope_report(sandbox)
    s.register_scope_incident(report, b.digest(report), assigned, Path(first["archive"]), root)
    fresh = s.archive_policy_inputs(reason, root, Path(first["archive"]))
    b.write_json(root / "results/v3/review_manager_state.json", {"freeze_not_attempted": True,
        "operator_a100_hours_spent_this_session": 0, "policy_id": b.REVIEW_POLICY_ID,
        "policy_sha256": b.digest(b.OUT / "REVIEW_CONTRACT.md")})
    (root / "results/v3/review_manager_state.md").write_text("Local coordinator notes, not inference", encoding="utf-8")
    with pytest.raises(ValueError, match="inference/results"):
        b.amend_review_policy_before_inference(reason, Path(fresh["archive"]))
    old_contract = b.digest(b.OUT / "REVIEW_CONTRACT.md")
    b.amend_review_policy_before_inference(reason, Path(fresh["archive"]), technical_scope_repair=True)
    assert b.digest(b.OUT / "REVIEW_CONTRACT.md") == old_contract
    b.verify_preparation_lineage(stage)
    assert not b.load_reviews(rows, stage, [packet])[0]
    ledger = stage / "reviews" / b.SCOPE_LEDGER
    original = ledger.read_bytes()
    ledger.unlink()
    with pytest.raises(ValueError, match="Missing anchored"):
        b.load_reviews(rows, stage, [packet])
    ledger.write_bytes(original + b"\n")  # Append-only bytes preserve the archived prefix.
    b.verify_preparation_lineage(stage)
    ledger.write_bytes(original.replace(b"review_scope_breach", b"removed_scope_breach"))
    with pytest.raises(ValueError, match="append-only"):
        b.verify_preparation_lineage(stage)


@pytest.mark.parametrize("tamper", ["empty", "truncated", "delete"])
def test_shared_scope_loader_cannot_restore_acceptance_by_removing_committed_incident(sandbox, tamper):
    register_fresh_contrast_pairs(sandbox)
    report, assigned = mock_scope_report(sandbox)
    s.register_scope_incident(report, b.digest(report), assigned, root=sandbox[0])
    ledger = sandbox[1] / "reviews" / b.SCOPE_LEDGER
    if tamper == "delete":
        ledger.unlink()
    else:
        ledger.write_bytes(b"" if tamper == "empty" else ledger.read_bytes()[:10])
    with pytest.raises(ValueError, match="Committed scope incident"):
        b.load_reviews(sandbox[2], sandbox[1], [sandbox[3]])


def test_scope_registration_is_serialized_without_deleting_another_callers_lock(sandbox):
    register_fresh_contrast_pairs(sandbox)
    report, assigned = mock_scope_report(sandbox)
    lock = sandbox[1] / "reviews/review_scope_incidents.lock"
    lock.write_text("Another incident-registration transaction", encoding="utf-8")
    before = inventory(sandbox[1])
    with pytest.raises(FileExistsError):
        s.register_scope_incident(report, b.digest(report), assigned, root=sandbox[0])
    assert before == inventory(sandbox[1]) and lock.exists()
