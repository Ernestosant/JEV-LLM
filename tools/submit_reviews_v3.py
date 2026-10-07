"""Bind archived agent judgments locally, without editing evidence or freezing data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submission_dataset_v3", ROOT / "data/build_dataset_v3.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

BLIND_VERDICTS = ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")
REFERENCE_VERDICTS = ("gold_matches", "solution_consistent", "approve")


def archive_policy_inputs(reason, root=ROOT, policy_inputs=None):
    """Snapshot the old executable/contracts BEFORE editing them; never amend here."""
    root = Path(root).resolve()
    stage = root / "data/v3/staging"
    b.ROOT = root
    b.verify_seal(stage)
    metadata = b.read_json(stage / "preparation_manifest.json")
    files = {p.relative_to(root).as_posix(): p for p in stage.rglob("*") if p.is_file()}
    inputs = root if policy_inputs is None else (root / policy_inputs).resolve()
    if policy_inputs is not None:
        if not inputs.is_relative_to((root / "data/v3/policy_amendments").resolve()):
            raise ValueError("Pinned policy inputs must be an existing sealed archive")
        b.verify_seal(inputs)
    files.update({name: inputs / name for name in ("data/build_dataset_v3.py", "data/v3/REVIEW_CONTRACT.md", "data/v3/ELIGIBILITY_POLICY.md")})
    if (not reason.strip() or metadata["builder_sha256"] != b.digest(files["data/build_dataset_v3.py"])
            or metadata["review"]["policy_sha256"] != b.digest(files["data/v3/REVIEW_CONTRACT.md"])):
        raise ValueError("Archive must preserve the original pinned builder and review contract before editing")
    if any(p.is_symlink() or any(parent.is_symlink() for parent in p.parents) for p in files.values()):
        raise ValueError("Cannot archive symlinked policy inputs")
    inventory = {name: b.digest(path) for name, path in sorted(files.items())}
    fingerprint = b.digest_bytes(b.canonical(inventory).encode("utf-8"))
    destination = root / "data/v3/policy_amendments" / fingerprint
    if destination.exists():
        raise ValueError("Original preparation snapshot already exists; refusing replacement")
    for name, path in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    b.write_json(destination / "snapshot.json", {"reason": reason, "sha256": inventory})
    b.seal(destination, [p for p in destination.rglob("*") if p.is_file()])
    if any(b.digest(files[name]) != sha for name, sha in inventory.items()):
        raise ValueError("Staging changed during snapshot; preserve a new complete snapshot before repair")
    return {"archive": destination.relative_to(root).as_posix(), "snapshot_sha256": b.digest(destination / "snapshot.json"),
            "previous_seal_sha256": b.digest(stage / "SHA256SUMS"), "archived_files": len(files)}


def write_generated(path, value):
    """Replace generated JSON atomically; never use this for raw evidence."""
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        try:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            pending.unlink()
            raise
    try:
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def submit(packet, reviewer, agent_run_id, blind, reference, root=ROOT):
    root = Path(root).resolve()
    b.ROOT = root  # The builder remains read-only; this also supports isolated unit fixtures.
    b.OUT = root / "data/v3"
    stage = root / "data/v3/staging"
    directory = stage / "reviews"
    if not re.fullmatch(r"batch_[0-9]{3,}", packet):
        raise ValueError("Unknown packet ID")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,100}", reviewer):
        raise ValueError("Invalid reviewer ID")
    if not isinstance(agent_run_id, str) or len(agent_run_id.strip()) < 6 or agent_run_id != agent_run_id.strip():
        raise ValueError("An actual agent execution ID must be supplied by the caller")
    if (stage.parent / "SHA256SUMS").exists():
        raise ValueError("Refusing to change reviews of a frozen dataset")
    b.verify_seal(stage)
    metadata = b.read_json(stage / "preparation_manifest.json")
    b.verify_preparation_lineage(stage, metadata)
    packets = metadata["packets"]
    matches = [entry for entry in packets if entry["batch"] == packet]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate packet ID")
    entry = matches[0]
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    candidates = {row["id"]: row for row in rows}
    ids = entry["candidate_ids"]
    if len(candidates) != len(rows) or len(set(ids)) != len(ids) or not set(ids) <= set(candidates):
        raise ValueError("Unknown or duplicate staged candidate IDs")
    for phase in ("blind", "reference"):
        path = root / entry[phase]
        if not path.resolve().is_relative_to(stage.resolve()) or path.is_symlink() or b.digest(path) != entry[phase + "_sha256"]:
            raise ValueError("Staged packet hash/path changed")
        expected = ([{key: candidates[pid][key] for key in ("id", "problem", "language")} for pid in ids]
                    if phase == "blind" else [candidates[pid] for pid in ids])
        if b.read_jsonl(path) != expected:
            raise ValueError("Packet content differs from staged candidates")
    if any(candidates[pid]["candidate_sha256"] != b.candidate_sha(candidates[pid]) for pid in ids):
        raise ValueError("Staged candidate SHA changed")

    evidence, emitted = [], {}
    for phase, source in (("blind", blind), ("reference", reference)):
        path = Path(source).absolute()
        if (path.is_symlink() or path.resolve().parent != (directory / "evidence").resolve()
                or any(parent.is_symlink() for parent in path.parents) or path.suffix != ".jsonl"):
            raise ValueError("Raw evidence must already be archived at reviews/evidence/*.jsonl")
        evidence.append({"path": path.resolve().relative_to(directory.resolve()).as_posix(), "sha256": b.digest(path)})
        text = path.read_text(encoding="utf-8")
        if any(marker in text.casefold() for marker in ("synthetic fixture", "dummy review", "fabricated review")):
            raise ValueError("Fixture/dummy evidence cannot authorize real publication")
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        if any(not isinstance(event, dict) or event.get("reviewer_id") != reviewer
               or event.get("phase") != phase or event.get("id") not in ids for event in events):
            raise ValueError("Evidence contains unknown candidate/reviewer/phase")
        if len(events) != len(ids) or {event["id"] for event in events} != set(ids):
            raise ValueError("Every packet candidate requires exactly one emitted judgment per phase")
        emitted[phase] = events

    completed_blind = [{**event, "candidate_sha256": candidates[event["id"]]["candidate_sha256"],
                        "packet_sha256": entry["blind_sha256"]} for event in emitted["blind"]]
    blind_by_id = {event["id"]: event for event in completed_blind}
    raw_blind_by_id = {event["id"]: event for event in emitted["blind"]}
    for event in emitted["reference"]:
        supplied = event.get("blind_record_sha256")
        if supplied is not None and supplied not in (
                b.digest_bytes(b.canonical(raw_blind_by_id[event["id"]]).encode("utf-8")),
                b.digest_bytes(b.canonical(blind_by_id[event["id"]]).encode("utf-8"))):
            raise ValueError("Emitted reference hash does not bind its preserved blind solution")
    completed_reference = [{**event, "candidate_sha256": candidates[event["id"]]["candidate_sha256"],
                            "packet_sha256": entry["reference_sha256"],
                            "blind_record_sha256": b.digest_bytes(b.canonical(blind_by_id[event["id"]]).encode("utf-8"))}
                           for event in emitted["reference"]]
    registry_path = directory / "reviewers.json"
    registry = b.read_json(registry_path)
    reviewers = registry["reviewers"]
    existing = [item for item in reviewers if item.get("reviewer_id") == reviewer]
    binding = existing[0].get("review_policy") if existing else {
        "id": metadata["review"]["policy_id"], "sha256": metadata["review"]["policy_sha256"]}
    bindings, historical_registrations = b.review_policy_bindings(stage)
    if binding is None and (not existing or b.canonical(existing[0]) not in historical_registrations):
        raise ValueError("Fresh reviewer requires explicit policy binding; no unarchived historical exceptions")
    if binding is not None and (not isinstance(binding, dict) or set(binding) != {"id", "sha256"}
                               or (binding["id"], binding["sha256"]) not in bindings):
        raise ValueError("Review registration must bind the declared current or archived policy")
    historical = True if binding is None else bindings[(binding["id"], binding["sha256"])]
    # False remains a veto; a source-hint contrast is not a new-policy correctness failure.
    votes = [b.validate_review(candidates[event["id"]], blind_by_id[event["id"]], event,
                              source_hint_agreement_required=historical) for event in completed_reference]
    bound = completed_blind + completed_reference
    registration = {"reviewer_id": reviewer, "agent_run_id": agent_run_id, "kind": "agent",
                    "independent_blind_review": True, "evidence": evidence}
    if binding is not None:
        registration["review_policy"] = binding
    if existing and (len(existing) != 1 or existing[0] != registration):
        raise ValueError("Changed reviewer registry collision; refusing to replace execution/evidence")
    if any(item.get("agent_run_id") == agent_run_id and item.get("reviewer_id") != reviewer for item in reviewers):
        raise ValueError("Independent reviewers require distinct execution IDs")
    adjudicator_path = directory / "adjudications/adjudicators.json"
    if adjudicator_path.exists() and any(item.get("agent_run_id") == agent_run_id or item.get("reviewer_id") == reviewer
                                       for item in b.read_json(adjudicator_path)["adjudicators"]):
        raise ValueError("Independent reviewers and adjudicators require distinct execution IDs")
    output = directory / f"b{packet.removeprefix('batch_')}-{reviewer}.jsonl"
    if reviewers:
        _, decisions, _ = b.load_reviews(rows, stage, packets)
        if not existing and any(decisions.get(pid, {}).get("scope_disqualified") for pid in ids):
            raise ValueError("Cannot retry scope-disqualified candidates to reverse their integrity exclusion")
        if not existing and any(decisions.get(pid, {}).get("adjudication") for pid in ids):
            raise ValueError("Cannot append new blind pairs after their hash-bound adjudication")
    elif list(directory.glob("*.jsonl")):
        raise ValueError("Unregistered bound review records already exist")
    if output.exists() and (not existing or output.is_symlink() or b.read_jsonl(output) != bound):
        raise ValueError("Changed bound-record collision; refusing to overwrite judgments")
    if existing and not output.exists():
        raise ValueError("Registered submission is missing its bound-record file")
    for item in evidence:
        if b.digest(directory / item["path"]) != item["sha256"]:
            raise ValueError("Raw evidence changed during submission")
    idempotent = bool(existing)
    if not idempotent:
        b.write_jsonl(output, bound)
        write_generated(registry_path, {**registry, "reviewers": reviewers + [registration]})

    progress = refresh_progress(root)
    raw_hashes = {item["path"]: item["sha256"] for item in evidence}
    if any(b.digest(directory / name) != sha for name, sha in raw_hashes.items()):
        raise ValueError("Raw evidence changed during submission")
    return {"status": "already_submitted" if idempotent else "submitted", "idempotent": idempotent,
            "packet": packet, "reviewer_id": reviewer, "agent_run_id": agent_run_id, "records": len(bound),
            "approved_reviews": sum(votes), "rejected_reviews": len(votes) - sum(votes),
            "approved_candidates": progress["approved_candidate_count"], "rejected_candidates": progress["rejected_candidate_count"],
            "awaiting_adjudication_candidates": progress["awaiting_adjudication_candidate_count"],
            "missing_review_count": progress["missing_review_count"], "missing_candidate_count": progress["missing_candidate_count"],
            "bound_path": output.relative_to(root).as_posix(), "bound_sha256": b.digest(output),
            "raw_sha256": raw_hashes, "raw_unchanged": True, "publication_status": progress["status"]}


def refresh_progress(root=ROOT):
    root = Path(root).resolve()
    b.ROOT, b.OUT = root, root / "data/v3"
    stage = b.OUT / "staging"
    if (b.OUT / "SHA256SUMS").exists():
        raise ValueError("Refusing to change progress of a frozen dataset")
    metadata = b.read_json(stage / "preparation_manifest.json")
    b.verify_preparation_lineage(stage, metadata)
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    approved, decisions, hashes = b.load_reviews(rows, stage, metadata["packets"])
    counts = {r["id"]: len(decisions.get(r["id"], {}).get("reviews", [])) for r in rows}
    reviewers, vetoes = {}, []
    for pid, decision in sorted(decisions.items()):
        rejected = []
        for review in decision["reviews"]:
            count = reviewers.setdefault(review["reviewer_id"], {"completed_pairs": 0, "approved": 0, "rejected": 0})
            count["completed_pairs"] += 1
            count["approved" if review["approved"] else "rejected"] += 1
            if not review["approved"]:
                rejected.append({"reviewer_id": review["reviewer_id"], "reasons": review["reasons"]})
        if rejected:
            vetoes.append({"id": pid, "reviews": rejected})
    progress = {"schema_version": 4, "sealed": False, "required_independent_reviews": 2, "quota_axis": b.QUOTA_AXIS,
                "candidate_count": len(rows), "emitted_record_count": 2 * sum(counts.values()),
                "completed_review_count": sum(counts.values()), "reviewed_candidate_count": len(decisions),
                "independently_reviewed_candidate_count": sum(n >= 2 for n in counts.values()),
                "approved_candidate_count": len(approved),
                "scope_disqualified_candidate_count": sum(d.get("scope_disqualified", False) for d in decisions.values()),
                "historically_accepted_candidate_count": sum(d.get("pre_scope_accepted", d["accepted"]) for d in decisions.values()),
                "rejected_candidate_count": sum(d["status"] == "rejected" or (d["status"] == "resolved" and not d["accepted"])
                                                for d in decisions.values()),
                "awaiting_adjudication_candidate_count": sum(d["status"] == "awaiting_adjudication" for d in decisions.values()),
                "resolved_adjudication_candidate_count": sum(d.get("pre_scope_status", d["status"]) == "resolved" for d in decisions.values()),
                "accepted_via_adjudication_count": sum(d["accepted_via_adjudication"] for d in decisions.values()),
                "effective_accepted_via_adjudication_count": sum(d["accepted_via_adjudication"] and d["accepted"] for d in decisions.values()),
                "preserved_veto_candidate_count": len(vetoes),
                "missing_candidate_count": sum(n < 2 for n in counts.values()),
                "missing_review_count": sum(max(0, 2 - n) for n in counts.values()),
                "approved_review_count": sum(c["approved"] for c in reviewers.values()),
                "rejected_review_count": sum(c["rejected"] for c in reviewers.values()),
                "reviewers": reviewers, "approved_candidate_ids": sorted(approved),
                "rejected_candidates": vetoes, "candidate_decisions": decisions,
                "accepted_reviewed_strata": b.reviewed_capacity(rows, approved, decisions),
                "accepted_sampling_strata": b.reviewed_capacity(rows, approved, decisions, axis=b.QUOTA_AXIS),
                "intrinsic_difficulty_counts": {f: sum(decisions[pid]["final_difficulty"] == f for pid in approved) for f in b.TEST_QUOTA},
                "evidence_sha256": hashes}
    progress["sampling_strata_shortfalls"] = [{"domain": domain, "sampling_tier": tier,
        "required_test_plus_pilot": b.TEST_QUOTA[tier] + b.PILOT_QUOTA[tier],
        "accepted": count, "missing": max(0, b.TEST_QUOTA[tier] + b.PILOT_QUOTA[tier] - count)}
        for domain, tiers in progress["accepted_sampling_strata"].items() for tier, count in tiers.items()]
    try:
        b.select_approved(rows, approved, decisions)
        progress["status"] = "ready_for_freeze"
    except ValueError as error:
        progress.update(status="blocked_dataset_review", publication_blocker=str(error))
    path = stage / "reviews/review_progress.json"
    if not path.exists() or b.read_json(path) != progress:
        write_generated(path, progress)
    return progress


def prepare_adjudication_packet(output, candidate_ids=None, root=ROOT):
    root = Path(root).resolve()
    b.ROOT, b.OUT = root, root / "data/v3"
    stage = b.OUT / "staging"
    metadata = b.read_json(stage / "preparation_manifest.json")
    b.verify_preparation_lineage(stage, metadata)
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    _, decisions, _ = b.load_reviews(rows, stage, metadata["packets"])
    pending = {pid for pid, d in decisions.items() if d["status"] == "awaiting_adjudication"}
    ids = set(candidate_ids) if candidate_ids is not None else pending
    if not ids or not ids <= pending:
        raise ValueError("Adjudicator packets may contain only awaiting-adjudication candidates")
    pairs = {}
    for path in sorted((stage / "reviews").glob("*.jsonl")):
        if path.name == b.SCOPE_LEDGER:
            continue
        for event in b.read_jsonl(path):
            pairs.setdefault((event["id"], event["reviewer_id"]), {})[event["phase"]] = event
    packet = []
    for candidate in rows:
        pid = candidate["id"]
        if pid not in ids:
            continue
        reviews = decisions[pid]["reviews"]
        packet.append({"candidate": candidate, "original_review_refs": b.original_review_refs(reviews),
                       "original_reviews": [pairs[(pid, r["reviewer_id"])] for r in reviews],
                       "original_unit_concerns": [{"reviewer_id": r["reviewer_id"], "blind_record_sha256": r["blind_record_sha256"],
                            "unit_explicit": False, "original_unit_reason": r["original_unit_reason"]}
                            for r in reviews if r["original_verdicts"]["blind"]["unit_explicit"] is False]})
    output = Path(output).absolute()
    if not output.resolve().is_relative_to((stage / "reviews/adjudication_packets").resolve()) or output.is_symlink():
        raise ValueError("Generated adjudicator packets belong under reviews/adjudication_packets/")
    if output.exists():
        if b.read_jsonl(output) != packet:
            raise ValueError("Refusing to replace an existing adjudicator packet")
    else:
        b.write_jsonl(output, packet)
    return {"path": output.relative_to(root).as_posix(), "sha256": b.digest(output), "candidate_count": len(packet),
            "candidate_ids": [r["candidate"]["id"] for r in packet], "quota_visibility": False}


def submit_adjudication(reviewer, agent_run_id, evidence, evidence_sha256, root=ROOT):
    root = Path(root).resolve()
    b.ROOT, b.OUT = root, root / "data/v3"
    stage, path = b.OUT / "staging", Path(evidence).absolute()
    directory = stage / "reviews"
    if (b.OUT / "SHA256SUMS").exists():
        raise ValueError("Refusing adjudication after dataset freeze")
    if (not re.fullmatch(r"[a-zA-Z0-9_-]{3,100}", reviewer) or not isinstance(agent_run_id, str)
            or len(agent_run_id.strip()) < 6 or agent_run_id != agent_run_id.strip()):
        raise ValueError("Actual adjudicator identity and execution ID required")
    if (path.is_symlink() or path.resolve().parent != (directory / "evidence").resolve()
            or any(p.is_symlink() for p in path.parents) or path.suffix != ".jsonl" or b.digest(path) != evidence_sha256):
        raise ValueError("Supplied adjudicator emitted JSONL must be archived with matching SHA256")
    if any(marker in path.read_text(encoding="utf-8").casefold() for marker in ("synthetic fixture", "dummy review", "fabricated review")):
        raise ValueError("Fixture/dummy evidence cannot authorize actual adjudication")
    metadata = b.read_json(stage / "preparation_manifest.json")
    b.verify_preparation_lineage(stage, metadata)
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    _, current, _ = b.load_reviews(rows, stage, metadata["packets"])
    _, decisions, _ = b.load_reviews(rows, stage, metadata["packets"], include_adjudications=False)
    candidates = {r["id"]: r for r in rows}
    events = b.read_jsonl(path)
    if (not events or any(not isinstance(e, dict) or e.get("id") not in candidates or e.get("reviewer_id") != reviewer
                          or e.get("phase") != "adjudication" for e in events)
            or len({e["id"] for e in events}) != len(events)):
        raise ValueError("Unknown/duplicate emitted adjudication candidate/reviewer/phase")
    bound = [{**e, "candidate_sha256": candidates[e["id"]]["candidate_sha256"]} for e in events]
    accepts = [b.validate_adjudication(candidates[e["id"]], decisions[e["id"]], e) for e in bound]
    registry_path = directory / "adjudications/adjudicators.json"
    registry = b.read_json(registry_path) if registry_path.exists() else {"adjudicators": []}
    registration = {"reviewer_id": reviewer, "agent_run_id": agent_run_id, "kind": "agent", "independent_adjudication": True,
                    "evidence": [{"path": path.relative_to(directory).as_posix(), "sha256": evidence_sha256}]}
    existing = [r for r in registry["adjudicators"] if r["reviewer_id"] == reviewer]
    if not existing and any(current[e["id"]].get("scope_disqualified") for e in events):
        raise ValueError("Cannot retry scope-disqualified candidates with replacement adjudication")
    if existing and (len(existing) != 1 or existing[0] != registration):
        raise ValueError("Changed adjudicator registration; refusing replacement")
    others = b.read_json(directory / "reviewers.json")["reviewers"] + [r for r in registry["adjudicators"] if r["reviewer_id"] != reviewer]
    if any(r["agent_run_id"] == agent_run_id or r["reviewer_id"] == reviewer for r in others):
        raise ValueError("Adjudicator requires separate actual reviewer and distinct execution IDs")
    output = directory / "adjudications" / (reviewer + ".jsonl")
    if existing:
        if not output.exists() or output.is_symlink() or b.read_jsonl(output) != bound:
            raise ValueError("Registered adjudication output changed/missing")
    else:
        if output.exists() or any(current[e["id"]]["adjudication"] for e in events):
            raise ValueError("Adjudication already resolved; never replace or retry a verdict")
        if b.digest(path) != evidence_sha256:
            raise ValueError("Raw adjudication changed during submission")
        b.write_jsonl(output, bound)
        write_generated(registry_path, {"adjudicators": registry["adjudicators"] + [registration]})
    progress = refresh_progress(root)
    if b.digest(path) != evidence_sha256:
        raise ValueError("Raw adjudication changed during submission")
    return {"status": "already_submitted" if existing else "submitted", "idempotent": bool(existing),
            "records": len(bound), "accepted": sum(accepts), "rejected": len(accepts) - sum(accepts),
            "bound_path": output.relative_to(root).as_posix(), "bound_sha256": b.digest(output),
            "raw_sha256": evidence_sha256, "raw_unchanged": True,
            "approved_candidates": progress["approved_candidate_count"],
            "awaiting_adjudication_candidates": progress["awaiting_adjudication_candidate_count"]}


def register_scope_incident(report_path, report_sha256, assigned_packet, policy_inputs=None, root=ROOT):
    lock = Path(root) / "data/v3/staging/reviews/review_scope_incidents.lock"
    acquired = False
    try:
        with lock.open("x", encoding="utf-8"):
            acquired = True
            return _register_scope_incident(report_path, report_sha256, assigned_packet, policy_inputs, root)
    finally:
        if acquired:
            lock.unlink()


def _register_scope_incident(report_path, report_sha256, assigned_packet, policy_inputs=None, root=ROOT):
    root = Path(root).resolve()
    b.ROOT, b.OUT = root, root / "data/v3"
    stage, directory = b.OUT / "staging", b.OUT / "staging/reviews"
    if (b.OUT / "SHA256SUMS").exists():
        raise ValueError("Scope incidents must be adjudicated administratively before inference/freeze")
    inputs = (root / policy_inputs).resolve() if policy_inputs is not None else None
    if inputs is not None:
        if not inputs.is_relative_to((b.OUT / "policy_amendments").resolve()):
            raise ValueError("Technical repair requires a sealed pinned policy archive")
        b.verify_seal(inputs)
    metadata = b.read_json(stage / "preparation_manifest.json")
    b.verify_preparation_lineage(stage, metadata, policy_inputs=inputs)
    report_path, assigned_packet = Path(report_path).absolute(), Path(assigned_packet).absolute()
    if (report_path.is_symlink() or not report_path.resolve().is_relative_to((directory / "evidence").resolve())
            or b.digest(report_path) != report_sha256 or not assigned_packet.resolve().is_relative_to(stage.resolve())):
        raise ValueError("Scope report must be preserved evidence with supplied SHA and an exact staged assignment")
    report = b.read_json(report_path)
    core = {"schema_version": 1, "rule_id": b.SCOPE_RULE_ID, "reason": "review_scope_breach",
            "reviewer_id": report["reviewer_id"], "agent_run_id": report["agent_run_id"], "candidate_ids": report["candidate_ids"],
            "evidence": {"path": report_path.relative_to(directory).as_posix(), "sha256": report_sha256,
                         "self_report_marker": report["agent_report"]},
            "assigned_packet": {"path": os.path.relpath(assigned_packet, directory).replace("\\", "/"),
                                "sha256": b.digest(assigned_packet)}}
    incident = {**core, "incident_id": "jev-v3-scope-" + b.digest_bytes(b.canonical(core).encode("utf-8"))[:16]}
    ledger = directory / b.SCOPE_LEDGER
    previous = b.read_jsonl(ledger) if ledger.exists() else []
    same = [e for e in previous if e["incident_id"] == incident["incident_id"]]
    if same and (len(same) != 1 or same[0] != incident):
        raise ValueError("Changed scope incident collision; refusing to replace evidence")
    rows = b.read_jsonl(stage / "candidate_pool.jsonl")
    _, decisions, _ = b.load_reviews(rows, stage, metadata["packets"])
    b.apply_review_scope_incidents(stage, decisions, incidents=previous if same else previous + [incident])
    if b.digest(report_path) != report_sha256:
        raise ValueError("Actual scope report changed during registration")
    if not same:
        with ledger.open("ab") as handle:
            handle.write((b.canonical(incident) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    receipt_path = directory / "scope_incident_receipts" / (incident["incident_id"] + ".json")
    committed = ledger.read_bytes()
    receipt = {"incident": incident, "committed_prefix_size": len(committed),
               "committed_prefix_sha256": b.digest_bytes(committed)}
    if not receipt_path.exists():
        b.write_json(receipt_path, receipt)
        receipt_path.chmod(0o444)
    approved, _, _ = b.load_reviews(rows, stage, metadata["packets"])
    if inputs is None:
        refresh_progress(root)
    return {"status": "already_registered" if same else "registered", "incident_id": incident["incident_id"],
            "reviewer_id": incident["reviewer_id"], "agent_run_id": incident["agent_run_id"], "candidate_ids": incident["candidate_ids"],
            "ledger_path": ledger.relative_to(root).as_posix(), "ledger_sha256": b.digest(ledger),
            "actual_report_sha256": report_sha256, "raw_verdicts_unchanged": True, "effective_approved_candidates": len(approved),
            "technical_attestation_required": inputs is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet")
    parser.add_argument("--reviewer")
    parser.add_argument("--agent-run-id")
    parser.add_argument("--blind", type=Path)
    parser.add_argument("--reference", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--adjudication", type=Path)
    modes.add_argument("--refresh-progress", action="store_true")
    modes.add_argument("--prepare-adjudication-packet", type=Path)
    modes.add_argument("--archive-policy-inputs", metavar="REASON")
    modes.add_argument("--register-scope-incident", type=Path)
    parser.add_argument("--scope-report-sha256")
    parser.add_argument("--assigned-packet", type=Path)
    parser.add_argument("--pinned-policy-inputs", type=Path,
                        help="Sealed original inputs for a fresh concurrent-review snapshot or administrative technical repair")
    parser.add_argument("--adjudication-sha256")
    parser.add_argument("--candidate-ids", nargs="+")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        if args.archive_policy_inputs:
            result = archive_policy_inputs(args.archive_policy_inputs, args.root, args.pinned_policy_inputs)
        elif args.register_scope_incident:
            if not args.scope_report_sha256 or not args.assigned_packet:
                raise ValueError("Scope registration requires --scope-report-sha256 and --assigned-packet")
            result = register_scope_incident(args.register_scope_incident, args.scope_report_sha256, args.assigned_packet,
                                            args.pinned_policy_inputs, args.root)
        elif args.prepare_adjudication_packet:
            result = prepare_adjudication_packet(args.prepare_adjudication_packet, args.candidate_ids, args.root)
        elif args.refresh_progress:
            progress = refresh_progress(args.root)
            result = {k: v for k, v in progress.items() if k.endswith("count") or k in
                      ("status", "quota_axis", "accepted_reviewed_strata", "accepted_sampling_strata", "intrinsic_difficulty_counts", "sampling_strata_shortfalls")}
        elif args.adjudication:
            if not args.reviewer or not args.agent_run_id or not args.adjudication_sha256:
                raise ValueError("Adjudication requires --reviewer --agent-run-id --adjudication-sha256")
            result = submit_adjudication(args.reviewer, args.agent_run_id, args.adjudication, args.adjudication_sha256, args.root)
        else:
            if not all((args.packet, args.reviewer, args.agent_run_id, args.blind, args.reference)):
                raise ValueError("Fresh review submission requires --packet --reviewer --agent-run-id --blind --reference")
            result = submit(args.packet, args.reviewer, args.agent_run_id, args.blind, args.reference, args.root)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as error:
        print(json.dumps({"status": "blocked_review_submission", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
