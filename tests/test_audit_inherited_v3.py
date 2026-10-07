"""Synthetic CPU-only inherited provenance audits; no real data or gold."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_verify_v3 import evidence, options, seal_public_fixture, snapshot, verify, write_archive

import audit_inherited_v3 as audit


@pytest.fixture
def inherited(evidence):
    root, inputs, *_ = evidence
    path = root / "dataset_manifest.json"
    metadata = json.loads(path.read_bytes())
    metadata["source_revisions"]["aime"] = dict(audit.INHERITED_AIME)
    path.write_text(json.dumps(metadata))
    seal_public_fixture(root, (inputs, path, root / "REVIEW_CONTRACT.md"))
    return evidence


def public_args(evidence):
    root = evidence[0]
    metadata = json.loads((root / "dataset_manifest.json").read_bytes())
    seals = {name: digest for digest, name in
             (line.split() for line in (root / "SHA256SUMS").read_text().splitlines())}
    return metadata, seals, root


def test_fresh_private_clone_and_receipt(inherited):
    original = verify.check_public_provenance
    first, second = audit.load_verifier(), audit.load_verifier()
    assert first is not second and first.__name__ != second.__name__
    assert first.__name__ not in sys.modules and second.__name__ not in sys.modules
    assert first.verify_run.__globals__ is first.__dict__
    assert first.verify_records.__globals__["check_public_provenance"] is first.check_public_provenance
    assert verify.check_public_provenance is original
    assert first.AUDITOR_ID == audit.AUDITOR_ID
    assert first.AUDITOR_VERSION == "1.0.0"
    assert first.AUDITOR_SHA256 == hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    assert first.VERIFIER_SHA256 == hashlib.sha256(Path(verify.__file__).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="pinned source provenance"):
        original(*public_args(inherited))
    assert first.check_public_provenance(*public_args(inherited))["source_revisions"]["aime"] == audit.INHERITED_AIME
    first.check_public_provenance = None
    assert second.check_public_provenance(*public_args(inherited))
    assert verify.check_public_provenance is original


def test_exact_public_mapping_preserved_without_mutation(inherited):
    args = public_args(inherited)
    before = copy.deepcopy(args[:2])
    result = audit.load_verifier().check_public_provenance(*args)
    assert args[:2] == before
    assert result["source_revisions"] is args[0]["source_revisions"]
    assert "repo" not in result["source_revisions"]["aime"]
    assert result["source_revisions"]["fixture"]["repo"] == "synthetic"


def test_repo_backed_case_matches_original(evidence):
    args = public_args(evidence)
    args[0]["source_revisions"]["aime"] = {"repo": "synthetic-aime", "revision": "b" * 40,
                                             "license": "synthetic"}
    assert audit.load_verifier().check_public_provenance(*args) == verify.check_public_provenance(*args)


@pytest.mark.parametrize("fault", ["family", "revision", "license", "extra", "repo_none", "repo_empty",
                                      "only_aime", "other_missing_repo", "other_extra", "private_metadata",
                                      "nonstring_revision", "nonstring_license", "other_nonstring_license",
                                      "policy_hash", "review_hash"])
def test_provenance_fails_closed(inherited, fault):
    metadata, seals, root = public_args(inherited)
    sources = metadata["source_revisions"]
    if fault == "family":
        sources["AIME"] = sources.pop("aime")
    elif fault == "revision":
        sources["aime"]["revision"] += " "
    elif fault == "license":
        sources["aime"]["license"] = "newly licensed"
    elif fault == "extra":
        sources["aime"]["private"] = "not public"
    elif fault == "repo_none":
        sources["aime"]["repo"] = None
    elif fault == "repo_empty":
        sources["aime"]["repo"] = ""
    elif fault == "only_aime":
        sources.pop("fixture")
    elif fault == "other_missing_repo":
        sources["fixture"].pop("repo")
    elif fault == "other_extra":
        sources["fixture"]["private"] = "not public"
    elif fault == "private_metadata":
        metadata["private"] = "not public"
    elif fault == "nonstring_revision":
        sources["aime"]["revision"] = 1
    elif fault == "nonstring_license":
        sources["aime"]["license"] = [audit.INHERITED_AIME["license"]]
    elif fault == "other_nonstring_license":
        sources["fixture"]["license"] = None
    elif fault == "policy_hash":
        metadata["policy_artifact_sha256"]["review_policy.json"] = "0" * 64
    else:
        seals["review_manifest.json"] = "0" * 64
    with pytest.raises(ValueError):
        audit.load_verifier().check_public_provenance(metadata, seals, root)


def test_repo_backed_nonstring_license_rejected(evidence):
    metadata, seals, root = public_args(evidence)
    metadata["source_revisions"]["fixture"]["license"] = False
    with pytest.raises(ValueError, match="Nonstring"):
        audit.load_verifier().check_public_provenance(metadata, seals, root)


def test_verify_run_and_cli_keep_original_json_pattern(inherited):
    root, inputs, config, prompts, _ = inherited
    write_archive(root, snapshot(inherited))
    opts = options(inherited)
    opts.pop("condition")
    report = audit.load_verifier().verify_run(root, conditions=["JFINAL"], **opts)
    assert report["ok"], report
    assert report["runs"]["JFINAL"]["dataset_provenance"]["source_revisions"] == public_args(inherited)[0]["source_revisions"]
    assert "repo" not in report["runs"]["JFINAL"]["dataset_provenance"]["source_revisions"]["aime"]
    assert not verify.verify_run(root, conditions=["JFINAL"], **opts)["ok"]
    output = root / "audit.json"
    result = subprocess.run([sys.executable, audit.__file__, str(root), "--conditions", "JFINAL",
                             "--inputs", str(inputs), "--config", str(config), "--prompts", str(prompts),
                             "--expected-count", "50", "--split", "pilot", "--run-tag", "mock",
                             "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(output.read_text()) == report
    assert set(report) == {"ok", "runs", "note"}
    assert audit.INHERITED_AIME["revision"] in result.stdout
    assert '"aime": {\n' in result.stdout


@pytest.mark.parametrize("fault", ["policy_bytes", "manifest_bytes", "input_bytes", "config_hash",
                                      "source_code", "model", "timestamp", "physical_row", "review"])
def test_remaining_validators_still_apply(inherited, fault):
    data = snapshot(inherited)
    root = inherited[0]
    if fault == "policy_bytes":
        (root / "ELIGIBILITY_POLICY.md").write_text("Changed synthetic policy")
    elif fault == "manifest_bytes":
        path = root / "dataset_manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "input_bytes":
        inherited[1].write_text("Changed synthetic input")
    elif fault == "config_hash":
        data["manifest"]["config_hash"] = "0" * 64
    elif fault == "source_code":
        data["manifest"]["config"]["code_sha256"] = {}
    elif fault == "model":
        data["manifest"]["model_lock"]["J"]["revision"] = "0" * 40
    elif fault == "timestamp":
        data["predictions"][0]["t_end_utc"] = data["predictions"][0]["t_start_utc"]
    elif fault == "physical_row":
        data["predictions"].pop()
    else:
        metadata = json.loads((root / "dataset_manifest.json").read_bytes())
        metadata["review"]["human_reviewed"] = True
        (root / "dataset_manifest.json").write_text(json.dumps(metadata))
        seal_public_fixture(root, (inherited[1], root / "dataset_manifest.json", root / "REVIEW_CONTRACT.md"))
        data = snapshot(inherited)
    with pytest.raises(ValueError):
        audit.load_verifier().verify_records(data, **options(inherited))
