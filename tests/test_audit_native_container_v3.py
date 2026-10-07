"""Synthetic CPU-only V2 checks; no real archives, data, gold, or execution."""

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import pytest

from test_audit_inherited_v3 import inherited, public_args
from test_verify_v3 import evidence, options, snapshot, verify, write_archive

import audit_inherited_v3 as v1
import audit_native_container_v3 as audit


@pytest.fixture
def equivalent():
    execution = "11111111-1111-4111-8111-111111111111"
    phase = {"phase_epoch": 1, "phase_id": "synthetic-phase", "gpu_uuid": "GPU-synthetic",
             "resident_roles": ["G", "J"]}
    logits = [2., 1., 0., -1.]
    exp = [math.exp((x - max(logits)) / 1.316) for x in logits]
    probs = [x / sum(exp) for x in exp]
    native = {"identity": {"synthetic": True}, "reference_id": "a" * 64,
              "native_execution_id": execution, "temperature": 1.316, "items": []}
    result = {"ok": True, "blocking": True, **phase, "identity": native["identity"],
              "reference_id": native["reference_id"], "native_execution_id": execution, "rows": []}
    for index in range(8):
        ids = [index, 10, 20]
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        reference = {"index": index, "reference_id": native["reference_id"], "ids": ids,
                     "ids_sha256": digest, "n_ids": len(ids), "ids_equal_prompt_text": True,
                     "raw_logits": logits, "probabilities": probs}
        native["items"].append(copy.deepcopy(reference))
        result["rows"].append(copy.deepcopy({**phase, **reference, "native_execution_id": execution,
            "same_ids": True, "native_ids": ids, "native_ids_sha256": digest,
            "vllm_logits": logits, "native_logits": logits, "vllm_probs": probs,
            "native_probs": probs, "argmax_vllm": 0, "argmax_native": 0}))
    inputs = [{"question": {"criteria": ["synthetic"] * 4}} for _ in range(8)]
    return result, native, inputs


@pytest.mark.parametrize("present", [False, True])
def test_absent_or_exact_item_uuid_only(equivalent, present):
    result, native, _ = equivalent
    if present:
        for item in native["items"]:
            item["native_execution_id"] = native["native_execution_id"]
    before = copy.deepcopy(equivalent)
    assert audit.load_verifier().check_equivalence(*equivalent) is None
    assert equivalent == before
    if present:
        verify.check_equivalence(*equivalent)
        v1.load_verifier().check_equivalence(*equivalent)
    else:
        for module in (verify, v1.load_verifier()):
            with pytest.raises(ValueError, match="actual native"):
                module.check_equivalence(*equivalent)


@pytest.mark.parametrize("value", [None, "wrong", "", 0, False])
def test_present_invalid_item_uuid_rejected(equivalent, value):
    equivalent[1]["items"][7]["native_execution_id"] = value
    with pytest.raises(ValueError, match="actual native"):
        audit.load_verifier().check_equivalence(*equivalent)


@pytest.mark.parametrize("target,key,value", [
    ("native", "native_execution_id", "wrong"), ("result", "native_execution_id", "wrong"),
    ("row", "native_execution_id", "wrong"), ("row", "native_execution_id", None),
    ("row", "ids", [99, 10, 20]), ("item", "ids", [99, 10, 20]),
    ("row", "native_ids", [99, 10, 20]), ("row", "ids_sha256", "0" * 64),
    ("row", "native_ids_sha256", "0" * 64), ("item", "ids_sha256", "0" * 64),
    ("row", "index", 0), ("item", "index", 0),
    ("row", "reference_id", "wrong"), ("item", "reference_id", "wrong"),
    ("native", "reference_id", "wrong"), ("result", "reference_id", "wrong"),
    ("result", "identity", {}), ("row", "phase_epoch", 99),
    ("row", "phase_id", "wrong"), ("row", "gpu_uuid", "wrong"),
    ("row", "resident_roles", ["B"]), ("row", "same_ids", False),
    ("item", "ids_equal_prompt_text", False), ("row", "n_ids", 99),
    ("item", "n_ids", 99), ("row", "native_logits", [0.] * 4),
    ("item", "raw_logits", [0.] * 4), ("row", "native_probs", [.25] * 4),
    ("item", "probabilities", [.25] * 4), ("row", "vllm_probs", [.25] * 4),
    ("row", "vllm_logits", [True] * 4), ("row", "vllm_logits", [float("inf")] * 4),
    ("row", "vllm_logits", [0.]), ("row", "argmax_vllm", 1),
    ("row", "argmax_native", 1), ("native", "temperature", 1.),
    ("result", "ok", False), ("result", "blocking", False), ("result", "skipped", True),
])
def test_remaining_predicates_reject(equivalent, target, key, value):
    result, native, _ = equivalent
    targets = {"result": result, "native": native, "row": result["rows"][7], "item": native["items"][7]}
    targets[target][key] = value
    with pytest.raises(ValueError):
        audit.load_verifier().check_equivalence(*equivalent)


def test_calibrated_argmax_outside_margin_rejected(equivalent):
    row = equivalent[0]["rows"][7]
    row["vllm_logits"] = [1., 2., 0., -1.]
    probs = row["vllm_probs"]
    row["vllm_probs"] = [probs[1], probs[0], *probs[2:]]
    row["argmax_vllm"] = 1
    with pytest.raises(ValueError, match="bf16 tolerance"):
        audit.load_verifier().check_equivalence(*equivalent)


def test_matched_but_uncalibrated_native_probabilities_rejected(equivalent):
    equivalent[0]["rows"][7]["native_probs"] = [.25] * 4
    equivalent[1]["items"][7]["probabilities"] = [.25] * 4
    with pytest.raises(ValueError, match="calibrated"):
        audit.load_verifier().check_equivalence(*equivalent)


def test_original_bf16_tie_tolerance_preserved(equivalent):
    result, native, _ = equivalent
    row, item = result["rows"][7], native["items"][7]
    row["native_logits"] = item["raw_logits"] = [2., 1.999, 0., -1.]
    row["vllm_logits"] = [1.999, 2., 0., -1.]
    for logits_key, probs_key in (("native_logits", "native_probs"), ("vllm_logits", "vllm_probs")):
        exp = [math.exp((x - 2.) / native["temperature"]) for x in row[logits_key]]
        row[probs_key] = [x / sum(exp) for x in exp]
    item["probabilities"] = row["native_probs"]
    row["argmax_vllm"] = 1
    audit.load_verifier().check_equivalence(*equivalent)
    for reference in native["items"]:
        reference["native_execution_id"] = native["native_execution_id"]
    verify.check_equivalence(*equivalent)


def test_missing_rows_or_inputs_rejected(equivalent):
    for position, key in ((0, "rows"), (1, "items"), (2, None)):
        altered = copy.deepcopy(equivalent)
        (altered[position][key] if key else altered[position]).pop()
        with pytest.raises(ValueError, match="Incomplete"):
            audit.load_verifier().check_equivalence(*altered)


def test_private_clones_code_bindings_and_originals_unchanged(equivalent):
    files = [Path(module.__file__) for module in (verify, v1)]
    before = [path.read_bytes() for path in files]
    original = verify.check_equivalence
    prior = v1.load_verifier()
    first, second = audit.load_verifier(), audit.load_verifier()
    assert first is not second and first.__name__ != second.__name__
    assert first.__name__ not in sys.modules and second.__name__ not in sys.modules
    for name, function in vars(prior).items():
        if isinstance(function, types.FunctionType) and name != "check_equivalence":
            assert getattr(first, name).__code__ == function.__code__, name
    assert first.verify_run.__globals__ is first.__dict__
    assert first.check_preflight.__globals__["check_equivalence"] is first.check_equivalence
    assert first.check_postflight.__globals__["check_equivalence"] is first.check_equivalence
    assert first.AUDITOR_ID == audit.AUDITOR_ID != v1.AUDITOR_ID
    assert first.AUDITOR_VERSION == "2.0.0"
    assert first.AUDITOR_SHA256 == hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    assert first.INHERITED_AUDITOR_SHA256 == audit.INHERITED_AUDITOR_SHA256 == v1.AUDITOR_SHA256
    assert first.VERIFIER_SHA256 == audit.VERIFIER_SHA256 == hashlib.sha256(before[0]).hexdigest()
    first.check_equivalence = None
    second.check_equivalence(*equivalent)
    assert verify.check_equivalence is original
    assert prior.AUDITOR_VERSION == v1.AUDITOR_VERSION == "1.0.0"
    with pytest.raises(ValueError):
        prior.check_equivalence(*equivalent)
    assert [path.read_bytes() for path in files] == before


def test_inherited_mapping_preserved(inherited):
    args = public_args(inherited)
    result = audit.load_verifier().check_public_provenance(*args)
    assert result == v1.load_verifier().check_public_provenance(*args)
    assert result["source_revisions"] is args[0]["source_revisions"]


def test_synthetic_cli_uses_original_report(inherited):
    data = snapshot(inherited)
    native = data["artifacts"]["preflight/phases/pre_engines/native_ref_out.json"]
    for item in native["items"]:
        del item["native_execution_id"]
    before = copy.deepcopy(data)
    module = audit.load_verifier()
    report = module.verify_records(data, **options(inherited))
    assert data == before
    assert report["dataset_provenance"]["source_revisions"] == public_args(inherited)[0]["source_revisions"]
    with pytest.raises(ValueError, match="actual native"):
        v1.load_verifier().verify_records(data, **options(inherited))
    root, inputs, config, prompts, _ = inherited
    write_archive(root, data)
    opts = options(inherited)
    opts.pop("condition")
    expected = module.verify_run(root, conditions=["JFINAL"], **opts)
    assert expected["ok"], expected
    output = root / "audit.json"
    process = subprocess.run([sys.executable, audit.__file__, str(root), "--conditions", "JFINAL",
        "--inputs", str(inputs), "--config", str(config), "--prompts", str(prompts),
        "--expected-count", "50", "--split", "pilot", "--run-tag", "mock", "--output", str(output)],
        capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == json.loads(output.read_text()) == expected
    assert set(expected) == {"ok", "runs", "note"}


@pytest.mark.parametrize("artifact,key", [
    ("native_ref_in.json", "native_execution_id"), ("native_ref_in.json", "identity"),
    ("native_ref_out.json", "native_execution_id"),
])
def test_original_preflight_container_bindings_still_reject(inherited, artifact, key):
    data = snapshot(inherited)
    artifacts = data["artifacts"]
    native = artifacts["preflight/phases/pre_engines/native_ref_out.json"]
    for item in native["items"]:
        del item["native_execution_id"]
    artifacts["preflight/phases/pre_engines/" + artifact][key] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.load_verifier().verify_records(data, **options(inherited))


@pytest.mark.parametrize("value", [None, "", "not-a-uuid"])
def test_bound_but_invalid_container_uuid_rejected(inherited, value):
    data = snapshot(inherited)
    artifacts = data["artifacts"]
    full = artifacts["preflight/summary.json"]
    native = artifacts["preflight/phases/pre_engines/native_ref_out.json"]
    native["native_execution_id"] = value
    artifacts["preflight/phases/pre_engines/native_ref_in.json"]["native_execution_id"] = value
    full["selector_native_reference"]["native_execution_id"] = value
    full["selector_equivalence"]["native_execution_id"] = value
    for item, row in zip(native["items"], full["selector_equivalence"]["rows"]):
        del item["native_execution_id"]
        row["native_execution_id"] = value
    with pytest.raises(ValueError):
        audit.load_verifier().verify_records(data, **options(inherited))
