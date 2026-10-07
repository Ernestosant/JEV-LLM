"""V2 offline auditor: inherited V1 provenance plus native container UUID binding.

Only an absent per-item native_execution_id is compatible. The original
preflight validates the container execution identity; indexed token/output
evidence remains mandatory. No evidence or verification report is rewritten.
Record AUDITOR_ID, AUDITOR_VERSION, AUDITOR_SHA256, INHERITED_AUDITOR_SHA256
and VERIFIER_SHA256 from load_verifier() as code bindings, not approval.
"""

import hashlib
import json
import math
from pathlib import Path
import re

import audit_inherited_v3 as inherited

AUDITOR_ID = "jev-v3-native-container-provenance-audit"
AUDITOR_VERSION = "2.0.0"
AUDITOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INHERITED_AUDITOR_SHA256 = inherited.AUDITOR_SHA256
VERIFIER_SHA256 = hashlib.sha256(
    Path(__file__).with_name("verify_run_v3.py").read_bytes()).hexdigest()


def load_verifier():
    """Extend a fresh V1 private clone, leaving every other checker unchanged."""
    module = inherited.load_verifier()
    require, sha256 = module.require, module.sha256

    def check_equivalence(result, native, native_inputs):
        rows = result.get("rows", [])
        require(result.get("ok") is True and result.get("blocking") is True and result.get("skipped") is not True
                and len(rows) == len(native["items"]) == len(native_inputs)
                and result.get("identity") == native.get("identity") and result.get("reference_id") == native.get("reference_id")
                and result.get("native_execution_id") == native.get("native_execution_id"), "Incomplete/unbound selector equivalence")
        for index, (row, reference, item) in enumerate(zip(rows, native["items"], native_inputs)):
            n = len(item.get("question", {}).get("criteria", []))
            vectors = [row.get("vllm_logits"), row.get("native_logits"), row.get("vllm_probs"), row.get("native_probs")]
            require(n >= 2 and all(isinstance(v, list) and len(v) == n
                    and all(type(x) in (int, float) and math.isfinite(x) for x in v) for v in vectors),
                    "Missing/invalid selector equivalence vectors")
            vl, nl, vp, np = vectors
            require(row.get("same_ids") is True and reference.get("ids_equal_prompt_text") is True
                    and isinstance(reference.get("ids_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", reference["ids_sha256"])
                    and type(reference.get("n_ids")) is int and reference["n_ids"] > 0
                    and row.get("n_ids") == reference["n_ids"]
                    and nl == reference.get("raw_logits") and np == reference.get("probabilities"),
                    "Native/equivalence token or output identity mismatch")
            ids = reference.get("ids")
            require(isinstance(ids, list) and len(ids) == reference["n_ids"] and all(type(x) is int and x >= 0 for x in ids)
                    and reference["ids_sha256"] == sha256(json.dumps(ids).encode())
                    and row.get("ids") == ids and row.get("ids_sha256") == reference["ids_sha256"]
                    and row.get("native_ids") == ids and row.get("native_ids_sha256") == reference["ids_sha256"]
                    and row.get("index") == reference.get("index") == index
                    and row.get("reference_id") == reference.get("reference_id") == native.get("reference_id")
                    # V2: omission only; a present item UUID must still match exactly.
                    and row.get("native_execution_id") == native.get("native_execution_id")
                    and ("native_execution_id" not in reference
                         or reference["native_execution_id"] == native.get("native_execution_id"))
                    and all(row.get(k) == result.get(k) for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles")),
                    "Missing actual native/equivalence token IDs")
            for logits, probs in ((vl, vp), (nl, np)):
                exponentials = [math.exp((x - max(logits)) / native["temperature"]) for x in logits]
                normalized = [x / sum(exponentials) for x in exponentials]
                require(all(0 <= p <= 1 and math.isclose(p, q, abs_tol=1e-5, rel_tol=1e-5)
                            for p, q in zip(probs, normalized)) and abs(sum(probs) - 1) <= 1e-5,
                        "Invalid calibrated selector probabilities")
            av, an = vp.index(max(vp)), np.index(max(np))
            sorted_logits = sorted(nl, reverse=True)
            margin = sorted_logits[0] - sorted_logits[1]
            ulp = 2.0 ** (math.floor(math.log2(max(abs(sorted_logits[0]), 1e-6))) - 7)
            tie = margin <= 2 * ulp and max(nl) - nl[av] <= 2 * ulp
            require(row.get("argmax_vllm") == av and row.get("argmax_native") == an
                    and (av == an or tie), "Selector equivalence argmax fails bf16 tolerance")

    module.check_equivalence = check_equivalence
    module.AUDITOR_ID = AUDITOR_ID
    module.AUDITOR_VERSION = AUDITOR_VERSION
    module.INHERITED_AUDITOR_SHA256 = module.AUDITOR_SHA256
    module.AUDITOR_SHA256 = AUDITOR_SHA256
    return module


def main(argv=None):
    return load_verifier().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
