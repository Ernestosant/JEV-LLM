"""Offline V3 audit: explicit, memory-only operational LATENCY proof projection.

The archive never claimed a global selector_equivalence check. V3 may project
the preserved unique H-phase proof into a private audit view, not the evidence.
Use load_verifier().verify_run(...) and retain its auditor/proof metadata.
"""

import copy
import hashlib
import json
from pathlib import Path

import audit_native_container_v3 as v2

AUDITOR_ID = "jev-v3-operational-latency-projection-audit"
AUDITOR_VERSION = "3.0.0"
AUDITOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INHERITED_AUDITOR_SHA256 = "66f965f110e15155b9eadf18dbfc6481534dbca264f161522763af92e38796e7"
V1_AUDITOR_SHA256 = "3a2299c8da375f215e07eb64da905163abd9f0bec47986c23da2d973e14d67a4"
VERIFIER_SHA256 = "e5d32d37239f7d224b13b409a9a15e1b9706035df2de65be700dd641b7dc50a3"
APPROVAL_SHA256 = "47a010594aaa1dfa0587c1033a7fc82e4476cc20dd1c6d39a6e131a5016bca43"
POLICY_ID = "jev-v3-balanced-residencies-25h-20261006"
REDUCED_FIELDS = ("ok", "blocking", "summary", "skipped", "phase_epoch", "phase_id",
                  "gpu_uuid", "resident_roles", "cleanup_confirmed")


def load_verifier():
    """Extend a fresh pinned V2 clone; leave all other validation unchanged."""
    for filename, expected in (("verify_run_v3.py", VERIFIER_SHA256),
                               ("audit_native_container_v3.py", INHERITED_AUDITOR_SHA256),
                               ("audit_inherited_v3.py", V1_AUDITOR_SHA256)):
        actual = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("V3 auditor SHA chain mismatch: " + filename)
    module = v2.load_verifier()
    module.require(module.AUDITOR_SHA256 == INHERITED_AUDITOR_SHA256
                   and module.INHERITED_AUDITOR_SHA256 == V1_AUDITOR_SHA256
                   and module.VERIFIER_SHA256 == VERIFIER_SHA256, "V3 loaded SHA chain mismatch")
    original_records, original_run = module.verify_records, module.verify_run
    require = module.require
    certificate = {
        "id": AUDITOR_ID, "version": AUDITOR_VERSION, "sha256": AUDITOR_SHA256,
        "v2_auditor_sha256": INHERITED_AUDITOR_SHA256,
        "v1_auditor_sha256": V1_AUDITOR_SHA256, "verifier_sha256": VERIFIER_SHA256,
    }

    def verify_records(data, **kwargs):
        full = data.get("artifacts", {}).get("preflight/summary.json", {})
        reduced = data.get("manifest", {}).get("preflight", {})
        name = "selector_equivalence"
        if kwargs.get("condition") != "LATENCY":
            return original_records(data, **kwargs)
        if name in full or name in reduced:
            # Present results, including failed/skipped/mismatched ones, are never replaced.
            info = original_records(data, **kwargs)
            return {**info, "auditor": copy.deepcopy(certificate),
                    "proof_projection": {"applied": False, "memory_only": True}}

        manifest = data["manifest"]
        approval = kwargs.get("approval")
        require(isinstance(approval, (str, Path)), "V3 projection requires original approval")
        raw = module.safe_read(approval)
        require(module.sha256(raw) == APPROVAL_SHA256, "V3 projection approval SHA mismatch")
        policy = json.loads(raw).get("operational_amendment", {})
        require(policy.get("id") == POLICY_ID
                and manifest.get("authorization", {}).get("approval_sha256") == APPROVAL_SHA256
                and manifest.get("authorization", {}).get("operational_amendment") == policy
                and manifest.get("condition") == "LATENCY" and kwargs.get("split") == "test",
                "V3 projection outside approval-bound operational LATENCY")
        require(manifest.get("preflight_full") == "preflight/summary.json"
                and reduced == {key: {field: result.get(field) for field in REDUCED_FIELDS}
                                for key, result in full.items()},
                "Original reduced/global full preflight mismatch before projection")

        phases = [s for s in data.get("swaps", []) if s.get("to") == "H"]
        require(len(phases) == 1, "V3 projection requires unique actual H phase")
        phase = phases[0]
        require(any(r.get("phase") == "H" and r.get("phase_id") == phase.get("phase_id")
                    and r.get("phase_epoch") == phase.get("phase_epoch")
                    for r in data.get("latency_metrics", [])), "V3 H phase has no actual cases")
        folder = module.phase_folder(phase)
        detail = data.get("artifacts", {}).get(folder + "/summary.json", {})
        proof = detail.get(name, {})
        require(len(proof.get("rows", [])) == 8, "V3 projection requires eight actual equivalence rows")

        cfg = manifest["config"]
        reference = full.get("selector_native_reference", {})
        native_folder = module.phase_folder(reference)
        native = data.get("artifacts", {}).get(native_folder + "/native_ref_out.json", {})
        native_inputs = data.get("artifacts", {}).get(native_folder + "/native_ref_in.json", {}).get("items", [])
        require(isinstance(native.get("items"), list) and isinstance(native_inputs, list),
                "V3 projection missing native reference")
        require(len(native["items"]) == len(native_inputs) == 8,
                "V3 projection requires eight native reference items")
        # Reject bad phase proof before copying the large case ledgers. The
        # unchanged full preflight guard below still authenticates native identity.
        module.check_phase_artifacts(data, detail, phase, native, native_inputs)
        module.check_postflight(detail, "H", cfg, native, native_inputs)
        module.check_swaps(data, cfg, data["latency_metrics"], manifest["gpu_uuid"], native, native_inputs)
        view = copy.deepcopy(data)
        view["artifacts"]["preflight/summary.json"][name] = copy.deepcopy(proof)
        view["manifest"]["preflight"][name] = {field: copy.deepcopy(proof.get(field))
                                                for field in REDUCED_FIELDS}
        info = original_records(view, **kwargs)
        return {**info, "auditor": copy.deepcopy(certificate), "proof_projection": {
            "applied": True, "memory_only": True, "artifact_global_equivalence_present": False,
            "check": name, "policy_id": POLICY_ID, "approval_sha256": APPROVAL_SHA256,
            "source": folder + "/selector_equivalence.json",
            "source_summary": folder + "/summary.json",
            "proof_sha256": module.canonical_hash(proof), "rows": 8,
            "phase_id": phase["phase_id"], "phase_epoch": phase["phase_epoch"],
            "gpu_uuid": phase["gpu_uuid"], "native_execution_id": native["native_execution_id"],
            "original_full_summary_sha256": module.canonical_hash(full),
            "original_reduced_preflight_sha256": module.canonical_hash(reduced),
            "targets": ["artifacts.preflight/summary.json.selector_equivalence",
                        "manifest.preflight.selector_equivalence"],
        }}

    def verify_run(*args, **kwargs):
        return {**original_run(*args, **kwargs), "auditor": copy.deepcopy(certificate)}

    module.verify_records = verify_records
    module.verify_run = verify_run
    module.AUDITOR_ID = AUDITOR_ID
    module.AUDITOR_VERSION = AUDITOR_VERSION
    module.AUDITOR_SHA256 = AUDITOR_SHA256
    module.INHERITED_AUDITOR_SHA256 = INHERITED_AUDITOR_SHA256
    module.V1_AUDITOR_SHA256 = V1_AUDITOR_SHA256
    return module


def main(argv=None):
    return load_verifier().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
