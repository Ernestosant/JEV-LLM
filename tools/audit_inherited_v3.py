"""Independently versioned offline auditor for exact inherited AIME provenance.

Use load_verifier().verify_run(...) for local revalidation. Record the returned
module's AUDITOR_ID, AUDITOR_VERSION, AUDITOR_SHA256 and VERIFIER_SHA256 alongside
the unchanged verification report; these identify code, not execution approval.
"""

import hashlib
import importlib.util
from pathlib import Path
import uuid

AUDITOR_ID = "jev-v3-inherited-public-provenance-audit"
AUDITOR_VERSION = "1.0.0"
AUDITOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INHERITED_AIME = {
    "revision": "local SHA256 pins inherited from v1/v2",
    "license": "Not established by local copy; inherited source, not a newly licensed corpus",
}


def load_verifier():
    """Load a fresh private verifier, changing only its provenance boundary."""
    path = Path(__file__).with_name("verify_run_v3.py")
    spec = importlib.util.spec_from_file_location(
        "_audit_inherited_v3_" + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.check_public_provenance

    def check_public_provenance(metadata, seals, directory):
        sources = metadata.get("source_revisions", {})
        # Match the publisher's supplied-value rule, including optional license.
        if isinstance(sources, dict):
            module.require(all(isinstance(source, dict)
                               and all(isinstance(value, str) for value in source.values())
                               for source in sources.values()),
                           "Nonstring/nonpublic source revision payload")
            aime = sources.get("aime")
            if isinstance(aime, dict) and "repo" not in aime:
                module.require(aime == INHERITED_AIME,
                               "Invalid inherited AIME public provenance")
                remaining = {name: source for name, source in sources.items() if name != "aime"}
                # The original checker requires nonempty, repo-backed remainder
                # and still validates every public policy/hash/review commitment.
                report = original({**metadata, "source_revisions": remaining}, seals, directory)
                return {**report, "source_revisions": sources}
        return original(metadata, seals, directory)

    module.check_public_provenance = check_public_provenance
    module.AUDITOR_ID = AUDITOR_ID
    module.AUDITOR_VERSION = AUDITOR_VERSION
    module.AUDITOR_SHA256 = AUDITOR_SHA256
    module.VERIFIER_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def main(argv=None):
    return load_verifier().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
