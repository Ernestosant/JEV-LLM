"""CPU-only recovery of one completed, released JFINAL collection. No backend calls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as r

RUN_ID = "4a27fadffa554214b5ffdd57dd7b41ee"
SNAPSHOT_SHA = "ff1421d917315187e3d635759182a487d33c72f5865b3fb2ef590f0a8719f858"
SNAPSHOT_BYTES = 39603919
FINAL_MEMBER = "results/v3/JFINAL_confirmatory_v3_7daa1458_final.zip"
NOTEBOOK_MEMBER = "notebooks/03_JFINAL.out." + RUN_ID + ".ipynb"
PINS = {FINAL_MEMBER: (13698047, "b3666200ba53ba3482fe385c4a8922c849f610d33e7a1ce28405c1b0f6cde994"),
        NOTEBOOK_MEMBER: (494033, "cf93fc87ebaf324a8e15090333c2daefdca3ea5271f6e39e09247128bcbb38ee")}
VERIFIER_SHA = "e5d32d37239f7d224b13b409a9a15e1b9706035df2de65be700dd641b7dc50a3"
AUDITOR_FIELDS = ("AUDITOR_ID", "AUDITOR_VERSION", "AUDITOR_SHA256", "VERIFIER_SHA256",
                  "INHERITED_AUDITOR_SHA256")
CHANGED_FIELDS = {"status", "verified", "updated_utc", "technical_collection_recovery"}
HISTORY = "collection_recovery_history"


def safe_path(path):
    path = Path(path)
    r.require(not any(p.is_symlink() for p in (path, *path.parents)), "Recovery symlink forbidden: " + str(path))
    return path


def inventory(archive):
    infos = archive.infolist()
    names = [i.filename for i in infos]
    r.require(len(names) == len(set(names)), "Duplicate recovery ZIP path")
    r.require(sum(i.file_size for i in infos) <= r.MAX_COLLECTION_BYTES, "Recovery ZIP expansion limit exceeded")
    for i in infos:
        name = i.filename
        r.require(i.orig_filename == name and not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16)
                  and not PurePosixPath(name).is_absolute() and "\\" not in name and ":" not in name
                  and all(p not in {"", ".", ".."} for p in name.split("/")), "Unsafe recovery ZIP path")
    return sum(i.file_size for i in infos)


def member_sha(archive, name):
    # Read every byte for CRC and SHA, but never interpret a checkpoint as a ZIP.
    h = hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_snapshot(path, marker):
    safe_path(path)
    r.require(path.stat().st_size == SNAPSHOT_BYTES and r.digest(path) == SNAPSHOT_SHA,
              "Immutable pending snapshot SHA/size mismatch")
    with zipfile.ZipFile(path) as outer:
        expanded = inventory(outer)
        manifest = json.loads(outer.read("COLLECTION_MANIFEST.json"))
        files = manifest["files"]
        r.require(manifest.get("marker") == marker
                  and set(outer.namelist()) == set(files) | {"COLLECTION_MANIFEST.json"},
                  "Recovery marker/inventory mismatch")
        for name, sha in files.items():
            r.require(name.startswith(("results/v3/", "notebooks/")) or name in
                      {"papermill_v3.log", "driver_v3.log", "job_status_v3.json", "worker_v3.json"},
                      "Unexpected recovery member")
            r.require(member_sha(outer, name) == sha, "Recovery member SHA mismatch: " + name)
        execution = json.loads(outer.read("job_status_v3.json"))
        r.require(execution.get("run_id") == RUN_ID and execution.get("done") is True
                  and type(execution.get("rc")) is int and execution["rc"] == 0,
                  "Snapshot execution not completed rc0")
        selected = {}
        for name, (size, sha) in PINS.items():
            content = outer.read(name)
            r.require(len(content) == size and hashlib.sha256(content).hexdigest() == sha,
                      "Selected recovery artifact SHA/size mismatch")
            selected[name] = content
        with zipfile.ZipFile(io.BytesIO(selected[FINAL_MEMBER])) as final:
            inner_expanded = inventory(final)
            for name in final.namelist():
                outer_name = "results/" + name
                r.require(outer_name in files and member_sha(final, name) == files[outer_name],
                          "Final member differs from loose outer snapshot: " + name)
        return selected, dict(outer_expanded_bytes=expanded, final_expanded_bytes=inner_expanded,
                              manifest_sha256=hashlib.sha256(outer.read("COLLECTION_MANIFEST.json")).hexdigest(),
                              member_sha256=files, marker=marker)


def execution_gate(root, directory, state):
    r.require(state.get("run_id") == RUN_ID and state.get("job") == r.job("JFINAL", "test")
              and state.get("root") == str(root) and state.get("output") == str(directory)
              and state.get("cli_config") == str(directory / "sessions.json") and state.get("series") == "v3",
              "Recovery requires exact original root/job/run identity")
    execution = state.get("health", {}).get("execution", {})
    r.require(all(state.get(k) is True for k in ("startup_ack", "remote_startup_ack", "completed_execution", "released"))
              and execution.get("run_id") == RUN_ID and execution.get("done") is True
              and type(execution.get("rc")) is int and execution["rc"] == 0,
              "Recovery requires completed rc0 and released execution")
    ledger_path = safe_path(root / r.BUDGET_LEDGER)
    ledger = r.read_json(ledger_path)  # Never import/update the allocation ledger.
    entry = ledger.get("jobs", {}).get(RUN_ID, {})
    r.require(ledger.get("experiment_id") == r.EXPERIMENT_ID and ledger.get("max_gpu_seconds") == r.MAX_GPU_SECONDS
              and entry.get("key") == "test/JFINAL" and entry.get("session") == state["job"]["session"]
              and entry.get("output") == str(directory) and entry.get("endpoint") == state.get("owned_endpoint")
              and entry.get("released") is True and entry.get("release_verified") is True
              and entry.get("release_evidence") == "backend_absent"
              and entry.get("actual_seconds") == state.get("observed_gpu_seconds_upper_bound"),
              "Original backend-absent ledger proof required")
    marker = state.get("remote_marker", {})
    r.require(marker == dict(series="v3", session=state["job"]["session"], endpoint=state["owned_endpoint"],
                            run_id=RUN_ID, output=str(directory), config_sha256=r.digest(root / r.CONFIG)),
              "Original remote marker identity changed")
    return r.digest(ledger_path)


def audit(root, directory, state):
    approval = safe_path(Path(state["options"]["approval"]))
    pilot = safe_path(Path(state["options"]["pilot_report"]))
    r.require(approval.is_relative_to(root) and pilot.is_relative_to(root), "Audit consent paths escaped root")
    approved = r.read_json(approval)
    r.require(approved.get("new_verifier_sha256") == VERIFIER_SHA, "Recovery current checker pin mismatch")
    module = r.install_private_auditor(root, approved)
    r.require(module.VERIFIER_SHA256 == VERIFIER_SHA, "Loaded recovery checker pin mismatch")
    r.runtime_approval_gate(root, state["job"], approval, pilot)
    report = module.verify_run(directory / "artifacts", conditions=["JFINAL"],
                              inputs=root / "data/v3/test_inputs.jsonl", config=root / r.CONFIG,
                              prompts=root / "prompts", expected_count=500, split="test",
                              run_tag="confirmatory_v3", approval=approval, pilot_report=pilot)
    result = report.get("runs", {}).get("JFINAL", {})
    r.require(report.get("ok") is True and set(report.get("runs", {})) == {"JFINAL"}
              and result.get("ok") is True and result.get("records") == 1500
              and result.get("archive_sha256") == PINS[FINAL_MEMBER][1]
              and result.get("notebook_sha256") == PINS[NOTEBOOK_MEMBER][1],
              "Fresh full V2 audit failed: " + str(result.get("error", "coverage/artifact mismatch")))
    return report, {k: getattr(module, k) for k in AUDITOR_FIELDS}


def immutable_bytes(path, content):
    safe_path(path)
    if path.exists():
        r.require(path.is_file() and path.read_bytes() == content, "Original recovery evidence changed: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(content)
    path.chmod(0o444)


def validate_receipt(root, directory, state):
    """Validate commitments only; the caller must additionally run a fresh V2 audit."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    safe_path(directory)
    execution_gate(root, directory, state)
    history = directory / HISTORY
    receipt = r.read_json(safe_path(history / "receipt.json"))
    r.require(receipt == state.get("technical_collection_recovery") and receipt.get("run_id") == RUN_ID
              and receipt.get("root") == str(root) and receipt.get("remote_snapshot_sha256") is None
              and receipt.get("remote_snapshot_hash_persisted") is False,
              "Collection recovery receipt identity/disclosure changed")
    original_path = safe_path(history / "original_status.json")
    original = r.read_json(original_path)
    r.require(r.digest(original_path) == receipt["original_status_sha256"]
              and original.get("status") == "failed" and original.get("verified") is False
              and {k: v for k, v in original.items() if k not in CHANGED_FIELDS} ==
                  {k: v for k, v in state.items() if k not in CHANGED_FIELDS},
              "Original execution/source/deadline/charges changed")
    for name, sha in receipt["evidence_sha256"].items():
        r.require("/" not in name and "\\" not in name and name not in {".", ".."}
                  and r.digest(safe_path(history / name)) == sha, "Recovery evidence SHA changed")
    r.require(set(receipt["evidence_sha256"]) in ({"original_verification.json"},
              {"original_verification_absent.json"}), "Original verification presence evidence required")
    if "original_verification_absent.json" in receipt["evidence_sha256"]:
        r.require(r.read_json(history / "original_verification_absent.json") == {"verification_file_present": False},
                  "Original absence evidence changed")
    r.require(receipt["local_snapshot_sha256"] == SNAPSHOT_SHA
              and r.digest(safe_path(history / "original_collection.pending.zip")) == SNAPSHOT_SHA,
              "Original recovery snapshot changed")
    _, commitments = validate_snapshot(history / "original_collection.pending.zip", original["remote_marker"])
    r.require(commitments == receipt.get("commitments"), "Recovery manifest commitments changed")
    r.require(receipt.get("auditor") == dict(AUDITOR_ID="jev-v3-native-container-provenance-audit",
              AUDITOR_VERSION="2.0.0", AUDITOR_SHA256=r.PRIVATE_AUDITOR_SHA256,
              VERIFIER_SHA256=VERIFIER_SHA, INHERITED_AUDITOR_SHA256=r.INHERITED_AUDITOR_SHA256),
              "Recovery auditor receipt pins changed")
    artifacts = safe_path(directory / "artifacts")
    r.require({p.name for p in artifacts.iterdir()} == {Path(n).name for n in PINS}, "Recovery requires exactly two artifacts")
    for name, (size, sha) in PINS.items():
        path = safe_path(artifacts / Path(name).name)
        r.require(path.stat().st_size == size and r.digest(path) == sha, "Recovered artifact changed")
    r.require(r.digest(safe_path(directory / "verification.json")) == receipt["verification_sha256"],
              "Recovery verification changed")
    return receipt


def recover(root):
    root = Path(root).resolve()
    directory = safe_path(root / "results/v3/test/JFINAL")
    status_path = safe_path(directory / "status.json")
    status_raw = status_path.read_bytes()
    state = json.loads(status_raw)
    ledger_sha = execution_gate(root, directory, state)
    r.require(state.get("status") == "failed" and state.get("verified") is False
              and not state.get("technical_collection_recovery"), "Only unrecovered failed collection is eligible")
    selected, commitments = validate_snapshot(directory / "collection.pending.zip", state["remote_marker"])
    history = safe_path(directory / HISTORY)
    history.mkdir(exist_ok=True)
    # Preserve the original failure and absence of verification before any derived writes.
    immutable_bytes(history / "original_status.json", status_raw)
    evidence = {}
    verification = safe_path(directory / "verification.json")
    safe_path(verification.with_suffix(".json.tmp"))
    safe_path(status_path.with_suffix(".json.tmp"))
    if verification.exists():
        immutable_bytes(history / "original_verification.json", verification.read_bytes())
        evidence["original_verification.json"] = r.digest(history / "original_verification.json")
    else:
        immutable_bytes(history / "original_verification_absent.json", b'{"verification_file_present": false}\n')
        evidence["original_verification_absent.json"] = r.digest(history / "original_verification_absent.json")
    snapshot_copy = safe_path(history / "original_collection.pending.zip")
    if not snapshot_copy.exists():
        with snapshot_copy.open("xb") as target, (directory / "collection.pending.zip").open("rb") as source:
            shutil.copyfileobj(source, target)
    r.require(r.digest(snapshot_copy) == SNAPSHOT_SHA, "Original snapshot copy SHA mismatch")
    snapshot_copy.chmod(0o444)
    artifacts = safe_path(directory / "artifacts")
    artifacts.mkdir(exist_ok=True)
    r.require({p.name for p in artifacts.iterdir()} <= {Path(n).name for n in PINS}, "Unexpected canonical artifact; no overwrite")
    for name, content in selected.items():
        immutable_bytes(artifacts / Path(name).name, content)
    report, metadata = audit(root, directory, state)
    r.require(status_path.read_bytes() == status_raw and r.digest(root / r.BUDGET_LEDGER) == ledger_sha,
              "Original status/ledger changed during audit")
    r.atomic_json(verification, report)
    receipt = dict(run_id=RUN_ID, root=str(root), original_status_sha256=hashlib.sha256(status_raw).hexdigest(),
                   local_snapshot_sha256=SNAPSHOT_SHA, local_snapshot_bytes=SNAPSHOT_BYTES,
                   remote_snapshot_sha256=None, remote_snapshot_hash_persisted=False,
                   transport_disclosure="Remote transport SHA existed only in stdout/memory; local SHA and manifest commitments only, not remote hash verification.",
                   evidence_sha256=evidence, commitments=commitments, auditor=metadata,
                   verification_sha256=r.digest(verification), original_ledger_sha256=ledger_sha, recovered_utc=r.utc())
    immutable_bytes(history / "receipt.json", (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    state.update(status="completed", verified=True, updated_utc=r.utc(), technical_collection_recovery=receipt)
    validate_receipt(root, directory, state)
    r.atomic_json(status_path, state)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Original recorded root identity (run under WSL)")
    parser.add_argument("--recover-existing-completed", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(recover(args.root), allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, r.CoordinatorError, zipfile.BadZipFile) as error:
        print(json.dumps(dict(ok=False, error=str(error), error_type=type(error).__name__)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
