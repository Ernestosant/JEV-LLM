"""Approval-bound test/LAT continuation; historical recovery evidence is read-only."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as r
from resume_v3 import RecoveryCoordinator, PROVENANCE_ERROR

AUDITOR_SHA256 = "66f965f110e15155b9eadf18dbfc6481534dbca264f161522763af92e38796e7"
AUDITOR_FIELDS = ("AUDITOR_ID", "AUDITOR_VERSION", "AUDITOR_SHA256", "VERIFIER_SHA256",
                  "INHERITED_AUDITOR_SHA256")


class PostPilotCoordinator(r.Coordinator):
    def __init__(self, root=r.ROOT, phase="test", **options):
        r.require(phase in {"test", "latency"}, "Post-pilot phase must be test or latency",
                  "blocked_user_authorization")
        prefix = options.pop("attempt_prefix", None)
        r.require(prefix in {None, "infra03"}, "Only historical infra03 linkage is supported",
                  "blocked_ownership")
        r.require(options.get("authorize_initial") is True
                  and options.get("authorize_confirmatory") is True,
                  "Both original and confirmatory authorization are required", "blocked_user_authorization")
        super().__init__(root, phase, **options)
        self.path = self.out / ("coordinator_postpilot_" + phase + "_state.json")
        self.recovery_path = self.out / "coordinator_full_infra03_recovery_state.json"
        self.receipt_path = self.out / "recovery/infra03/recovery_receipt.json"
        self.assets_path = self.out / "operational_amendment/pre_amendment_assets.zip"

    def install_private_auditor(self, native_container=True):
        """Install V2 only in this process, bound to the approved *current* checker."""
        approved = r.read_json(self.options["approval"])
        r.private_auditor_files(self.root, approved)
        path = self.root / "tools/audit_native_container_v3.py"
        verifier = self.root / "tools/verify_run_v3.py"
        for file in (path, verifier, self.root / "tools/audit_inherited_v3.py"):
            r.require(not any(p.is_symlink() for p in (file, *file.parents)), "Auditor symlink forbidden")
        r.require(r.digest(path) == AUDITOR_SHA256
                   and r.digest(verifier) == approved.get("new_verifier_sha256"),
                  "Approved current verifier/V2 auditor SHA mismatch", "blocked_budget_approval")
        spec = importlib.util.spec_from_file_location("_postpilot_auditor", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        module = helper.load_verifier()
        r.require(module.AUDITOR_SHA256 == AUDITOR_SHA256
                  and module.VERIFIER_SHA256 == r.digest(verifier)
                  and module.INHERITED_AUDITOR_SHA256 == r.digest(self.root / "tools/audit_inherited_v3.py"),
                  "Loaded auditor identity mismatch")
        metadata = {k: getattr(module, k) for k in AUDITOR_FIELDS}
        r.require(self.state.get("private_auditor", metadata) == metadata, "Post-pilot auditor changed")
        self.state["private_auditor"] = metadata
        sys.modules["verify_run_v3"] = module
        if "pilot_decision_v3" in sys.modules:
            importlib.reload(sys.modules["pilot_decision_v3"])
        return module

    def startup_gate(self):
        self.validate_history()
        self.install_private_auditor()
        r.runtime_approval_gate(self.root, self.jobs[0], self.options["approval"], self.options["pilot_report"])

    def validate_history(self):
        """Check immutable original proofs, not a regenerated RecoveryReceipt."""
        approved = r.read_json(self.options["approval"])
        policy = approved.get("operational_amendment")
        r.require(isinstance(policy, dict) and approved.get("max_gpu_hours") == 25
                  and approved.get("no_additional_allowance_after_pilot") is True,
                  "Original 25-hour ceiling/no additional allowance required", "blocked_budget_approval")
        pins = {self.assets_path: "preserved_assets_sha256", self.recovery_path: "recovery_state_sha256",
                self.receipt_path: "historical_recovery_receipt_sha256"}
        for path, field in pins.items():
            r.require(not any(p.is_symlink() for p in (path, *path.parents))
                      and r.re.fullmatch(r"[0-9a-f]{64}", approved.get(field, ""))
                      and r.digest(path) == approved[field], "Historical amendment SHA mismatch: " + field)
        history = r.read_json(self.recovery_path)
        receipt = r.read_json(self.receipt_path)
        parent_path = self.out / "coordinator_full_infra03_state.json"
        r.require(receipt.get("experiment_id") == r.EXPERIMENT_ID
                  and receipt.get("root") == str(self.root)
                  and receipt.get("parent") == str(parent_path)
                  and receipt.get("parent_sha256") == r.digest(parent_path)
                  and receipt.get("config_sha256") == r.digest(self.root / r.CONFIG)
                  and receipt.get("initial_consent_sha256") == r.digest(self.options["initial_budget"]),
                  "Original recovery source/config/consent proof changed")
        r.require(history.get("series") == "v3" and history.get("root") == str(self.root)
                  and history.get("phase") == "full" and history.get("max_total_assignments") == 2
                  and history.get("technical_go") is True
                  and history.get("pilot_report") == str(Path(self.options["pilot_report"]).resolve())
                  and history.get("pilot_report_sha256") == r.digest(self.options["pilot_report"]),
                  "Original actual technical GO linkage changed")
        package = self.root / "src/jevlab"
        current = {p.relative_to(package).as_posix(): r.digest(p) for p in
                   sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
        from jevlab.v3.operational import validate_code_transition
        validate_code_transition(policy["old_code_sha256"], current, approved,
                                 r.digest(self.root / r.CONFIG), r.digest(self.options["pilot_report"]))
        r.operational_calendar(self.root, approved)
        archives = r.pilot_inputs(self.root, self.options["pilot_report"])
        ledger = r.budget_ledger(self.root)
        with zipfile.ZipFile(self.assets_path) as assets:
            r.require(len(assets.namelist()) == len(set(assets.namelist())), "Duplicate preserved asset")
            old_source = assets.read("tools/colab/run_v3.py")
            r.require(r.hashlib.sha256(old_source).hexdigest() == receipt["source_sha256"],
                      "Historical controller source proof changed")
            for name, sha in policy["old_code_sha256"].items():
                r.require(r.hashlib.sha256(assets.read("src/jevlab/" + name)).hexdigest() == sha,
                          "Preserved old runtime inventory changed")
        for initial in r.plan("prepared"):
            key, condition = initial["key"], initial["condition"]
            item = (r.job(condition, "smoke", initial["session"] + "-infra03")
                    if initial["phase"] == "smoke" else initial)
            directory = (self.out / "recovery/infra03" / key if key == "smoke/B13_GREEDY" else
                         self.out / "attempts/infra03" / key if initial["phase"] == "smoke" else self.out / key)
            state = r.status_gate(self.root, directory, item)
            r.require(not any(p.is_symlink() for p in (directory, *directory.parents))
                      and not any(p.is_symlink() for p in directory.rglob("*")), "Historical output symlink forbidden")
            r.require(state.get("status") == "completed" and all(state.get(k) is True for k in
                      ("startup_ack", "completed_execution", "verified", "released")),
                      "All thirteen historical jobs must be completed/verified/released")
            entry = ledger["jobs"].get(state.get("run_id"), {})
            r.require(entry.get("key") == key and entry.get("session") == item["session"]
                      and entry.get("output") == str(directory) and entry.get("released") is True
                      and entry.get("release_verified") is True and entry.get("release_evidence") == "backend_absent",
                      "Historical release/charge proof changed", "blocked_budget_or_eta")
            verification = r.read_json(directory / "verification.json")
            r.require(history["verified_jobs"].get(key) == verification and verification.get("ok") is True,
                      "Recorded historical verification changed")
            result = verification["runs"][condition]
            archive = Path(result["archive"])
            notebooks = list((directory / "artifacts").glob(r.NOTEBOOKS[condition] + ".out.*.ipynb"))
            r.require(result.get("ok") is True and result.get("records") == item["expected_rows"]
                      and archive.parent == directory / "artifacts" and r.digest(archive) == result["archive_sha256"]
                      and len(notebooks) == 1 and r.digest(notebooks[0]) == result["notebook_sha256"],
                      "Historical physical coverage/archive changed")
            frozen = history["frozen_evidence"][key]
            r.require(frozen["frozen_files"].get("tools/colab/run_v3.py") == receipt["source_sha256"]
                      and frozen["bundle_manifest"]["config_sha256"] == receipt["config_sha256"]
                      and frozen["frozen_files"].get(frozen["initial_budget"]) == receipt["initial_consent_sha256"],
                      "Historical frozen source/config/consent changed")
            inventory = {n.removeprefix("src/jevlab/"): h for n, h in frozen["bundle_manifest"]["files"].items()
                         if n.startswith("src/jevlab/") and n.endswith(".py")}
            r.require(inventory == policy["old_code_sha256"], "Historical old source inventory changed")
            if item["phase"] == "pilot":
                r.require(archives[condition].name == archive.name
                          and r.digest(archives[condition]) == result["archive_sha256"],
                          "Original six pilot archives not linked")
                with zipfile.ZipFile(archive) as pilot:
                    manifests = [n for n in pilot.namelist() if n == "manifest.json" or n.endswith("/manifest.json")]
                    r.require(len(manifests) == 1, "Historical pilot requires exactly one manifest")
                    manifest = json.loads(pilot.read(manifests[0]))
                r.require(manifest["config"]["code_sha256"] == policy["old_code_sha256"],
                          "Historical pilot executed source inventory changed")
            repaired = state.get("technical_revalidation")
            if repaired:
                directory_history = directory / "revalidation_history"
                original = r.read_json(directory_history / "original_status.json")
                excluded = {"status", "verified", "updated_utc", "technical_revalidation"}
                r.require(r.read_json(directory_history / "receipt.json") == repaired
                          and r.digest(directory_history / "original_status.json") == repaired["original_status_sha256"]
                          and r.digest(directory_history / "original_verification.json") == repaired["original_verification_sha256"]
                          and repaired["run_id"] == state["run_id"] and repaired["job"] == item
                          and repaired["archive_sha256"] == result["archive_sha256"]
                          and repaired["notebook_sha256"] == result["notebook_sha256"],
                          "Recorded historical revalidation receipt changed")
                r.require(original.get("status") == "failed" and original.get("verified") is False
                          and {k: v for k, v in original.items() if k not in excluded} ==
                          {k: v for k, v in state.items() if k not in excluded},
                          "Historical execution/deadline/charges changed")
        return policy

    def job_options(self, item):
        r.require(item in self.jobs and item == r.job(item["condition"], self.phase),
                  "Only canonical post-pilot jobs may be submitted", "blocked_ownership")
        r.require(self.state.get("phase") == self.phase and self.state.get("max_total_assignments") == 2,
                  "Post-pilot phase/cap identity changed", "blocked_ownership")
        self.validate_history()
        self.install_private_auditor()
        return super().job_options(item)

    def revalidate_operator(self, item):
        self.job_options(item)
        r.runtime_approval_gate(self.root, item, self.options["approval"], self.options["pilot_report"])
        # Explicit method reuse, never RecoveryCoordinator.__init__/validate_recovery.
        return RecoveryCoordinator.revalidate_operator(self, item)

    def wait_operator(self, item):
        self.job_options(item)
        directory = self.job_directory(item)
        state = r.status_gate(self.root, directory, item)
        until = state["deadline_epoch"] + 30
        while state["status"] not in {"completed", "failed"}:
            r.require(r.time.time() < until and r.time.monotonic() <
                      state.get("deadline_monotonic", float("inf")) + 30,
                      "Owned worker deadline reached; no restart", "blocked_deadline")
            r.time.sleep(min(r.POLL_SECONDS, until - r.time.time()))
            state = r.status_gate(self.root, directory, item)
        collection_recovery = state.get("technical_collection_recovery")
        if collection_recovery:
            from recover_jfinal_collection_v3 import validate_receipt
            r.require(state["status"] == "completed" and item == r.job("JFINAL", "test"),
                      "Collection recovery is only valid for completed test/JFINAL")
            validate_receipt(self.root, directory, state)
        elif state.get("technical_revalidation"):
            return self.revalidate_operator(item)
        if state["status"] == "failed" and state.get("completed_execution") is True:
            verification = directory / "verification.json"
            try:
                failure = r.read_json(verification)
            except (FileNotFoundError, r.CoordinatorError):
                detail = "Completed execution has no verification.json; collection failed at " + str(
                    state.get("collection_operation", "unknown"))
                if item == r.job("JFINAL", "test"):
                    from recover_jfinal_collection_v3 import RUN_ID
                    if state.get("run_id") == RUN_ID and state.get("collection_operation") == "local_integrity":
                        detail += ("; JFINAL cumulative outer/checkpoint/final expansion is 696186938 bytes"
                                   " > 512 MiB; explicit CPU-only selected-final collection recovery required")
                r.require(verification.is_file(), detail + "; inspect local collection evidence", "blocked_execution")
                raise
            if failure.get("runs", {}).get(item["condition"], {}).get("error") == PROVENANCE_ERROR:
                return self.revalidate_operator(item)
        r.require(state["status"] == "completed" and all(state.get(k) is True for k in
                  ("verified", "released", "completed_execution")), "Operator failed or release unconfirmed", "blocked_execution")
        if not collection_recovery:
            data = r.Backend(directory, item["session"]).identity()
            r.require(state.get("owned_endpoint") and state["owned_endpoint"] not in
                      {row["endpoint"] for row in data["assignments"]}, "Owned endpoint still active")
        auditor = self.install_private_auditor()
        if collection_recovery:
            r.require({k: getattr(auditor, k) for k in AUDITOR_FIELDS} == collection_recovery["auditor"],
                      "Fresh collection recovery auditor identity changed")
        r.runtime_approval_gate(self.root, item, self.options["approval"], self.options["pilot_report"])
        report = auditor.verify_run(directory / "artifacts", conditions=[item["condition"]],
            inputs=self.root / "data/v3/test_inputs.jsonl", config=self.root / r.CONFIG, prompts=self.root / "prompts",
            expected_count=item["count"], split="test", run_tag=item["params"]["RUN_TAG"],
            approval=self.options["approval"], pilot_report=self.options["pilot_report"])
        r.require(report.get("ok") is True and set(report.get("runs", {})) == {item["condition"]}
                  and report["runs"][item["condition"]].get("records") == item["expected_rows"],
                  "Fresh current-checker/V2 coverage audit failed")
        if collection_recovery:
            from recover_jfinal_collection_v3 import PINS, FINAL_MEMBER, NOTEBOOK_MEMBER
            result = report["runs"]["JFINAL"]
            r.require(result.get("archive_sha256") == PINS[FINAL_MEMBER][1]
                      and result.get("notebook_sha256") == PINS[NOTEBOOK_MEMBER][1],
                      "Fresh collection recovery artifact SHA mismatch")
        self.state["verified_jobs"][item["key"]] = report
        self.save("job_verified_and_released")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--phase", choices=("test", "latency"), required=True)
    parser.add_argument("--attempt-prefix", choices=("infra03",))
    parser.add_argument("--authorize-initial", action="store_true", required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true", required=True)
    for name in ("initial-budget", "host-ack", "approval", "pilot-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    options = {k: str(getattr(args, k).resolve()) for k in ("initial_budget", "host_ack", "approval", "pilot_report")}
    options.update(authorize_initial=args.authorize_initial, authorize_confirmatory=args.authorize_confirmatory,
                   attempt_prefix=args.attempt_prefix, dev_bundle=False)
    try:
        coordinator = PostPilotCoordinator(args.root, args.phase, **options)
        rc = coordinator.run()
        print(json.dumps(dict(phase=args.phase, state=str(coordinator.path), outcome=coordinator.state.get("outcome"),
                              verified_jobs=len(coordinator.state["verified_jobs"]), rc=rc)))
        return rc
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps(dict(phase=args.phase, outcome=getattr(error, "state", "blocked_integrity"),
                              error_type=type(error).__name__, rc=1)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
