"""Resume the authorized infra03 plan without rewriting its failed startup."""

from __future__ import annotations

import argparse
import importlib.util
import importlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3

TARGET = "smoke/B13_GREEDY"
FAILED_RUN_ID = "0cc62aae2e3a4a05982f31dc73b9b325"
PRIOR = ("G_SINGLE", "G_GREEDY", "JFINAL", "B13")
AUTHORIZATION = (
    "User verified an actual A100 allocation and explicitly requested continuing "
    "the existing plan until definitive results; recover only the released, "
    "unassigned infra03 B13_GREEDY startup. Preserve all accumulated charges, "
    "including the probe, the conditional 25-hour budget, and two global assignments."
)
PROVENANCE_ERROR = "ValueError: Missing/nonpublic pinned source provenance"


class RecoveryCoordinator(run_v3.Coordinator):
    def __init__(self, root=run_v3.ROOT, phase="full", **options):
        run_v3.require(phase == "full" and options.get("attempt_prefix") == "infra03"
                       and options.get("authorize_initial") is True,
                       "Recovery requires authorized full infra03", "blocked_user_authorization")
        super().__init__(root, phase, **options)
        self.parent_path = self.path
        self.path = self.out / "coordinator_full_infra03_recovery_state.json"
        self.receipt_path = self.out / "recovery/infra03/recovery_receipt.json"
        # Share the original coordinator lock, including the receipt's first creation.
        with run_v3.flock(self.out / "coordinator_full.lock"):
            self.validate_recovery(create=True)

    def job_directory(self, item):
        if item["key"] == TARGET:
            return self.out / "recovery/infra03" / item["key"]
        return super().job_directory(item)

    def job_options(self, item):
        # Called by the unchanged runner before preflight and every submission.
        self.validate_recovery()
        options = super().job_options(item)
        if item["key"] == TARGET:
            options["attempt_prefix"] = None
        return options

    def install_private_auditor(self, native_container=False):
        """Select the independent validator in this parent process, never workers."""
        path = self.root / "tools/verify_run_v3.py"
        auditor_path = self.root / ("tools/audit_native_container_v3.py" if native_container else "tools/audit_inherited_v3.py")
        run_v3.require(not any(p.is_symlink() for file in (path, auditor_path)
                               for p in (file, *file.parents)), "Validator symlink forbidden")
        sha = run_v3.digest(path)
        evidence = run_v3.read_json(self.parent_path).get("frozen_evidence", {})
        run_v3.require(evidence and all(
            frozen.get("frozen_files", {}).get("tools/verify_run_v3.py") == sha
            for frozen in evidence.values()), "Original frozen verifier SHA mismatch")
        spec = importlib.util.spec_from_file_location("_recovery_auditor", auditor_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        module = helper.load_verifier()
        run_v3.require(module.VERIFIER_SHA256 == sha
                       and module.AUDITOR_SHA256 == run_v3.digest(auditor_path),
                       "Auditor/verifier identity mismatch")
        self.state["private_auditor"] = {k: getattr(module, k) for k in
            ("AUDITOR_ID", "AUDITOR_VERSION", "AUDITOR_SHA256", "VERIFIER_SHA256")}
        sys.modules["verify_run_v3"] = module
        # A prior parent import must not retain the original function binding.
        if "pilot_decision_v3" in sys.modules:
            importlib.reload(sys.modules["pilot_decision_v3"])
        return module

    def decide_pilot(self):
        self.install_private_auditor(native_container=True)
        return super().decide_pilot()

    def revalidate_operator(self, item):
        """Audit an existing released execution; never allocate, restart or infer."""
        require = run_v3.require
        directory = self.job_directory(item)
        with run_v3.flock("/tmp/jev-v3-operator-" + item["session"] + ".lock"):
            require(item in self.jobs and item["phase"] in {"pilot", "test", "latency"},
                    "Only canonical numerical executions may be revalidated")
            require(not any(p.is_symlink() for p in (directory, *directory.parents))
                    and not any(p.is_symlink() for p in directory.rglob("*")),
                    "Revalidation output symlink forbidden", "blocked_ownership")
            state = run_v3.status_gate(self.root, directory, item)
            history = directory / "revalidation_history"
            receipt_path = history / "receipt.json"
            repaired = state.get("technical_revalidation")
            require(type(item["count"]) is int and item["count"] > 0
                    and type(item["expected_rows"]) is int and item["expected_rows"] > 0,
                    "Exact numeric execution coverage required")
            require((state.get("status") == "failed" and state.get("verified") is False)
                    or (state.get("status") == "completed" and state.get("verified") is True and repaired),
                    "Not an original verification-only failure")
            health = state.get("health", {})
            execution = health.get("execution", {})
            require(state.get("completed_execution") is True and state.get("released") is True
                    and health.get("known") is True and health.get("alive") is False
                    and health.get("run_id") == state.get("run_id")
                    and execution.get("done") is True and type(execution.get("rc")) is int
                    and execution["rc"] == 0, "Execution/release proof missing")
            original_status = history / "original_status.json"
            original_verification = history / "original_verification.json"
            if repaired:
                original = run_v3.read_json(original_status)
                excluded = {"status", "verified", "updated_utc", "technical_revalidation"}
                require(original.get("status") == "failed" and original.get("verified") is False
                        and {k: v for k, v in original.items() if k not in excluded} ==
                        {k: v for k, v in state.items() if k not in excluded},
                        "Original execution identity/deadline/charges changed")
            failure = run_v3.read_json(original_verification if repaired else directory / "verification.json")
            require(failure.get("ok") is False and set(failure.get("runs", {})) == {item["condition"]}
                    and failure["runs"][item["condition"]] == {"ok": False, "error": PROVENANCE_ERROR},
                    "Only exact inherited provenance boundary may be repaired")
            data = run_v3.Backend(directory, item["session"]).identity()
            require(state.get("owned_endpoint") and state["owned_endpoint"] not in
                    {r["endpoint"] for r in data["assignments"]}, "Owned endpoint still active")
            auditor = self.install_private_auditor(native_container=item["condition"] in {"JFINAL", "LATENCY"})
            inputs = self.root / "data/v3" / (item["params"]["SPLIT"] + "_inputs.jsonl")
            options = state["options"]
            report = auditor.verify_run(directory / "artifacts", conditions=[item["condition"]],
                inputs=inputs, config=self.root / run_v3.CONFIG, prompts=self.root / "prompts",
                expected_count=item["count"], split=item["params"]["SPLIT"], run_tag=item["params"]["RUN_TAG"],
                approval=options.get("approval"), pilot_report=options.get("pilot_report"))
            require(report.get("ok") is True and set(report.get("runs", {})) == {item["condition"]},
                    "Fresh independent full audit failed")
            result = report["runs"][item["condition"]]
            require(result.get("ok") is True and result.get("condition") == item["condition"]
                    and type(result.get("records")) is int and result["records"] == item["expected_rows"]
                    and result.get("certificate_schema") == "v3-offline-audit-1", "Full audit coverage/schema mismatch")
            archive = Path(result["archive"])
            notebooks = auditor.condition_notebooks(directory / "artifacts", item["condition"])
            require(archive.resolve().is_relative_to((directory / "artifacts").resolve())
                    and run_v3.digest(archive) == result["archive_sha256"] and len(notebooks) == 1
                    and run_v3.digest(notebooks[0]) == result["notebook_sha256"], "Audited physical artifacts changed")
            history.mkdir(exist_ok=True)
            for source, destination in ((directory / "status.json", original_status),
                                        (directory / "verification.json", original_verification)):
                if not destination.exists():
                    require(not repaired, "Original revalidation history missing")
                    with destination.open("xb") as stream:
                        stream.write(source.read_bytes())
                    destination.chmod(0o444)
                elif not repaired:
                    require(destination.read_bytes() == source.read_bytes(), "Original failure history changed")
            receipt = dict(schema_version=1, run_id=state["run_id"], job=item,
                original_status_sha256=run_v3.digest(original_status),
                original_verification_sha256=run_v3.digest(original_verification),
                auditor=self.state["private_auditor"], certificate_schema=result["certificate_schema"],
                certificate_schema_sha256=run_v3.hashlib.sha256(result["certificate_schema"].encode("ascii")).hexdigest(),
                archive_sha256=result["archive_sha256"], notebook_sha256=result["notebook_sha256"],
                config_sha256=run_v3.digest(self.root / run_v3.CONFIG), inputs_sha256=run_v3.digest(inputs),
                scope="Technical revalidation only; unchanged execution, predictions, parameters and budget")
            if receipt_path.exists():
                require(run_v3.read_json(receipt_path) == receipt, "Revalidation receipt changed")
            else:
                require(not repaired, "Revalidation receipt missing")
                run_v3.atomic_json(receipt_path, receipt)
                receipt_path.chmod(0o444)
            if repaired:
                require(repaired == receipt and run_v3.read_json(directory / "verification.json") == report,
                        "Promoted verification changed")
            else:
                run_v3.atomic_json(directory / "verification.json", report)
                run_v3.Operator(self.root, directory, state).save("completed", verified=True,
                    technical_revalidation=receipt)
            self.state["verified_jobs"][item["key"]] = report
            self.save("job_verified_and_released")
            return report

    def local_analysis(self):
        """Dispatch once after LAT, then independently reverify the real CPU receipt."""
        require = run_v3.require
        self.validate_recovery()
        item = next(j for j in self.jobs if j["phase"] == "analysis")
        preceding = self.jobs[:self.jobs.index(item)]
        require(len(preceding) == 20 and all(self.state["verified_jobs"].get(j["key"], {}).get("ok") is True
                for j in preceding),
                "All twenty preceding canonical jobs must be verified")
        for previous in preceding:
            state = run_v3.status_gate(self.root, self.job_directory(previous), previous)
            require(state.get("status") == "completed" and all(state.get(k) is True for k in
                    ("startup_ack", "completed_execution", "verified", "released")),
                    "Pre-analysis completed execution/release barrier closed")
        options = self.job_options(item)
        approved, _ = run_v3.approval_gate(self.root, "analysis", options.get("approval"),
            options.get("pilot_report"), options.get("authorize_confirmatory"), fresh=False)
        initial = run_v3.initial_budget_gate(self.root, options.get("initial_budget"))
        initial_sha = run_v3.hashlib.sha256(json.dumps(initial, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode()).hexdigest()
        require(approved.get("experiment_id") == run_v3.EXPERIMENT_ID
                and approved.get("initial_authorization_sha256") == initial_sha
                and approved.get("no_additional_allowance_after_pilot") is True,
                "Analysis approval must bind original conditional consent")
        ledger = run_v3.budget_ledger(self.root)
        require(ledger.get("jobs") and all(e.get("released") is True and e.get("release_verified") is True
                for e in ledger["jobs"].values()), "All accumulated leases must be released")
        directory = self.job_directory(item)
        require(not any(p.is_symlink() for p in (directory, *directory.parents))
                and not any(p.is_symlink() for p in directory.rglob("*")), "Local analysis symlink forbidden")

        def native(path):
            value = str(path)
            match = run_v3.re.fullmatch(r"/mnt/([a-zA-Z])/(.+)", value)
            return str(PureWindowsPath(match[1].upper() + ":/" + match[2])) if match else value

        def local(path):
            win = PureWindowsPath(str(path))
            if run_v3.os.name == "posix" and len(win.drive) == 2 and win.drive[1] == ":":
                return Path("/mnt/" + win.drive[0].lower() + "/" + "/".join(win.parts[1:]))
            return Path(path)

        status_path = directory / "status.json"
        existing = status_path.exists()
        if existing:
            state = run_v3.status_gate(self.root, directory, item)
            require(state.get("status") == "completed" and all(state.get(k) is True for k in
                    ("startup_ack", "completed_execution", "verified", "released"))
                    and state.get("allocation_attempted") is False and state.get("owned_endpoint") is None
                    and state.get("execution") == "local_cpu_papermill" and state.get("options") == options,
                    "Existing analysis is not a completed local CPU execution")
            require(state.get("local_analysis_receipt_sha256") ==
                    run_v3.digest(directory / "local_analysis_receipt.json"), "Local completion receipt changed")
        else:
            require(not directory.exists() or not any(directory.iterdir()), "Partial local analysis output collision")
            python = self.root / ".venv-analysis/Scripts/python.exe"
            helper = self.root / "tools/local_analysis_v3.py"
            require(python.is_file() and helper.is_file(), "Native Windows analysis environment missing")
            directory.mkdir(parents=True, exist_ok=True)
            rc, _ = run_v3.bounded_call([str(python), native(helper), "--root", native(self.root),
                "--approval", native(options["approval"]), "--pilot-report", native(options["pilot_report"])],
                timeout=run_v3.JOB_SECONDS, log=directory / "local_analysis.log")
            require(rc == 0, "Native local analysis failed; no retry", "blocked_execution")
        receipt = run_v3.read_json(directory / "local_analysis_receipt.json")
        require(receipt.get("schema_version") == 1 and receipt.get("execution") == "local_cpu_papermill"
                and receipt.get("allocation_attempted") is False and receipt.get("owned_endpoint") is None
                and receipt.get("gpu_seconds") == 0, "Genuine local CPU receipt required")
        result = receipt["result"]
        archive = directory / "artifacts/analysis_confirmatory_v3.zip"
        notebooks = list((directory / "artifacts").glob("07_ANALYSIS.out.*.ipynb"))
        require(result.get("ok") is True and len(notebooks) == 1
                and local(result["archive"]) == archive and local(result["notebook"]) == notebooks[0]
                and local(result["receipt"]) == directory / "local_analysis_receipt.json"
                and local(result["verification"]) == directory / "verification.json",
                "Local receipt artifact identity mismatch")
        require(receipt.get("pins") and receipt.get("sources"), "Local receipt evidence missing")
        for pin in (*receipt["pins"].values(), *receipt["sources"].values()):
            require(run_v3.digest(local(pin["path"])) == pin["sha256"], "Local receipt pinned evidence changed")
        require(run_v3.digest(directory / "local_analysis.ipynb") == receipt["helper_notebook_sha256"]
                and run_v3.digest(directory / "analysis_inputs_receipt.json") == receipt["input_receipt_sha256"],
                "Local helper/input receipt changed")
        stored = run_v3.read_json(directory / "verification.json")
        # The frozen verifier writes its basic report here, not over helper metadata.
        with TemporaryDirectory(prefix="jev-local-analysis-") as temporary:
            # A namespace path redirects only verification.json, leaving artifacts in place.
            class VerificationOutput:
                def __truediv__(self, name):
                    return Path(temporary) / name if name == "verification.json" else directory / name
            run_v3.Operator.verify(SimpleNamespace(item=item, out=VerificationOutput()))
            verified = run_v3.read_json(Path(temporary) / "verification.json")
        require(verified.get("ok") is True and stored.get("ok") is True
                and all(verified.get(k) == result.get(k) == stored.get(k)
                for k in ("archive_sha256", "notebook_sha256")), "Fresh local analysis verification mismatch")
        if not existing:
            run_v3.atomic_json(status_path, dict(series="v3", root=str(self.root), output=str(directory),
                cli_config=str(directory / "sessions.json"), job=item, options=options, status="completed",
                startup_ack=True, completed_execution=True, verified=True, released=True,
                allocation_attempted=False, owned_endpoint=None, execution="local_cpu_papermill",
                local_analysis_receipt_sha256=run_v3.digest(directory / "local_analysis_receipt.json")))
        self.state["verified_jobs"][item["key"]] = result
        self.save("job_verified_and_released")
        return result

    def wait_operator(self, item):
        if item["phase"] == "analysis":
            return self.local_analysis()
        directory = self.job_directory(item)
        state = run_v3.status_gate(self.root, directory, item)
        until = state["deadline_epoch"] + 30
        while state["status"] not in {"completed", "failed"}:
            run_v3.require(run_v3.time.time() < until and run_v3.time.monotonic() <
                           state.get("deadline_monotonic", float("inf")) + 30,
                           "Owned worker deadline reached; reconcile only, never inline restart", "blocked_deadline")
            run_v3.time.sleep(min(run_v3.POLL_SECONDS, until - run_v3.time.time()))
            state = run_v3.status_gate(self.root, directory, item)
        if item["phase"] in {"pilot", "test", "latency"}:
            if state.get("technical_revalidation"):
                result = self.revalidate_operator(item)
                if item["phase"] == "latency":
                    self.local_analysis()
                return result
            if state["status"] == "failed" and state.get("completed_execution") is True:
                failure = run_v3.read_json(directory / "verification.json")
                if failure.get("runs", {}).get(item["condition"], {}).get("error") == PROVENANCE_ERROR:
                    result = self.revalidate_operator(item)
                    if item["phase"] == "latency":
                        self.local_analysis()
                    return result
        result = super().wait_operator(item)
        if item["phase"] == "latency":
            self.local_analysis()
        return result

    def validate_recovery(self, *, create=False):
        require = run_v3.require
        recovery_output = self.out / "recovery/infra03" / TARGET
        for path in (self.parent_path, self.path, self.receipt_path, recovery_output):
            require(not any(p.is_symlink() for p in (path, *path.parents)),
                    "Recovery metadata symlink forbidden", "blocked_ownership")
        parent = run_v3.read_json(self.parent_path)
        require(parent.get("series") == "v3" and parent.get("root") == str(self.root)
                and parent.get("phase") == "full" and parent.get("jobs") == self.jobs
                and parent.get("max_total_assignments") == 2,
                "Original full infra03 coordinator identity changed", "blocked_ownership")
        config_sha = run_v3.digest(self.root / run_v3.CONFIG)
        source_sha = run_v3.digest(self.root / "tools/colab/run_v3.py")
        initial = self.options.get("initial_budget")
        require(initial is not None, "Recovery initial consent required", "blocked_budget_approval")
        initial_sha = run_v3.digest(initial)
        evidence = parent.get("frozen_evidence", {})
        required_keys = {j["key"] for j in self.jobs if j["phase"] in {"smoke", "pilot"}}
        require(required_keys <= evidence.keys(), "Original full initial preflights missing")
        for key in required_keys:
            frozen = evidence[key]
            files = frozen.get("frozen_files", {})
            consent = frozen.get("initial_budget")
            require(frozen.get("bundle_manifest", {}).get("config_sha256") == config_sha
                    and files.get("tools/colab/run_v3.py") == source_sha
                    and consent is not None and files.get(consent) == initial_sha,
                    "Original frozen config/source/consent mismatch")
        ledger = run_v3.budget_ledger(self.root)
        require(ledger.get("experiment_id") == run_v3.EXPERIMENT_ID
                and ledger.get("max_gpu_seconds") == run_v3.MAX_GPU_SECONDS,
                "Recovery ledger experiment/cap changed", "blocked_budget_or_eta")
        hashes = {}
        failed_entry = None
        for condition in (*PRIOR, "B13_GREEDY"):
            key = "smoke/" + condition
            item = next(j for j in self.jobs if j["key"] == key)
            directory = super().job_directory(item)
            require(not any(p.is_symlink() for p in (directory / "status.json", directory, *directory.parents)),
                    "Original output symlink forbidden", "blocked_ownership")
            state = run_v3.status_gate(self.root, directory, item)
            hashes[key] = run_v3.digest(directory / "status.json")
            require(state.get("released") is True and state.get("allocation_attempted") is True,
                    "Original smoke release/allocation proof missing", "blocked_ownership")
            entry = ledger.get("jobs", {}).get(state.get("run_id"), {})
            require(entry.get("key") == key and entry.get("session") == item["session"]
                    and entry.get("output") == str(directory) and entry.get("released") is True
                    and entry.get("release_verified") is True and entry.get("release_evidence") == "backend_absent",
                    "Original smoke lease identity/release proof missing", "blocked_budget_or_eta")
            if key != TARGET:
                require(state.get("status") == "completed" and state.get("startup_ack") is True
                        and state.get("completed_execution") is True and state.get("verified") is True,
                        "Prior four successful smokes must remain completed")
                continue
            charge = entry.get("actual_seconds")
            require(state.get("run_id") == FAILED_RUN_ID and state.get("status") == "failed"
                    and not state.get("remote_startup_ack") and not state.get("owned_endpoint")
                    and state.get("before_endpoints") == [] and state.get("completed_execution") is False
                    and state.get("verified") is False and state.get("unassigned_startup_reconciled") is True
                    and not entry.get("endpoint") and entry.get("unassigned_startup_reconciled") is True
                    and type(charge) in (int, float) and math.isfinite(charge) and charge >= 1513
                    and state.get("observed_gpu_seconds_upper_bound") == charge,
                    "Failed startup is not the reconciled unassigned infra03 lease", "blocked_ownership")
            for artifact in directory.rglob("*"):
                require(not artifact.is_symlink(), "Original failed output symlink forbidden", "blocked_ownership")
                if artifact.is_file():
                    parts = artifact.relative_to(directory).parts
                    require(artifact.suffix not in {".ipynb", ".zip"}
                            and not artifact.name.lower().startswith(("prediction", "final"))
                            and not any(p in {"results", "artifacts", "checkpoints"} for p in parts)
                            and (artifact.suffix != ".jsonl" or artifact.name == "events.jsonl"),
                            "Failed startup contains execution/results artifacts", "blocked_execution")
            failed_entry = entry
        receipt = dict(schema_version=1, experiment_id=run_v3.EXPERIMENT_ID, root=str(self.root),
                       parent=str(self.parent_path), parent_sha256=run_v3.digest(self.parent_path),
                       failed_status_sha256=hashes.pop(TARGET), prior_status_sha256=hashes,
                       initial_consent_sha256=initial_sha, config_sha256=config_sha, source_sha256=source_sha,
                       failed_run_id=FAILED_RUN_ID, failed_lease=failed_entry,
                       authorization=AUTHORIZATION, recovery_output=str(recovery_output))
        if self.receipt_path.exists():
            require(run_v3.read_json(self.receipt_path) == receipt,
                    "Immutable recovery receipt changed", "blocked_ownership")
        else:
            require(create and not self.path.exists(), "Recovery receipt missing for existing coordinator", "blocked_ownership")
            require(not recovery_output.exists() or not any(recovery_output.iterdir()),
                    "Recovery output must be fresh before receipt", "blocked_ownership")
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            run_v3.atomic_json(self.receipt_path, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=run_v3.ROOT)
    parser.add_argument("--phase", choices=("full",), required=True)
    parser.add_argument("--attempt-prefix", choices=("infra03",), required=True)
    parser.add_argument("--authorize-initial", action="store_true", required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    for name in ("initial-budget", "host-ack", "approval", "pilot-report"):
        parser.add_argument("--" + name, type=Path, required=name in {"initial-budget", "host-ack"})
    args = parser.parse_args(argv)
    options = {k: str(getattr(args, k).resolve()) if getattr(args, k) is not None else None
               for k in ("initial_budget", "host_ack", "approval", "pilot_report")}
    options.update(authorize_initial=args.authorize_initial, authorize_confirmatory=args.authorize_confirmatory,
                   attempt_prefix=args.attempt_prefix, dev_bundle=False)
    try:
        coordinator = RecoveryCoordinator(args.root, args.phase, **options)
        rc = coordinator.run()
        print(json.dumps(coordinator.state, indent=2))
        return rc
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(status=getattr(error, "state", "blocked_integrity"), error_type=type(error).__name__)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
