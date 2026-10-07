"""Local CPU analysis, never allocation or inference.

API: execute(root, approval, pilotreport) returns a JSON-serializable verification
dict with archive/notebook/receipt/verification paths and SHA256s. The parent owns
dispatch and status.json; failures propagate without retry or synthetic completion.
Run with the parent's .venv-analysis Python (system site packages + papermill).
The existing python3 kernel must have the analysis dependencies; no installs here.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PureWindowsPath
import sys
from types import SimpleNamespace
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "colab"))
import run_v3 as core

OUTPUTS = ("analysis.json", "report.md", "parser_audit.json", "graded_runs.csv",
           "final_table.csv", "primary_comparison.json", "secondary_comparisons.csv",
           "shared_controls.csv", "latency_per_problem.csv")


def safe(path):
    path = Path(path)
    core.require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                         for p in (path, *path.parents)),
                 "Symlink/junction forbidden: " + str(path))
    return path


def digest(path):
    return core.digest(safe(path))


def wsl_path(path):
    """Translate drive identity, not via Path.resolve on a WSL string."""
    win = PureWindowsPath(str(path))
    if win.drive:
        core.require(len(win.drive) == 2 and win.drive[1] == ":" and win.is_absolute(),
                     "Only absolute Windows drive paths supported")
        return "/mnt/" + win.drive[0].lower() + "/" + "/".join(win.parts[1:])
    return str(path)


def write_receipt(path, value):
    with safe(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.chmod(0o444)


def execute(root, approval, pilotreport):
    """Fresh local analysis only after all seven canonical released runs audit."""
    root = safe(root).resolve()
    approval, pilotreport = safe(approval).resolve(), safe(pilotreport).resolve()
    config = root / core.CONFIG
    initial_path = root / "results/v3/initial_budget.json"
    ledger_path = root / core.BUDGET_LEDGER
    out = safe(root / "results/v3/analysis/ANALYSIS")
    safe(config)
    safe(initial_path)
    safe(ledger_path)
    initial = core.initial_budget_gate(root, initial_path)
    approved, _ = core.approval_gate(root, "analysis", approval, pilotreport,
                                     authorized=True, fresh=False)
    initial_sha = hashlib.sha256(json.dumps(initial, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    core.require(approved.get("experiment_id") == core.EXPERIMENT_ID
                 and approved.get("initial_authorization_sha256") == initial_sha
                 and approved.get("no_additional_allowance_after_pilot") is True,
                 "Approval must bind original conditional authorization")
    ledger = core.read_json(ledger_path)
    jobs = ledger.get("jobs", {})
    core.require(ledger.get("experiment_id") == core.EXPERIMENT_ID
                 and ledger.get("max_gpu_seconds") == core.MAX_GPU_SECONDS
                 and isinstance(jobs, dict) and jobs, "Initial 25h ledger missing/changed")
    for entry in jobs.values():
        charge = entry.get("actual_seconds")
        no_allocation = (entry.get("release_evidence") == "durable_no_allocation"
                         and entry.get("allocation_attempted") is False and charge == 0
                         and not entry.get("endpoint"))
        core.require(entry.get("released") is True and entry.get("release_verified") is True
                     and (entry.get("release_evidence") == "backend_absent" or no_allocation)
                     and type(charge) in (int, float) and math.isfinite(charge) and charge >= 0,
                     "All ledger leases must be authoritatively released with finite charges")
    core.require(sum(e["actual_seconds"] for e in jobs.values()) <=
                 min(core.MAX_GPU_SECONDS, approved["max_gpu_hours"] * 3600), "25h accumulated cap exceeded")
    # Historical GPU leases cannot disappear from the accumulated ledger.
    for path in (root / "results/v3").rglob("status.json"):
        if any(p in {"remote", "pilot_verified", "artifacts", "checkpoints"}
               for p in path.relative_to(root / "results/v3").parts):
            continue
        state = core.read_json(safe(path))
        if state.get("allocation_attempted") and state.get("job", {}).get("gpu") != "CPU":
            entry = jobs.get(state.get("run_id"), {})
            core.require(state.get("root") == wsl_path(root) and state.get("series") == "v3"
                         and state.get("output") == wsl_path(path.parent)
                         and state.get("released") is True
                         and entry.get("output") == wsl_path(path.parent)
                         and entry.get("key") == state.get("job", {}).get("key")
                         and entry.get("session") == state.get("job", {}).get("session"), "Historical lease identity missing")
    amended = "operational_amendment" in approved
    latency_status = safe(root / "results/v3/latency/LATENCY/status.json")
    auditor_path = safe(root / ("tools/audit_operational_latency_v3.py" if amended
                                else "tools/audit_native_container_v3.py"))
    spec = importlib.util.spec_from_file_location("_local_analysis_auditor_" + uuid.uuid4().hex, auditor_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    auditor = helper.load_verifier()
    core.require(auditor.AUDITOR_SHA256 == digest(auditor_path)
                 and auditor.VERIFIER_SHA256 == digest(root / "tools/verify_run_v3.py"), "Auditor identity changed")
    if amended:
        phase_receipt = core.read_json(latency_status).get("technical_phase_revalidation", {})
        core.require(auditor.AUDITOR_VERSION == helper.AUDITOR_VERSION == "3.0.0"
                     and phase_receipt.get("auditor", {}).get("AUDITOR_SHA256") == auditor.AUDITOR_SHA256
                     and digest(approval) == helper.APPROVAL_SHA256
                     and auditor.VERIFIER_SHA256 == helper.VERIFIER_SHA256
                     and auditor.INHERITED_AUDITOR_SHA256 == helper.INHERITED_AUDITOR_SHA256
                     == digest(root / "tools/audit_native_container_v3.py")
                     and auditor.V1_AUDITOR_SHA256 == helper.V1_AUDITOR_SHA256
                     == digest(root / "tools/audit_inherited_v3.py"),
                     "Amended analysis requires parent V3 phase-proof receipt and original SHA chain")
    sources, evidence = {}, {}
    for item in [*core.plan("test"), *core.plan("latency")]:
        directory = safe(root / "results/v3" / item["key"])
        state = core.read_json(safe(directory / "status.json"))
        core.require(state.get("series") == "v3" and state.get("root") == wsl_path(root)
                     and state.get("output") == wsl_path(directory) and state.get("job") == item
                     and state.get("cli_config") == wsl_path(directory / "sessions.json")
                     and state.get("status") == "completed" and state.get("completed_execution") is True
                     and state.get("verified") is True and state.get("released") is True,
                     "Canonical completed/verified/released execution required")
        options = state.get("options", {})
        core.require(options.get("approval") == wsl_path(approval)
                     and options.get("pilot_report") == wsl_path(pilotreport)
                     and options.get("authorize_confirmatory") is True, "Execution approval/report binding changed")
        entry = jobs.get(state.get("run_id"), {})
        core.require(entry.get("key") == item["key"] and entry.get("session") == item["session"]
                     and entry.get("output") == wsl_path(directory), "Canonical ledger lease missing")
        artifacts = safe(directory / "artifacts")
        files = list(artifacts.iterdir())
        core.require(len(files) == 2 and all(safe(p).is_file() for p in files), "Exactly ZIP/raw notebook pair required")
        condition = item["condition"]
        report = auditor.verify_run(artifacts, conditions=[condition], inputs=root / "data/v3/test_inputs.jsonl",
            config=config, prompts=root / "prompts", approval=approval, pilot_report=pilotreport,
            expected_count=500, split="test", run_tag="confirmatory_v3")
        core.require(report.get("ok") is True and set(report.get("runs", {})) == {condition}, "Fresh audit failed")
        result = report["runs"][condition]
        stored = core.read_json(safe(directory / "verification.json"))
        core.require(stored.get("ok") is True and set(stored.get("runs", {})) == {condition}, "Stored audit missing")
        previous = stored["runs"][condition]
        if amended and condition == "LATENCY":
            proof = result.get("proof_projection", {})
            metadata = report.get("auditor", {})
            core.require(metadata.get("sha256") == auditor.AUDITOR_SHA256
                         and metadata.get("version") == "3.0.0"
                         and stored.get("auditor") == metadata
                         and previous.get("auditor") == result.get("auditor") == metadata
                         and proof.get("applied") is True and proof.get("memory_only") is True
                         and previous.get("proof_projection") == proof,
                         "Stored LAT phase proof must match fresh V3 artifact audit")
        notebooks = auditor.condition_notebooks(artifacts, condition)
        archive = artifacts / Path(str(result.get("archive", "")).replace("\\", "/")).name
        core.require(len(notebooks) == 1 and set(files) == {archive, notebooks[0]}
                     and Path(result["archive"]).resolve() == archive.resolve(), "Noncanonical artifacts")
        core.require(result.get("ok") is True and result.get("condition") == condition
                     and type(result.get("records")) is int and result["records"] == item["expected_rows"]
                     and result.get("certificate_schema") == "v3-offline-audit-1"
                     and all(previous.get(k) == result.get(k) for k in
                             ("condition", "records", "archive_sha256", "notebook_sha256"))
                     and digest(archive) == result["archive_sha256"]
                     and digest(notebooks[0]) == result["notebook_sha256"], "Stored/physical audit hash or coverage mismatch")
        evidence[condition] = report
        for path in (archive, notebooks[0]):
            core.require(path.name.casefold() not in {n.casefold() for n in sources}, "Staging name collision")
            sources[path.name] = {"path": str(path), "sha256": digest(path)}
    staging, artifacts, reports = (safe(out / n) for n in ("analysis_inputs", "artifacts", "reports"))
    receipt_path = safe(out / "local_analysis_receipt.json")
    for path in (staging, artifacts, reports, receipt_path, out / "analysis_inputs_receipt.json",
                 out / "local_analysis.ipynb", out / "verification.json"):
        core.require(not safe(path).exists(), "Local analysis output collision: " + str(path))
    code_paths = {"helper": Path(__file__), "auditor": auditor_path,
                  "verifier": root / "tools/verify_run_v3.py", "original_analysis": root / "src/jevlab/v3/analysis.py",
                  "config": config, "approval": approval, "pilot_report": pilotreport,
                   "initial": initial_path, "ledger": ledger_path}
    if amended:
        code_paths.update(v2_auditor=root / "tools/audit_native_container_v3.py",
                          v1_auditor=root / "tools/audit_inherited_v3.py",
                          latency_status=latency_status,
                          latency_verification=root / "results/v3/latency/LATENCY/verification.json")
    pins = {name: {"path": str(path), "sha256": digest(path)} for name, path in code_paths.items()}
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    artifacts.mkdir()
    for name, source in sources.items():
        with (staging / name).open("xb") as stream:
            stream.write(safe(source["path"]).read_bytes())
        core.require(digest(staging / name) == source["sha256"], "Copy changed bytes")
        (staging / name).chmod(0o444)
    write_receipt(out / "analysis_inputs_receipt.json", dict(sources=sources, audits=evidence, pins=pins))
    import nbformat
    import papermill
    archive = artifacts / "analysis_confirmatory_v3.zip"
    notebook = artifacts / ("07_ANALYSIS.out." + str(uuid.uuid4()) + ".ipynb")
    code = f'''import sys, importlib.util, zipfile
from pathlib import Path
root = Path({str(root)!r})
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'tools'))
spec = importlib.util.spec_from_file_location('_local_private_auditor', {str(auditor_path)!r})
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
auditor = helper.load_verifier()
from jevlab.v3 import analysis
analysis._integrity_module = lambda: auditor
analysis.run_analysis(Path({str(staging)!r}), root / 'data/v3/test_gold.jsonl',
    Path({str(reports)!r}), 'confirmatory_v3', config=root / {core.CONFIG!r},
    prompts=root / 'prompts', approval=Path({str(approval)!r}),
    inputs=root / 'data/v3/test_inputs.jsonl', pilot_report=Path({str(pilotreport)!r}))
with zipfile.ZipFile({str(archive)!r}, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for name in {OUTPUTS!r}:
        z.write(Path({str(reports)!r}) / name, arcname=name)
print({str(archive)!r})
'''
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)], metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    helper_nb = out / "local_analysis.ipynb"
    with helper_nb.open("x", encoding="utf-8") as stream:
        nbformat.write(nb, stream)
    papermill.execute_notebook(str(helper_nb), str(notebook), kernel_name="python3", cwd=str(root), progress_bar=False)
    executed = nbformat.read(safe(notebook), as_version=4)
    core.require(len(executed.cells) == 1 and executed.cells[0].cell_type == "code"
                 and executed.cells[0].source == code and type(executed.cells[0].execution_count) is int
                 and executed.cells[0].execution_count > 0, "Raw helper notebook source/execution changed")
    for pin in [*pins.values(), *sources.values()]:
        core.require(digest(pin["path"]) == pin["sha256"], "Pinned evidence changed during analysis")
    core.require({p.name for p in staging.iterdir()} == set(sources)
                 and all(digest(staging / n) == s["sha256"] for n, s in sources.items()), "Staging changed")
    with zipfile.ZipFile(safe(archive)) as z:
        core.require(len(z.namelist()) == len(OUTPUTS) and set(z.namelist()) == set(OUTPUTS)
                     and z.testzip() is None, "Exactly nine ZIP-root outputs required")
    # This existing branch is local: no Backend, owner, endpoint or allocation.
    core.Operator.verify(SimpleNamespace(item=core.job("ANALYSIS", "analysis"), out=out, state={}))
    verification = core.read_json(out / "verification.json")
    result = dict(verification, archive=str(archive), notebook=str(notebook), receipt=str(receipt_path),
                  verification=str(out / "verification.json"), analysis_inputs=str(staging))
    write_receipt(receipt_path, dict(schema_version=1, execution="local_cpu_papermill",
        allocation_attempted=False, owned_endpoint=None, gpu_seconds=0, pins=pins, sources=sources,
        helper_notebook_sha256=digest(helper_nb), input_receipt_sha256=digest(out / "analysis_inputs_receipt.json"),
        auditor={k: getattr(auditor, k) for k in ("AUDITOR_ID", "AUDITOR_VERSION", "AUDITOR_SHA256", "VERIFIER_SHA256")},
        result=result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "approval", "pilot-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.root, args.approval, args.pilot_report), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
