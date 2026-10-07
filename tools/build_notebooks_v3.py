"""Generate eight isolated v3 notebooks without allocating any runtime."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import INSTALL, JEVK5_COMMIT, VLLM_PIN, code, md
import make_bundle_v3 as bundles

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("G_SINGLE", "G_GREEDY", "JFINAL", "B13", "B13_GREEDY", "Q9_GREEDY")


def bundle_cell(gold=False):
    # Embed the same validator used by the operator, not a weaker extraction-only copy.
    validator = "\n".join(inspect.getsource(f) for f in
                          (bundles.sha, bundles.safe_name, bundles.input_rows, bundles.validate_config, bundles.validate_public, bundles.inspect_bundle))
    constants = f"CONFIG = {bundles.CONFIG!r}\nOFFICIAL_Q9_REVISION = {bundles.OFFICIAL_Q9_REVISION!r}\nOPERATIONAL_POLICY_ID = {bundles.OPERATIONAL_POLICY_ID!r}\nREVIEW_POLICY_ID = {bundles.REVIEW_POLICY_ID!r}\nREVIEW_CONTRACT_SHA256 = {bundles.REVIEW_CONTRACT_SHA256!r}\nPUBLIC_DATA = {bundles.PUBLIC_DATA!r}\nGOLD_DATA = {bundles.GOLD_DATA!r}\nSHARED_MODULES = {bundles.SHARED_MODULES!r}\nTOOL_SOURCES = {bundles.TOOL_SOURCES!r}\n"
    return '''import ast, hashlib, json, re, stat, zipfile, sys, os
from collections import Counter
from pathlib import Path, PurePosixPath
''' + constants + validator + f'''
ROOT = Path(ROOT)
ZIP = Path(BUNDLE_FILE)
if not ZIP.is_file():
    raise ValueError(f"Missing explicit v3 bundle: {{ZIP}}; upload it before running")
man = inspect_bundle(ZIP, check_local=False)
if man["contains_gold"] is not {gold!r}:
    raise ValueError("Wrong bundle gold visibility for this notebook")
if man["development_only"] and SPLIT != "dev":
    raise ValueError("Development bundle cannot run test/pilot")
if not {gold!r} and SPLIT == "dev" and (RUN_TAG != "smoke" or SEEDS != "17"):
    raise ValueError("Explicit dev smoke requires RUN_TAG=smoke and SEEDS=17")
marker = ROOT / "BUNDLE_MANIFEST.json"
if ROOT.is_symlink():
    raise ValueError("Symlink ROOT forbidden")
if ROOT.exists() and any(ROOT.iterdir()):
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != man:
        raise ValueError("Stale/foreign root: use an empty isolated v3 ROOT, never legacy fallback")
    for name, digest in man["files"].items():
        path = ROOT / name
        if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
                or any(p.is_symlink() for p in path.parents) or not path.is_file() or sha(path.read_bytes()) != digest):
            raise ValueError(f"Stale root hash mismatch: {{name}}")
    for directory in ("src", "data", "config", "prompts", "tools"):
        for path in (ROOT / directory).rglob("*"):
            name = path.relative_to(ROOT).as_posix()
            if path.is_symlink() or (path.is_file() and "__pycache__" not in path.parts and name not in man["files"]):
                raise ValueError(f"Extra/stale root member: {{name}}")
else:
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        for name in [*man["files"], "BUNDLE_MANIFEST.json"]:
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
sys.path.insert(0, str(ROOT / "src"))
import jevlab, jevlab.v3
if jevlab.__version__ != "0.2.1" or jevlab.v3.__version__ != "0.3.0":
    raise ValueError("Wrong parent/v3 runtime versions")
for module in (jevlab, jevlab.v3):
    if not Path(module.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
        raise ValueError("Previously imported foreign runtime; restart kernel")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
'''


def parameters(condition):
    seeds = "17" if condition.endswith("GREEDY") else "17,29,43"
    return f'''CONDITION = "{condition}"
CONFIG_FILE = "config/experiment_v3.json"  # @param {{type:"string"}}
DATA_SUBDIR = "data/v3"  # @param {{type:"string"}}
ROOT = "/content/jev_llm_v3"  # @param {{type:"string"}}
BUNDLE_FILE = "/content/jev_llm_v3_bundle.zip"  # @param {{type:"string"}}
RUN_TAG = "main"  # @param {{type:"string"}}
SPLIT = "test"  # @param ["test", "pilot", "dev"]
N_PROBLEMS = 500  # @param {{type:"integer"}}
SEEDS = "{seeds}"  # @param {{type:"string"}}
APPROVAL_FILE = ""  # @param {{type:"string"}}
PILOT_REPORT_FILE = ""  # @param {{type:"string"}}
CONFIRMATORY_AUTHORIZED = False  # @param {{type:"boolean"}}
PREFLIGHT_MODE = "full"
ABORT_ON_PREFLIGHT_FAIL = True
HASH_WEIGHTS = True
REQUIRED_GPU = "A100"
WARMUP_N = 3  # @param {{type:"integer"}}
CHECKPOINT_EVERY = 5  # @param {{type:"integer"}}
''' + ('''LAT_N_ITEMS = 100  # @param {type:"integer"}
LAT_REPETITIONS = 3  # @param {type:"integer"}
LAT_BLOCK_SIZE = 10  # @param {type:"integer"}
LAT_CONDITIONS = "JFINAL,B13_GREEDY"
LATENCY_SWAP_MODE = "reload"
''' if condition == "LATENCY" else "")


APPROVAL = '''if CONFIG_FILE != "config/experiment_v3.json" or DATA_SUBDIR != "data/v3":
    raise ValueError("Only isolated v3 config/data are accepted")
if SPLIT == "test":
    if CONFIRMATORY_AUTHORIZED is not True or not APPROVAL_FILE or not PILOT_REPORT_FILE:
        raise ValueError("Explicit authorization, user budget sentinel and actual pilot GO report are required")
    approval = json.loads(Path(APPROVAL_FILE).read_text(encoding="utf-8"))
    pilot_raw = Path(PILOT_REPORT_FILE).read_bytes()
    pilot = json.loads(pilot_raw)
    import math
    from datetime import datetime, timezone
    scopes = {"test_and_latency"} if CONDITION == "ANALYSIS" else {"latency", "test_and_latency"} if CONDITION == "LATENCY" else {"test", "test_and_latency"}
    if (approval.get("approved") is not True or approval.get("scope") not in scopes
            or approval.get("config_sha256") != man["config_sha256"]
            or approval.get("pilot_report_sha256") != hashlib.sha256(pilot_raw).hexdigest()
            or not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip() or not approval.get("approved_utc")
            or type(approval.get("max_gpu_hours")) not in (int, float)
            or not math.isfinite(approval["max_gpu_hours"]) or not 0 < approval["max_gpu_hours"] <= 25):
        raise ValueError("Invalid budget approval sentinel")
    arms = {"G_SINGLE", "G_GREEDY", "JFINAL", "B13", "B13_GREEDY", "Q9_GREEDY"}
    entries = pilot.get("conditions", {})
    aggregate = pilot.get("aggregate", {})
    if (pilot.get("protocol_version") != "3" or pilot.get("code_version") != "0.3.0"
            or pilot.get("go") is not True or pilot.get("decision") != "go" or pilot.get("errors") != []
            or pilot.get("budget_approval") is not False or set(entries) != arms
            or aggregate.get("complete") is not True or aggregate.get("denominator") != 600
            or aggregate.get("observed_cases") != 600 or not 0 <= aggregate.get("timeout_infra_rate", 1) < .05):
        raise ValueError("Actual complete technical pilot GO required, not a quality-win rule")
    for arm, entry in entries.items():
        count = 150 if arm in {"G_SINGLE", "JFINAL", "B13"} else 50
        if (entry.get("verified") is not True or entry.get("denominator") != count
                or entry.get("verification", {}).get("ok") is not True
                or entry.get("verification", {}).get("condition") != arm
                or entry.get("verification", {}).get("records") != count
                or entry.get("rates", {}).get("denominator") != count
                or not 0 <= entry["rates"].get("timeout_infra_rate", 1) < .05):
            raise ValueError("Incomplete/failed technical pilot arm")
    approved_at = datetime.fromisoformat(approval["approved_utc"].replace("Z", "+00:00"))
    if approved_at.tzinfo is None or approved_at > datetime.now(timezone.utc):
        raise ValueError("Approval must precede execution")
'''


def condition_notebook(condition):
    if condition not in CONDITIONS + ("LATENCY",):
        raise ValueError(f"Unknown v3 condition: {condition}")
    latency = condition == "LATENCY"
    smoke = ("SPLIT=dev, RUN_TAG=smoke, SEEDS=17, LAT_N_ITEMS=2, LAT_REPETITIONS=1 (4 actual executions)."
             if latency else "SPLIT=dev, RUN_TAG=smoke, N_PROBLEMS=1, SEEDS=17.")
    source = parameters(condition)
    names = [n.targets[0].id for n in ast.parse(source).body if isinstance(n, ast.Assign)]
    names = [n for n in names if n != "BUNDLE_FILE"]
    module, cls = ("latency_study", "LatencyStudy") if latency else ("runner", "Experiment")
    checks = ('''if SPLIT == "dev" and (LAT_N_ITEMS != 2 or LAT_REPETITIONS != 1):
    raise ValueError("Latency smoke requires exactly 2 items x 1 repetition x seed17 x 2 arms")
''' if latency else '''if SPLIT == "dev" and N_PROBLEMS != 1:
    raise ValueError("Quality smoke requires N_PROBLEMS=1")
if SPLIT == "test" and N_PROBLEMS != 500:
    raise ValueError("Full quality requires N_PROBLEMS=500")
if SPLIT != "dev" and SEEDS != ("17" if CONDITION.endswith("GREEDY") else "17,29,43"):
    raise ValueError("Full quality seed schedule must match the physical condition")
''')
    nb = nbf.v4.new_notebook(cells=[
        md(f"# JEV-LLM v3: {condition}\n\nA100 40GB; isolated v3 runtime 0.3.0, shared parent 0.2.1. "
           + ("100 test items x 3 repetitions x 3 seeds x 2 arms = 1800; same GPU, reload outside timer. "
              + "Per reviewed domain, 6/8/6 balances ORIGINAL source sampling tiers, not intrinsic difficulty; both labels remain separate. "
              if latency else "Full test: 500 problems; sampled arms seeds17,29,43, greedy arms seed17 only. ")
           + f"\n\nExplicit development smoke: {smoke} Use jev_llm_v3_dev_bundle.zip. No full data/seal is implied. "
           + "\n\nUpload the explicit bundle before running; no legacy root or bundle fallback. "
           + "Confirmatory execution requires APPROVAL_FILE supplied separately by the operator. "
           + "All GPU phases, including smoke/pilot, share the operator's fixed 25 A100-hour accumulated ledger; "
           + "the parent supplies initial authorization and post-pilot confirmation, never this notebook builder. "
           + f"Papermill output must be condition-bound: {condition}.out.main.ipynb "
           + "(or .out.smoke.ipynb). Final archive: {CONDITION}_{RUN_TAG}_{config_hash[:8]}_final.zip. "
           + "No VM is allocated by this builder."),
        code(INSTALL, "Install pinned stack"),
        code(source, "Parameters", tags=["parameters"]),
        code(bundle_cell(), "Strict v3 bundle loader"),
        code(APPROVAL + checks, "Validate run and approval"),
        code(f"from jevlab.v3.{module} import {cls}\n"
             + f"exp = {cls}(**dict(" + ", ".join(f"{n}={n}" for n in names) + "))\nprint(exp.dir)", "Create v3 runtime"),
        code('''try:
    exp.capture_environment()
    exp.prepare_models()
    exp.preflight_before_engines()
    exp.start_engines()
    exp.preflight_after_engines()
    exp.warmup()
    exp.run()
    final_archive = exp.finalize()
    print("Final archive:", final_archive)
finally:
    exp.shutdown()
''', "Run and release GPU"),
    ])
    nb.metadata.update(kernelspec={"name": "python3", "display_name": "Python 3", "language": "python"},
                       accelerator="GPU", colab={"provenance": [], "gpuType": "A100"})
    return nb


def analysis_notebook():
    nb = nbf.v4.new_notebook(cells=[
        md("# JEV-LLM v3: Offline Analysis (CPU)\n\nExplicit gold-only offline analysis of complete test runs, "
           "not development smoke or pilot. Put immutable final archives and condition-bound executed "
           "*.out.*.ipynb in ROOT/analysis_incoming (or ROOT/results/v3). "
           "ROOT/incoming is reserved for the six bound pilot ZIPs. Never extract untrusted run ZIPs. "
           "The inference/public dataset seal is identical in the analysis bundle. "
           "A separately supplied APPROVAL_FILE must cover test_and_latency. No GPU required."),
        code('''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "tabulate", "papermill"])
''', "CPU dependencies"),
        code('''CONFIG_FILE = "config/experiment_v3.json"  # @param {type:"string"}
DATA_SUBDIR = "data/v3"  # @param {type:"string"}
ROOT = "/content/jev_llm_v3"  # @param {type:"string"}
BUNDLE_FILE = "/content/jev_llm_v3_analysis_bundle.zip"  # @param {type:"string"}
RUN_TAG = "main"  # @param {type:"string"}
SPLIT = "test"
APPROVAL_FILE = ""  # @param {type:"string"}
PILOT_REPORT_FILE = ""  # @param {type:"string"}
CONFIRMATORY_AUTHORIZED = False  # @param {type:"boolean"}
N_BOOT = 10000
''', "Parameters", tags=["parameters"]),
        code(bundle_cell(gold=True), "Strict offline analysis bundle"),
         code('CONDITION = "ANALYSIS"\n' + APPROVAL + '''if not APPROVAL_FILE:
    raise ValueError("APPROVAL_FILE sentinel is required for offline v3 analysis")
if CONFIG_FILE != "config/experiment_v3.json" or DATA_SUBDIR != "data/v3":
    raise ValueError("Only isolated v3 config/data are accepted")
from jevlab.v3.analysis import run_analysis as run_analysis_v3
import shutil
incoming = ROOT / "analysis_incoming"
results = ROOT / "results/v3"
results.mkdir(parents=True, exist_ok=True)
for path in sorted(incoming.glob("*")):
    if path.is_symlink() or not path.is_file() or not (path.name.endswith("_final.zip") or ".out." in path.name and path.suffix == ".ipynb"):
        raise ValueError("Unexpected offline analysis evidence")
    target = results / path.name
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise ValueError("Conflicting immutable analysis evidence")
    if not target.exists():
        shutil.copyfile(path, target)
A = run_analysis_v3(ROOT / "results/v3", ROOT / DATA_SUBDIR / "test_gold.jsonl",
                    ROOT / ("analysis_v3_" + RUN_TAG), RUN_TAG,
                    config=ROOT / CONFIG_FILE, prompts=ROOT / "prompts",
                    approval=Path(APPROVAL_FILE), inputs=ROOT / DATA_SUBDIR / "test_inputs.jsonl",
                    n_boot=N_BOOT)
print(A["report"])
archive = shutil.make_archive(str(ROOT / ("analysis_" + RUN_TAG)), "zip", ROOT / ("analysis_v3_" + RUN_TAG))
print("Analysis archive:", archive)
''', "Explicit offline analysis"),
    ])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def build(root=ROOT, output_dir=None):
    out = Path(output_dir) if output_dir is not None else Path(root) / "notebooks/v3"
    out.mkdir(parents=True, exist_ok=True)
    notebooks = [(f"{i:02d}_{c}.ipynb", condition_notebook(c)) for i, c in enumerate(CONDITIONS, 1)]
    notebooks += [("07_ANALYSIS.ipynb", analysis_notebook()), ("08_LATENCY.ipynb", condition_notebook("LATENCY"))]
    frozen = output_dir is None and any((Path(root) / "dist/v3" / n).exists() for n in
                                       ("jev_llm_v3_bundle.zip", "jev_llm_v3_analysis_bundle.zip"))
    for name, nb in notebooks:
        for i, cell in enumerate(nb.cells):
            cell.id = f"v3-cell-{i:02d}"
            if cell.cell_type == "code":
                ast.parse(cell.source)
        nbf.validate(nb)
        if frozen and (not (out / name).is_file() or (out / name).read_text(encoding="utf-8") != nbf.writes(nb)):
            raise ValueError("Refusing notebook source changes after full bundle seal")
    for name, nb in notebooks:
        nbf.write(nb, out / name)
    return [out / name for name, _ in notebooks]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    for path in build(output_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
