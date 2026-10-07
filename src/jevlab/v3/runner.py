"""Protocol-3 inference without importing gold or changing legacy runtime globals.

Non-development data require a SHA256SUMS-sealed dataset_manifest.json with dataset_version="v3",
sealed=true, actual_n={test:500,pilot:50,dev:20}, input_sha256 keyed by input
filenames and schedule.json, and review={agent_reviewed_all:true,
human_reviewed:false,policy_id:<nonempty>,policy_sha256:<64 hex>}. Optional
config.dataset_review_policy={id,sha256} binds that review policy explicitly.
Development alone can use the copied twenty inputs before the dataset is sealed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import math
import os
import re
import shutil
import time
import traceback
import zipfile
import uuid
from collections import Counter
from pathlib import Path

from ..common import (append_jsonl, config_hash, load_prompt, read_json, read_jsonl,
                      sha256_file, sha256_text, utc_now, write_json)
from ..logutil import EventLog, setup_logging
from ..runner import Experiment as LegacyExperiment, shutdown_engine_checked
from . import __version__
from .operational import build_calendar, validate_calendar, validate_code_transition

log = logging.getLogger("jevlab.v3.runner")

STOCHASTIC = ("G_SINGLE", "JFINAL", "B13")
GREEDY = ("G_GREEDY", "B13_GREEDY", "Q9_GREEDY")
CONDITIONS = STOCHASTIC + GREEDY
HYBRID = ("JFINAL",)
SINGLE_G = ("G_SINGLE", "G_GREEDY")
BASELINE = ("B13", "B13_GREEDY")
DEFAULTS = dict(
    CONDITION="JFINAL", RUN_TAG="main", SPLIT="test", N_PROBLEMS=500, SEEDS="17,29,43",
    TEMPERATURE=0.7, TOP_P=0.9, TOP_K=0, REPETITION_PENALTY=1.0,
    PRESENCE_PENALTY=0.0, FREQUENCY_PENALTY=0.0, ENABLE_THINKING=False, N_CANDIDATES=4,
    MAX_OUTPUT_TOKENS=2048, CANDIDATE_BUDGET=8192, GB_CONTEXT=4096, J_MAX_INPUT=16384,
    MAX_PROMPT_TOKENS=1024, TIMEOUT_S=600, ENFORCE_EAGER=False,
    KV_CACHE_GB_G=3.0, KV_CACHE_GB_J=4.0, KV_CACHE_GB_B=3.0, KV_CACHE_GB_O=3.0,
    GPU_MEMORY_UTILIZATION=0.05, J_READOUT="logprob_token_ids", PREFLIGHT_MODE="full",
    ABORT_ON_PREFLIGHT_FAIL=True, WARMUP_N=3, CHECKPOINT_EVERY=5, NVML_HZ=10.0,
    HASH_WEIGHTS=True, REQUIRED_GPU="A100", PERSIST_TO_DRIVE=False,
    DRIVE_DIR="/content/drive/MyDrive/jev_llm_results/v3", LOG_LEVEL="INFO",
    ROOT="/content/jev_llm_v3", HF_CACHE="/content/hf_cache",
    CONFIG_FILE="config/experiment_v3.json", DATA_SUBDIR="data/v3",
    LAT_N_ITEMS=100, LAT_REPETITIONS=3, LAT_BLOCK_SIZE=10,
    LAT_CONDITIONS="JFINAL,B13_GREEDY", LATENCY_SWAP_MODE="reload",
    ENGINE_CLOSE_TIMEOUT_S=60.0,
    APPROVAL_FILE="", PILOT_REPORT_FILE="", CONFIRMATORY_AUTHORIZED=False,
)
NON_SEMANTIC = {"N_PROBLEMS", "PREFLIGHT_MODE", "ABORT_ON_PREFLIGHT_FAIL", "WARMUP_N",
                "CHECKPOINT_EVERY", "NVML_HZ", "PERSIST_TO_DRIVE", "DRIVE_DIR", "LOG_LEVEL",
                "ROOT", "HF_CACHE", "REQUIRED_GPU", "ENGINE_CLOSE_TIMEOUT_S",
                "APPROVAL_FILE", "PILOT_REPORT_FILE", "CONFIRMATORY_AUTHORIZED"}
OFFICIAL_Q9_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def check_budget_approval(exp):
    """Runtime consent gate, matching operator field names without importing tools.

    GPU-hour reservations/deadlines remain operator-owned. This gate verifies
    explicit consent and the config/data bindings of the supplied pilot evidence.
    """
    if exp.p["SPLIT"] != "test":
        return {"required": False, "scope": exp.p["SPLIT"]}
    if (exp.p.get("CONFIRMATORY_AUTHORIZED") is not True or not exp.p.get("APPROVAL_FILE")
            or not exp.p.get("PILOT_REPORT_FILE")):
        raise RuntimeError("test requires explicit APPROVAL_FILE, PILOT_REPORT_FILE and CONFIRMATORY_AUTHORIZED=True")
    approval_path, report_path = Path(exp.p["APPROVAL_FILE"]), Path(exp.p["PILOT_REPORT_FILE"])
    approval_path = approval_path if approval_path.is_absolute() else exp.root / approval_path
    report_path = report_path if report_path.is_absolute() else exp.root / report_path
    if any("gold" in path.name.lower() for path in (approval_path, report_path)):
        raise ValueError("approval/pilot evidence must not be gold files")
    approval, report = read_json(approval_path), read_json(report_path)
    report_sha = sha256_file(report_path)
    scope = "latency" if exp.cond == "LATENCY" else "test"
    hours = approval.get("max_gpu_hours")
    if (approval.get("approved") is not True or approval.get("scope") not in (scope, "test_and_latency")
            or approval.get("config_sha256") != exp.config["experiment_config_sha256"]
            or approval.get("pilot_report_sha256") != report_sha
            or not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip()
            or type(hours) not in (int, float) or not math.isfinite(hours) or hours <= 0):
        raise ValueError("invalid config/pilot-bound budget approval sentinel")
    approved_at = dt.datetime.fromisoformat(str(approval.get("approved_utc", "")).replace("Z", "+00:00"))
    created_at = dt.datetime.fromisoformat(exp.manifest["created_utc"])
    if approved_at.tzinfo is None or approved_at > created_at or approved_at > dt.datetime.now(dt.timezone.utc):
        raise ValueError("approval must predate experiment construction and execution")
    entries, aggregate = report.get("conditions", {}), report.get("aggregate", {})
    if (report.get("protocol_version") != "3" or report.get("code_version") != __version__
            or report.get("go") is not True or report.get("decision") != "go" or report.get("errors") != []
            or report.get("budget_approval") is not False or set(entries) != set(CONDITIONS)
            or aggregate.get("complete") is not True or aggregate.get("denominator") != 600
            or aggregate.get("observed_cases") != 600 or not 0 <= aggregate.get("timeout_infra_rate", 1) < .05):
        raise ValueError("complete technical pilot GO required; quality-win rules do not authorize budget")
    artifacts, tags = {}, set()
    for condition, entry in entries.items():
        count = 150 if condition in STOCHASTIC else 50
        verification, rates = entry.get("verification", {}), entry.get("rates", {})
        if (entry.get("verified") is not True or entry.get("denominator") != count
                or verification.get("ok") is not True or verification.get("records") != count
                or verification.get("condition") != condition or rates.get("denominator") != count
                or not 0 <= rates.get("timeout_infra_rate", 1) < .05):
            raise ValueError(f"incomplete/failed technical pilot arm: {condition}")
        name = str(verification.get("archive", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if not name.endswith("_final.zip") or "gold" in name.lower():
            raise ValueError("pilot evidence requires condition-bound final archives")
        recorded = verification["archive"]
        drive = re.fullmatch(r"/mnt/([a-z])/(.+)", recorded) if os.name == "nt" else None
        if drive:
            recorded = drive[1].upper() + ":/" + drive[2]
        candidates = [exp.root / "incoming" / name, exp.root / "results" / "v3" / name,
                      Path(recorded)]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None or sha256_file(path) != verification.get("archive_sha256"):
            raise ValueError(f"missing/changed pilot archive: {condition}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            manifests = [n for n in names if n == "manifest.json" or n.endswith("/manifest.json")]
            if len(names) != len(set(names)) or len(manifests) != 1 or any("gold" in n.lower() for n in names):
                raise ValueError("pilot archive must contain one public manifest and no gold")
            manifest = json.loads(archive.read(manifests[0]))
        cfg = manifest.get("config", {})
        policy = validate_code_transition(cfg.get("code_sha256"), exp.config["code_sha256"],
                                          approval, exp.config["experiment_config_sha256"], report_sha)
        digest = config_hash(cfg)
        params = cfg.get("params", {})
        coverage = manifest.get("expected_coverage", {})
        if (digest != manifest.get("config_hash") or digest != verification.get("config_hash")
                or manifest.get("condition") != condition or params.get("CONDITION") != condition
                or params.get("SPLIT") != "pilot" or coverage.get("n_problems") != 50
                or coverage.get("n_predictions") != count
                or params.get("SEEDS") != ("17,29,43" if condition in STOCHASTIC else "17")
                or cfg.get("protocol_version") != "3" or cfg.get("code_version") != __version__
                or cfg.get("experiment_config_sha256") != exp.config["experiment_config_sha256"]
                or cfg.get("data_sha256") != exp.config["data_sha256"]
                or cfg.get("prompt_sha256") != exp.config["prompt_sha256"]
                or cfg.get("models") != exp.config["models"] or cfg.get("profile") != exp.config["profile"]
                or manifest.get("coverage_complete") is not True or not manifest.get("finished_utc")):
            raise ValueError(f"pilot evidence config/data/coverage binding mismatch: {condition}")
        tags.add(params.get("RUN_TAG"))
        artifacts[condition] = verification["archive_sha256"]
    if len(tags) != 1 or not next(iter(tags)):
        raise ValueError("pilot evidence must share one nonempty run tag")
    operational = {}
    if policy is not None:
        amended_at = dt.datetime.fromisoformat(policy["approved_utc"].replace("Z", "+00:00"))
        if amended_at > created_at or hours > 25:
            raise ValueError("operational approval must predate construction within 25 GPU hours")
        for field, source in (("latency_source_plan_sha256", "latency_plan.json"),
                              ("latency_schedule_sha256", "schedule.json")):
            if policy[field] != exp.config["data_sha256"].get(source):
                raise ValueError("operational sealed source SHA mismatch")
        items = read_jsonl(exp.data_dir / "latency_inputs.jsonl")
        order = read_json(exp.data_dir / "schedule.json")["order"]
        metadata = {row["item_id"]: row for row in read_json(exp.data_dir / "latency_plan.json")["items"]}
        calendar = build_calendar(items, order, metadata)
        validate_calendar(calendar, items, order, metadata, policy)
        operational = {"operational_amendment": policy,
                       "operational_policy_sha256": config_hash(policy)}
    return {"required": True, "scope": scope, "approved_by": approval["approved_by"],
            "approved_utc": approval["approved_utc"], "max_gpu_hours": hours,
            "approval_sha256": sha256_file(approval_path), "pilot_report_sha256": report_sha,
             "pilot_archive_sha256": artifacts, "gpu_hour_meter": "operator-owned", **operational}


def _input_rows(path: Path, count: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    if any(set(r) != {"id", "problem", "language"} for r in rows):
        raise ValueError("inference inputs must contain only id, problem, language (no gold)")
    if any(not all(isinstance(r[k], str) and r[k].strip() for k in r) or r["language"] != "en"
           for r in rows):
        raise ValueError("input IDs/problems must be nonempty strings and language must be en")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate input IDs")
    if len(rows) != count:
        raise ValueError(f"{path.name} requires {count} inputs, found {len(rows)}")
    return rows


class Experiment(LegacyExperiment):
    # The version-neutral stage context is inherited; runtime/publication are v3-owned.
    def __init__(self, **params):
        p = {**DEFAULTS, **params}
        self.root = Path(p["ROOT"])
        for key in ("CONFIG_FILE", "DATA_SUBDIR"):
            path = Path(p[key])
            if path.anchor or ".." in path.parts:
                raise ValueError(f"{key} must be relative to ROOT without '..'")
        self.cfg = read_json(self.root / p["CONFIG_FILE"])
        self._frozen_config_sha256 = sha256_file(self.root / p["CONFIG_FILE"])
        self._frozen_config_content_hash = config_hash(self.cfg)
        if str(self.cfg.get("protocol_version")) != "3":
            raise ValueError("v3 inference requires protocol_version 3 only")
        if self.cfg.get("code_version") != __version__:
            raise ValueError("configuration code_version must be 0.3.0 for jevlab.v3")
        runtime = self.cfg.get("runtime_defaults", {})
        p.update({k: v for k, v in runtime.items() if k not in params})
        if any(Path(p[k]).anchor or ".." in Path(p[k]).parts for k in ("CONFIG_FILE", "DATA_SUBDIR")):
            raise ValueError("CONFIG_FILE and DATA_SUBDIR must remain relative to ROOT")
        if Path(p["DATA_SUBDIR"]).as_posix() != "data/v3":
            raise ValueError("v3 inference requires isolated DATA_SUBDIR=data/v3")
        self.p, self.cond = p, p["CONDITION"]
        if self.cond not in CONDITIONS + ("LATENCY",):
            raise ValueError(f"unknown v3 condition: {self.cond}")
        if p["SPLIT"] not in ("dev", "pilot", "test"):
            raise ValueError("SPLIT must be dev, pilot or test")
        if "N_PROBLEMS" not in params:
            p["N_PROBLEMS"] = {"dev": 20, "pilot": 50, "test": 500}[p["SPLIT"]]
        if not isinstance(p["RUN_TAG"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", p["RUN_TAG"]):
            raise ValueError("RUN_TAG must be a safe nonempty filename component")
        if "smoke" in p["RUN_TAG"].lower() and p["SPLIT"] != "dev":
            raise ValueError("smoke runs must use SPLIT=dev, never test/pilot")
        if p["PREFLIGHT_MODE"] not in ("full", "light", "off"):
            raise ValueError("unknown PREFLIGHT_MODE")
        for key in ("N_PROBLEMS", "MAX_OUTPUT_TOKENS", "CANDIDATE_BUDGET", "GB_CONTEXT",
                    "J_MAX_INPUT", "MAX_PROMPT_TOKENS", "CHECKPOINT_EVERY", "TIMEOUT_S"):
            if float(p[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if int(p["N_CANDIDATES"]) != 4:
            raise ValueError("v3 requires exactly four candidates")
        if not p["HASH_WEIGHTS"]:
            raise ValueError("v3 requires full snapshot weight verification")
        if p.get("ENABLE_THINKING", False) or self.cfg.get("sampling", {}).get("enable_thinking", False):
            raise ValueError("v3 requires enable_thinking=False")
        models = self.cfg.get("models", {})
        if set(models) != {"G", "J", "B", "O"}:
            raise ValueError("v3 models must define distinct roles G, J, B and O")
        for role, model in models.items():
            if not model.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
                raise ValueError(f"model {role} requires a frozen full revision SHA")
        if models["O"]["repo"] != "Qwen/Qwen3.5-9B" or models["O"]["revision"] != OFFICIAL_Q9_REVISION:
            raise ValueError("O must be the exact pinned official Qwen3.5-9B checkpoint")
        if self.cfg.get("implementation_profile") != "1COPY-G4-VLLM-GRAPH-V3":
            raise ValueError("v3 requires implementation_profile=1COPY-G4-VLLM-GRAPH-V3")
        raw_seeds = p["SEEDS"]
        seeds = ([int(s.strip()) for s in raw_seeds.split(",") if s.strip()]
                 if isinstance(raw_seeds, str) else [int(s) for s in raw_seeds])
        if not seeds or len(set(seeds)) != len(seeds) or any(s not in (17, 29, 43) for s in seeds):
            raise ValueError("SEEDS must be distinct members of 17,29,43")
        if self.cond in GREEDY:
            if "SEEDS" in params and seeds != [17]:
                raise ValueError("greedy quality references run once with SEEDS=17")
            seeds = [17]
        self.seeds = seeds
        p["SEEDS"] = ",".join(map(str, seeds))
        self._validate_configuration()
        self.data_dir = self.root / p["DATA_SUBDIR"]
        self.dataset_manifest = None
        self._validate_dataset()
        self.prompts = {n: load_prompt(self.root / "prompts" / f"{n}.txt")
                        for n in ("generador", "criterio_paso", "criterio_final")}
        self._frozen_prompts = copy.deepcopy(self.prompts)
        self._frozen_params = copy.deepcopy({k: v for k, v in p.items() if k not in NON_SEMANTIC})
        self.config = {"params": {k: v for k, v in p.items() if k not in NON_SEMANTIC},
                       "models": models, "jevk5_runtime": self.cfg["jevk5_runtime"],
                       "prompt_sha256": {k: sha256_text(v) for k, v in self.prompts.items()},
                       "data_sha256": self._data_hashes(), "profile": self.cfg["implementation_profile"],
                        "code_version": __version__, "protocol_version": "3",
                        "code_sha256": self._code_hashes(),
                        "experiment_config_sha256": sha256_file(self.root / p["CONFIG_FILE"])}
        self.config = copy.deepcopy(self.config)
        self.chash = config_hash(self.config)
        self.run_name = f"{self.cond}_{p['RUN_TAG']}_{self.chash[:8]}"
        self.artifact_root = self.root / "results" / "v3"
        self.dir = self.artifact_root / self.cond / self.run_name
        self._published = False
        self._assert_unpublished()
        previous_manifest = read_json(self.dir / "manifest.json") if (self.dir / "manifest.json").is_file() else None
        if previous_manifest and (previous_manifest.get("config_hash") != self.chash
                                  or previous_manifest.get("config") != self.config):
            raise ValueError("partial staging identity differs from frozen configuration")
        for sub in ("", "logs", "preflight"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        setup_logging(str(self.dir / "logs"), p["LOG_LEVEL"])
        from ..engines import vllm_logging_config

        vllm_logging_config(str(self.dir / "logs"))
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.event = EventLog(str(self.dir / "events.jsonl"))
        self.f = {k: str(self.dir / f"{k}.jsonl") for k in
                  ("predictions", "proposals", "candidates", "decisions", "rounds", "metrics", "memory",
                   "failures", "warmup")}
        self.engines, self.preflight_results = {}, {}
        self.sampler = self.selector = self.ctx = None
        self.gpu_uuid = None
        self.phase_epoch = 0
        self.phase_id = None
        self.phase_cleanup_confirmed = False
        split_n = {"test": 500, "pilot": 50, "dev": 20}[p["SPLIT"]]
        full_seeds = [17] if self.cond in GREEDY else [17, 29, 43]
        full_ids = ([r["id"] for r in sorted(self.dev_problems(), key=lambda row: row["id"])]
                    if p["SPLIT"] == "dev" else read_json(self.data_dir / "schedule.json")[
                        "pilot_order" if p["SPLIT"] == "pilot" else "order"])
        self.manifest = {"run_name": self.run_name, "condition": self.cond, "config_hash": self.chash,
                         "condition_label": self.cfg.get("condition_labels", {}).get(self.cond, self.cond),
                         "config": self.config, "created_utc": utc_now(), "stages": {},
                         "dataset_manifest_sha256": self.config["data_sha256"].get("dataset_manifest.json"),
                         "expected_coverage": {"split": p["SPLIT"], "n_problems": split_n,
                                               "problem_ids": full_ids, "seeds": full_seeds,
                                               "n_predictions": split_n * len(full_seeds)},
                         "requested_coverage": {"n_problems": min(int(p["N_PROBLEMS"]), split_n),
                                                "seeds": seeds},
                           "development_only": p["SPLIT"] == "dev"}
        if previous_manifest:
            self.manifest = previous_manifest
            self.phase_epoch = int(previous_manifest.get("active_phase", {}).get("phase_epoch", 0))
            self.preflight_results = read_json(self.dir / "preflight" / "summary.json") if (
                self.dir / "preflight" / "summary.json").is_file() else {}
        self.manifest["runtime_schema_version"] = "v3-audit-1"
        self.manifest["failure_schema_version"] = "v3-context-1"
        if self.dataset_manifest:
            self.manifest["quota_axis"] = self.dataset_manifest["quota_axis"]
            self.manifest["intrinsic_difficulty_counts"] = copy.deepcopy(
                self.dataset_manifest["review"]["intrinsic_difficulty_counts"])
        self.manifest["requested_coverage"] = {"n_problems": min(int(p["N_PROBLEMS"]), split_n), "seeds": seeds}
        self.manifest["phase_schema"] = {
            "epoch": "increments only after successful postflight and warmup",
            "phase_id": "UUID per load attempt, shared by checks/warmup/cases/swap",
            "evidence": "preflight/phases/<phase_id>/<check>.json"}
        self.manifest["authorization"] = {"required": p["SPLIT"] == "test", "checked": False,
                                            "direct_api_gate": True, "gpu_hour_meter": "operator-owned"}
        self.event("init", condition=self.cond, config_hash=self.chash, params=p)
        self.save_manifest()

    @staticmethod
    def _code_hashes():
        package = Path(__file__).resolve().parents[1]
        return {str(path.relative_to(package)).replace("\\", "/"): sha256_file(path)
                for path in sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}

    def _assert_unpublished(self):
        marker = self.dir / "manifest.json"
        if (self._published or (self.artifact_root / f"{self.run_name}_final.zip").exists()
                or (marker.is_file() and read_json(marker).get("finished_utc"))):
            raise FileExistsError("finished v3 run is immutable; use a new RUN_TAG")

    def save_manifest(self):
        self._assert_unpublished()
        write_json(self.dir / "manifest.json", self.manifest)

    def _validate_identity(self, condition=None, *, context=True):
        self._validate_configuration()
        self._validate_dataset()
        if (config_hash(self.config) != self.chash or self._code_hashes() != self.config["code_sha256"]
                or self.cond != self._frozen_params["CONDITION"]
                or self.seeds != [int(seed) for seed in self._frozen_params["SEEDS"].split(",")]
                or self.prompts != self._frozen_prompts or any(
                    load_prompt(self.root / "prompts" / f"{name}.txt") != text
                    for name, text in self._frozen_prompts.items())):
            raise ValueError("frozen runtime identity changed; preserve staging and use a new RUN_TAG")
        for role, frozen in getattr(self, "_frozen_termination", {}).items():
            if self.eos[role] != frozen["eos_ids"] or sorted(self.nl[role]) != frozen["newline_ids"]:
                raise ValueError("prepared termination ID drift from frozen runtime identity")
            engine = self.engines.get(role)
            if engine and ((hasattr(engine, "eos_ids") and engine.eos_ids != frozen["eos_ids"])
                           or (hasattr(engine, "newline_ids") and sorted(engine.newline_ids) != frozen["newline_ids"])):
                raise ValueError("engine termination ID drift from frozen runtime identity")
        if context:
            cond = condition or self.cond
            role = "O" if cond == "Q9_GREEDY" else "B" if cond in BASELINE else "G"
            p = self._frozen_params
            expected = {
                "sampling": {k.lower(): p[k] for k in ("TEMPERATURE", "TOP_P", "TOP_K", "REPETITION_PENALTY",
                                                       "PRESENCE_PENALTY", "FREQUENCY_PENALTY")},
                "limits": {"n_candidates": 4, "max_output_tokens": p["MAX_OUTPUT_TOKENS"],
                           "hybrid_candidate_budget": p["CANDIDATE_BUDGET"], "gb_context": p["GB_CONTEXT"]},
                "system_prompt": self._frozen_prompts["generador"],
                "crit_step": self._frozen_prompts["criterio_paso"],
                "crit_final": self._frozen_prompts["criterio_final"], "profile": self.config["profile"]}
            if (self.ctx is None or any(getattr(self.ctx, k) != v for k, v in expected.items())
                    or self.ctx.gen is not self.engines.get(role) or self.ctx.tok is not self.toks.get(role)
                    or self.ctx.selector is not (self.selector if cond == "JFINAL" else None)):
                raise ValueError("active context drift from frozen runtime identity")
            if (self.ctx.eos_ids != self._frozen_termination[role]["eos_ids"]
                    or sorted(self.ctx.newline_ids) != self._frozen_termination[role]["newline_ids"]):
                raise ValueError("context termination ID drift from frozen runtime identity")
            if cond == "JFINAL" and hasattr(self.selector, "temperature") and (
                    self.selector.temperature != self.cfg["selector"]["calibration_temperature_expected"]
                    or self.selector.max_input != p["J_MAX_INPUT"]
                    or self.selector.engine is not self.engines.get("J")
                    or self.selector.readout != self.manifest["selector"]["readout_used"]):
                raise ValueError("selector context drift from frozen runtime identity")
            frame = self._context_frame()
            if not hasattr(self, "_context_frames"):
                self._context_frames = {}
            if cond not in self._context_frames:
                self._context_frames[cond] = copy.deepcopy(frame)
            if frame != self._context_frames[cond]:
                raise ValueError("raw context/engine extras drift from phase safety snapshot")
            self._assert_engines_usable()

    def _context_frame(self, ctx=None):
        ctx = self.ctx if ctx is None else ctx
        frame = copy.deepcopy({key: value for key, value in vars(ctx).items()
                               if key not in ("gen", "tok", "selector")})
        frame["newline_ids"] = sorted(ctx.newline_ids)
        frame["generator_engine_kwargs"] = copy.deepcopy(getattr(ctx.gen, "engine_kwargs", {}))
        frame["selector"] = {key: copy.deepcopy(getattr(ctx.selector, key)) for key in
                             ("temperature", "max_input", "readout", "letter_ids")
                             if hasattr(ctx.selector, key)} if ctx.selector is not None else None
        frame["selector_engine_kwargs"] = copy.deepcopy(getattr(getattr(ctx.selector, "engine", None), "engine_kwargs", {}))
        frame["selector_engine_kwargs"].pop("seed", None)
        return frame

    def _case_context(self):
        frame = self._context_frame()
        return {"context_frame": frame, "context_frame_id": config_hash(frame), "execution_id": str(uuid.uuid4())}

    def _assert_engines_usable(self):
        bad = [role for role, engine in self.engines.items() if getattr(engine, "usable", True) is not True]
        if bad:
            raise RuntimeError(f"unusable engines after unconfirmed request cleanup: {bad}")

    def _phase_metadata(self):
        return {"phase_epoch": getattr(self, "_pending_phase_epoch", self.phase_epoch), "phase_id": self.phase_id,
                "gpu_uuid": self.gpu_uuid, "resident_roles": list(self.engines),
                "cleanup_confirmed": self.phase_cleanup_confirmed, "outside_T_total": True}

    def _begin_phase(self):
        self.phase_id = str(uuid.uuid4())
        self._pending_phase_epoch = self.phase_epoch + 1
        path = self.dir / "logs" / "vllm.log"
        self._phase_log_start = path.stat().st_size if path.is_file() else 0
        pre_engine = {"parser_selftest", "lock", "chat_template", "tokenizer", "prompt_length",
                      "sampling_semantics", "selector_native_reference"}
        for name in list(self.preflight_results):
            if name not in pre_engine:
                del self.preflight_results[name]

    def _persist_preflight(self, name, result):
        result = {**result, **self._phase_metadata()}
        self.preflight_results[name] = result
        folder = self.dir / "preflight" / "phases" / (self.phase_id or "pre_engines")
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / f"{name}.json", result)
        write_json(self.dir / "preflight" / f"{name}.json", result)
        self._write_preflight_summary()

    def _write_preflight_summary(self):
        write_json(self.dir / "preflight" / "summary.json", self.preflight_results)
        self.manifest["preflight"] = {k: {field: v.get(field) for field in
                                          ("ok", "blocking", "summary", "skipped", "phase_epoch", "phase_id",
                                           "gpu_uuid", "resident_roles", "cleanup_confirmed")}
                                      for k, v in self.preflight_results.items()}
        self.save_manifest()

    def _engine_failure(self, row):
        append_jsonl(self.f["failures"], {"ts": utc_now(), "config_hash": self.chash,
                                        **self._phase_metadata(), **getattr(self, "_active_case", {}), **row})

    def _validate_configuration(self):
        if (sha256_file(self.root / self.p["CONFIG_FILE"]) != self._frozen_config_sha256
                or config_hash(self.cfg) != self._frozen_config_content_hash):
            raise ValueError("frozen experiment configuration changed after construction")
        n, split = self.p["N_PROBLEMS"], self.p["SPLIT"]
        if type(n) is not int or n <= 0 or (split == "dev" and n > 20):
            raise ValueError("N_PROBLEMS must be a positive integer; dev has at most 20 inputs")
        if split != "dev" and n != {"pilot": 50, "test": 500}[split]:
            raise ValueError("non-dev N_PROBLEMS must be exactly pilot=50 or test=500; partial smoke is dev-only")
        expected_seeds = [17] if self.cond in GREEDY else [17, 29, 43]
        if split != "dev" and (self.seeds != expected_seeds or self.p["SEEDS"] != ",".join(map(str, expected_seeds))):
            raise ValueError("non-dev seed schedule must exactly match the frozen physical condition")
        if hasattr(self, "_frozen_params") and {k: v for k, v in self.p.items() if k not in NON_SEMANTIC} != self._frozen_params:
            raise ValueError("frozen protocol parameter drift after construction")
        if split == "dev":
            return
        aliases = {"hybrid_candidate_budget": "CANDIDATE_BUDGET", "n_candidates": "N_CANDIDATES"}
        expected = {k: DEFAULTS[k] for k in (
            "MAX_OUTPUT_TOKENS", "CANDIDATE_BUDGET", "MAX_PROMPT_TOKENS", "GB_CONTEXT", "J_MAX_INPUT",
            "TIMEOUT_S", "N_CANDIDATES", "TEMPERATURE", "TOP_P", "TOP_K", "REPETITION_PENALTY",
            "PRESENCE_PENALTY", "FREQUENCY_PENALTY", "ENABLE_THINKING", "ENFORCE_EAGER",
            "KV_CACHE_GB_G", "KV_CACHE_GB_J", "KV_CACHE_GB_B", "KV_CACHE_GB_O", "GPU_MEMORY_UTILIZATION")}
        for section in ("sampling", "limits", "runtime_defaults"):
            for key, value in self.cfg.get(section, {}).items():
                parameter = aliases.get(key, key.upper())
                if parameter in NON_SEMANTIC or parameter in ("N_PROBLEMS", "SEEDS"):
                    continue
                if self.p.get(parameter) != value:
                    raise ValueError(f"frozen protocol parameter cannot be overridden: {parameter}")
                expected[parameter] = value
        for parameter, value in expected.items():
            if self.p.get(parameter) != value:
                raise ValueError(f"frozen protocol parameter cannot be overridden: {parameter}")

    def _check_authorization(self):
        self._assert_unpublished()
        self._validate_identity(context=False)
        authorization = check_budget_approval(self)
        self.manifest["authorization"] = {**authorization, "checked": True, "direct_api_gate": True}
        self.save_manifest()

    def _data_hashes(self):
        names = ("dev_inputs.jsonl",) if self.p["SPLIT"] == "dev" else (
            "test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "schedule.json",
            "dataset_manifest.json", "latency_inputs.jsonl", "latency_plan.json", "SHA256SUMS")
        return {n: sha256_file(self.data_dir / n) for n in names if (self.data_dir / n).is_file()}

    def _validate_dataset(self):
        if hasattr(self, "config") and self._data_hashes() != self.config["data_sha256"]:
            raise ValueError("frozen inference inputs or dataset manifest changed after construction")
        if self.p["SPLIT"] == "dev":
            _input_rows(self.data_dir / "dev_inputs.jsonl", 20)
            return
        path = self.data_dir / "dataset_manifest.json"
        if not path.is_file():
            raise ValueError("missing sealed v3 dataset_manifest.json; inference is blocked")
        manifest = read_json(path)
        review = manifest.get("review", {})
        if manifest.get("dataset_version") != "v3" or manifest.get("sealed") is not True:
            raise ValueError("v3 dataset is unsealed; inference is blocked")
        if review.get("agent_reviewed_all") is not True or review.get("human_reviewed") is not False:
            raise ValueError("review gate requires agent_reviewed_all=true and human_reviewed=false")
        intrinsic = review.get("intrinsic_difficulty_counts")
        if (manifest.get("quota_axis") != "source_sampling_tier"
                or review.get("source_sampling_tier_is_intrinsic_difficulty") is not False
                or not isinstance(intrinsic, dict) or set(intrinsic) != {"test", "pilot"}
                or any(not isinstance(intrinsic[split], dict) or set(intrinsic[split]) != {"easy", "medium", "hard"}
                       or any(type(n) is not int or n < 0 for n in intrinsic[split].values())
                       or sum(intrinsic[split].values()) != count for split, count in (("test", 500), ("pilot", 50)))):
            raise ValueError("sealed dataset requires source_sampling_tier axis and actual intrinsic difficulty counts")
        if not isinstance(review.get("policy_id"), str) or not review["policy_id"].strip():
            raise ValueError("review gate requires a policy_id")
        if not re.fullmatch(r"[0-9a-f]{64}", review.get("policy_sha256", "")):
            raise ValueError("review gate requires policy_sha256")
        bound = self.cfg.get("dataset_review_policy")
        if bound and (review["policy_id"] != bound.get("id") or review["policy_sha256"] != bound.get("sha256")):
            raise ValueError("sealed dataset review policy differs from configuration")
        counts = {"test": 500, "pilot": 50, "dev": 20}
        actual_n = manifest.get("actual_n", {})
        if any(actual_n.get(k) != n for k, n in counts.items()) or (
                "latency" in actual_n and actual_n["latency"] != 100):
            raise ValueError("sealed v3 dataset requires 500 test, 50 pilot and 20 dev inputs")
        seal_path = self.data_dir / "SHA256SUMS"
        if not seal_path.is_file():
            raise ValueError("missing SHA256SUMS seal; inference is blocked")
        seals = {}
        for line in seal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if name in seals or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("malformed or duplicate SHA256SUMS seal entry")
            seals[name] = digest
        if seals.get("dataset_manifest.json") != sha256_file(path):
            raise ValueError("dataset manifest SHA256SUMS seal mismatch")
        hashes = manifest.get("input_sha256", {})
        ids = {}
        for split, count in counts.items():
            name = f"{split}_inputs.jsonl"
            rows = _input_rows(self.data_dir / name, count)
            ids[split] = {r["id"] for r in rows}
            if hashes.get(name) != sha256_file(self.data_dir / name) or seals.get(name) != hashes.get(name):
                raise ValueError(f"sealed input SHA mismatch: {name}")
        if any(ids[a] & ids[b] for a, b in (("test", "pilot"), ("test", "dev"), ("pilot", "dev"))):
            raise ValueError("dataset split IDs overlap")
        schedule_path = self.data_dir / "schedule.json"
        if (not schedule_path.is_file() or hashes.get("schedule.json") != sha256_file(schedule_path)
                or seals.get("schedule.json") != hashes.get("schedule.json")):
            raise ValueError("sealed schedule SHA mismatch or missing schedule")
        schedule = read_json(schedule_path)
        for split, name in (("test", "order"), ("pilot", "pilot_order")):
            order = schedule.get(name, [])
            if len(order) != len(ids[split]) or set(order) != ids[split]:
                raise ValueError(f"schedule does not cover full {split} inputs")
        self.dataset_manifest = manifest
        self.dataset_seals = seals

    def problems(self):
        self._validate_dataset()
        split = self.p["SPLIT"]
        rows = _input_rows(self.data_dir / f"{split}_inputs.jsonl", {"test": 500, "pilot": 50, "dev": 20}[split])
        items = {r["id"]: r for r in rows}
        order = (read_json(self.data_dir / "schedule.json")["pilot_order" if split == "pilot" else "order"]
                 if split != "dev" else sorted(items))
        return [items[i] for i in order][:int(self.p["N_PROBLEMS"])]

    def dev_problems(self):
        return _input_rows(self.data_dir / "dev_inputs.jsonl", 20)

    def roles(self):
        if self.cond == "LATENCY":
            return ["G", "J", "B"]
        if self.cond == "JFINAL":
            return ["G", "J"]
        return ["O" if self.cond == "Q9_GREEDY" else "B" if self.cond in BASELINE else "G"]

    def assert_same_gpu(self):
        from .. import hw

        gpus = hw.gpu_info()
        if len(gpus) != 1:
            raise RuntimeError("v3 requires one physical A100 40GB with a stable GPU UUID")
        gpu = gpus[0]
        gib = float(gpu.get("memory.total", 0)) / 1024
        uuid = gpu.get("uuid")
        if "a100" not in gpu.get("name", "").lower() or not 39 <= gib <= 41 or not uuid:
            raise RuntimeError("v3 requires an A100 40GB and its physical GPU UUID")
        if self.gpu_uuid is not None and uuid != self.gpu_uuid:
            raise RuntimeError(f"physical GPU UUID changed: {self.gpu_uuid} -> {uuid}")
        self.gpu_uuid = uuid
        return gpu

    def capture_environment(self):
        from .. import hw

        self._check_authorization()
        with self.stage("environment"):
            previous = self.dir / "manifest_prev_gpu.json"
            if previous.is_file():
                self.gpu_uuid = read_json(previous).get("gpu_uuid")
                if not self.gpu_uuid:
                    raise RuntimeError("resume is missing a physical GPU UUID")
            gpu = self.assert_same_gpu()
            write_json(previous, {"gpu_uuid": self.gpu_uuid, "gpu": gpu["name"]})
            self.manifest["gpu_uuid"] = self.gpu_uuid
            self.manifest["environment"] = {"gpus": [gpu], "cuda_driver_reported": hw.cuda_version_smi(),
                                            "host": hw.host_info(), "packages": hw.package_versions()}
            (self.dir / "pip_freeze.txt").write_text("\n".join(hw.pip_freeze()), encoding="utf-8")
            self.sampler = hw.NvmlSampler(self.p["NVML_HZ"]).start()
            self.manifest["hw_probe"] = hw.run_probe(str(self.dir / "preflight" / "hw_probe.json"))
            self.manifest["environment"]["cuda_available"] = bool(self.manifest["hw_probe"].get("cuda"))
            if not self.manifest["environment"]["cuda_available"]:
                raise RuntimeError("mandatory CUDA hardware probe failed")

    def prepare_models(self):
        from .. import models
        from ..prompts import load_tokenizer

        self._check_authorization()
        with self.stage("snapshots"):
            self.paths, self.toks, self.lock, self.eos, self.nl = {}, {}, {}, {}, {}
            for role in self.roles():
                spec = self.cfg["models"][role]
                path = models.snapshot(spec["repo"], spec["revision"], self.p["HF_CACHE"])
                rep = models.verify_snapshot(spec["repo"], spec["revision"], path, True)
                write_json(self.dir / "preflight" / f"lock_{role}.json", rep)
                if not rep["ok"]:
                    raise RuntimeError(f"snapshot verification failed for {role}: {rep['mismatches']}")
                cfg = read_json(Path(path) / "config.json")
                tie = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings"))
                self.lock[role] = {"repo": spec["repo"], "revision": spec["revision"], "ok": True,
                                   "mismatches": rep["mismatches"], "architectures": cfg.get("architectures"),
                                   "params": models.param_inventory(path, bool(tie))}
                self.paths[role], self.toks[role] = path, load_tokenizer(path)
                if role != "J":
                    self.eos[role] = models.eos_ids(self.toks[role])
                    self.nl[role] = models.newline_token_ids(self.toks[role])
                    cache = Path(self.p["HF_CACHE"]) / f"newline_ids_{role}_{spec['revision'][:10]}.json"
                    write_json(cache, sorted(self.nl[role]))
            self.manifest["model_lock"] = self.lock
            self.manifest["tokens"] = {r: {"eos_ids": self.eos[r], "newline_set_size": len(self.nl[r]),
                "special": models.special_ids(self.toks[r]),
                "tokenizer_sha256": sha256_file(Path(self.paths[r]) / "tokenizer.json")} for r in self.eos}
            self._frozen_termination = copy.deepcopy({r: {"eos_ids": self.eos[r], "newline_ids": sorted(self.nl[r])}
                                                     for r in self.eos})

    def _load_roles(self, roles):
        from .engines import SafeEngine
        from ..selector import Selector

        p = self.p
        for role in roles:
            if role in self.engines or (role == "B" and self.engines) or (role != "B" and "B" in self.engines):
                raise RuntimeError("refusing duplicate weights or overlapping B with G+J")
            if not self.lock.get(role, {}).get("ok") or role not in self.paths:
                raise RuntimeError(f"unverified snapshot for {role}")
            kw = dict(kv_cache_gb=p[f"KV_CACHE_GB_{role}"],
                      gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"], enforce_eager=p["ENFORCE_EAGER"],
                      sleep_mode=False, extra={"enable_prefix_caching": False})
            if role == "J":
                kw.update(max_model_len=p["J_MAX_INPUT"] + 16, max_num_seqs=1, eos_ids=[],
                          logprobs_mode="raw_logits", max_logprobs=-1, language_model_only=False)
            else:
                kw.update(max_model_len=p["GB_CONTEXT"], max_num_seqs=4 if role == "G" and
                          self.cond in ("JFINAL", "LATENCY") else 1,
                          eos_ids=self.eos[role], newline_ids=self.nl[role])
            self.engines[role] = SafeEngine(role, self.paths[role], self.toks[role],
                                           failure_callback=self._engine_failure, **kw)
            if role == "J":
                temperature = float(read_json(Path(self.paths[role]) / "jevk5_config.json")["temperature"])
                if abs(temperature - self.cfg["selector"]["calibration_temperature_expected"]) > 1e-9:
                    raise RuntimeError("selector calibration differs from pinned configuration")
                self.selector = Selector(self.engines[role], self.toks[role], temperature,
                                         p["J_MAX_INPUT"], p["J_READOUT"])
                readout = self.selector.probe()
                self.manifest["selector"] = {"temperature": temperature, "readout_used": readout,
                                              "letter_ids": self.selector.letter_ids[:4]}

    def start_engines(self):
        self._check_authorization()
        with self.stage("engines"):
            self._validate_identity(context=False)
            self.assert_same_gpu()
            self._begin_phase()
            self.phase_cleanup_confirmed = not self.engines
            self._load_roles(self.roles())
            self.ctx = self._ctx()
            self.manifest["memory_static"] = {
                "engine_load_seconds": {r: e.load_seconds for r, e in self.engines.items()},
                "engine_kwargs": {r: {k: v for k, v in e.engine_kwargs.items() if k not in ("model", "tokenizer")}
                                  for r, e in self.engines.items()},
                "nvml_used_after_engines": self.sampler.now_used() if self.sampler else None,
                "outside_T_total": True,
                "note": "Load/compile/probe unmeasured; NVML is periodically sampled, not an exact peak."}

    def _ctx(self, condition=None):
        from ..algorithms import Ctx

        cond = condition or self.cond
        role = "O" if cond == "Q9_GREEDY" else "B" if cond in BASELINE else "G"
        p = self.p
        self.eos_ids, self.newline_ids = self.eos[role], self.nl[role]
        if not hasattr(self, "_frozen_termination"):
            self._frozen_termination = {}
        if role not in self._frozen_termination:
            self._frozen_termination[role] = {"eos_ids": list(self.eos[role]), "newline_ids": sorted(self.nl[role])}
        ctx = Ctx(gen=self.engines[role], tok=self.toks[role], system_prompt=self.prompts["generador"],
                   sampling={"temperature": p["TEMPERATURE"], "top_p": p["TOP_P"], "top_k": p["TOP_K"],
                             "repetition_penalty": p["REPETITION_PENALTY"],
                             "presence_penalty": p["PRESENCE_PENALTY"],
                             "frequency_penalty": p["FREQUENCY_PENALTY"]},
                   limits={"n_candidates": 4, "max_output_tokens": p["MAX_OUTPUT_TOKENS"],
                           "hybrid_candidate_budget": p["CANDIDATE_BUDGET"], "gb_context": p["GB_CONTEXT"]},
                   eos_ids=self.eos[role], newline_ids=self.nl[role],
                   selector=self.selector if cond == "JFINAL" else None,
                   crit_step=self.prompts["criterio_paso"], crit_final=self.prompts["criterio_final"],
                    profile=self.cfg["implementation_profile"])
        if not hasattr(self, "_context_frames"):
            self._context_frames = {}
        self._context_frames[cond] = self._context_frame(ctx)
        return ctx

    def _provenance(self, condition):
        role = "O" if condition == "Q9_GREEDY" else "B" if condition in BASELINE else "G"
        sampling = dict(self.ctx.sampling)
        if condition in GREEDY:
            sampling.update(temperature=0.0, top_p=1.0, top_k=0)
        return {"protocol_version": "3", "code_version": __version__, "model": self.cfg["models"][role],
                "generator_prompt_sha256": self.config["prompt_sha256"]["generador"],
                "sampling": sampling, "limits": dict(self.ctx.limits), "profile": self.ctx.profile,
                "data_sha256": self.config["data_sha256"].get(f"{self.p['SPLIT']}_inputs.jsonl"),
                "enable_thinking": False}

    def preflight_before_engines(self):
        from . import preflight

        self._validate_identity(context=False)
        with self.stage("preflight_pre_engines"):
            res = preflight.pre_engine_checks(self, "full")
            self.preflight_results.update(res)
            self._check_blocking(res)

    def preflight_after_engines(self):
        from . import preflight

        self._validate_identity()
        with self.stage("preflight_post_engines"):
            res = preflight.post_engine_checks(self, "full")
            self.preflight_results.update(res)
            self._check_blocking(res)

    def _check_blocking(self, res):
        self.preflight_results.update(res)
        self._write_preflight_summary()
        bad = [k for k, v in res.items() if v.get("blocking", True) and
               (v.get("ok") is not True or v.get("skipped"))]
        if bad:
            raise RuntimeError(f"blocking v3 preflight failures: {bad}")

    def warmup(self):
        from .algorithms import Case, run_case

        self._validate_identity()
        self.assert_same_gpu()
        with self.stage("warmup"):
            pending_epoch = self.phase_epoch + 1
            for i, item in enumerate(self.dev_problems()[:int(self.p["WARMUP_N"])]):
                self._validate_identity()
                self.assert_same_gpu()
                out = run_case(Case(item["id"], item["problem"], 17, self.cond, generation=900 + i),
                               self.ctx, self.p["TIMEOUT_S"], provenance=self._provenance(self.cond))
                append_jsonl(self.f["warmup"], {**out.run, **self._phase_metadata(),
                                              "phase_epoch": pending_epoch})
                self._assert_engines_usable()
            self.phase_epoch = pending_epoch
            self.__dict__.pop("_pending_phase_epoch", None)
            self.manifest["active_phase"] = self._phase_metadata()

    def _key(self, pid, seed):
        return sha256_text(f"{self.chash}|{pid}|{seed}|{self.cond}|{self.cfg['implementation_profile']}")

    def _proposal_callback(self, key, extra=None):
        common = {"resume_key": key, "config_hash": self.chash, "run_name": self.run_name,
                  **{k: v for k, v in self._phase_metadata().items() if k != "outside_T_total"}, **(extra or {})}
        def persist(row):
            record = {**common, **row}
            if "candidates" in row:
                record["candidates"] = [{**common, **candidate} for candidate in row["candidates"]]
                record["no_candidates_produced"] = not row["candidates"]
                record["candidate_count_observed"] = len(row["candidates"])
            append_jsonl(self.f["proposals"], record)
            if not hasattr(self, "_persisted_proposals"):
                self._persisted_proposals = set()
            self._persisted_proposals.add(key)
        return persist

    def _persist(self, out, key, wall_start, extra=None, completion_callback=None):
        r = out.run
        common = {"resume_key": key, "config_hash": self.chash, "run_name": self.run_name,
                   **{k: v for k, v in self._phase_metadata().items() if k != "outside_T_total"}, **(extra or {})}
        common.update(no_candidates_produced=not out.candidates, candidate_count_observed=len(out.candidates),
                      proposal_expected=r["condition"] == "JFINAL",
                      proposal_persisted=key in getattr(self, "_persisted_proposals", set()),
                      candidate_trace_absence=("no_observed_engine_results" if r["rounds"] else "no_generation_budget")
                                             if not out.candidates else None,
                      processed_prompt_tokens_basis="planned_request_inputs_not_observed_compute")
        common["request_ids"] = list(dict.fromkeys(
            [candidate["request_id"] for candidate in out.candidates]
            + (r.get("engine_cleanup") or {}).get("attempted_ids", [])))
        common["request_id_scope"] = "observed_candidates_and_cleanup_attempts"
        if r["status"] in ("exception", "timeout", "engine_error", "selector_error", "selector_context_limit"):
            failure = {**common, **r, "t_start_utc": wall_start}
            if out.decisions:
                failure["decision"] = copy.deepcopy(out.decisions[-1])
            append_jsonl(self.f["failures"], failure)
        for name, records in (("candidates", out.candidates), ("decisions", out.decisions), ("rounds", out.rounds)):
            for record in records:
                append_jsonl(self.f[name], {**common, **record})
        peak = self.sampler.window_peak(r["t_start_perf"], r["t_end_perf"]) if self.sampler else None
        timing = {"t_start_utc": wall_start, "t_end_utc": (extra or {}).get("t_end_utc", utc_now()),
                  "nvml_peak_used_bytes": peak}
        append_jsonl(self.f["metrics"], {**common, **r, **timing})
        append_jsonl(self.f["memory"], {**common, "problem_id": r["problem_id"], "seed": r["seed"], **timing})
        if completion_callback:
            completion_callback({**common, **r, **timing})
        append_jsonl(self.f["predictions"], {**common, **r, **timing})

    def run(self):
        from .algorithms import Case, run_case

        self._validate_identity(context=self.ctx is not None)
        self._check_authorization()
        probs = self.problems()
        done = self._completed_keys()
        orphaned = self._orphan_keys(done)
        if orphaned:
            raise RuntimeError("uncommitted durable proposal pools found; preserve this run and use a new RUN_TAG")
        with self.stage("main_loop"):
            n = 0
            for seed in self.seeds:
                for item in probs:
                    key = self._key(item["id"], seed)
                    if key in done:
                        continue
                    self._validate_identity()
                    self.assert_same_gpu()
                    wall_start = utc_now()
                    provenance = self._provenance(self.cond)
                    extra = {**provenance, **self._case_context()}
                    self._active_case = {**extra, "resume_key": key, "problem_id": item["id"],
                                         "seed": seed, "condition": self.cond}
                    out = run_case(Case(item["id"], item["problem"], seed, self.cond), self.ctx,
                                    self.p["TIMEOUT_S"], proposal_callback=self._proposal_callback(key, extra),
                                    provenance=provenance)
                    self._persist(out, key, wall_start, extra=extra)
                    self._active_case = {}
                    done.add(key)
                    self._assert_engines_usable()
                    n += 1
                    if n % int(self.p["CHECKPOINT_EVERY"]) == 0:
                        self.checkpoint()
        self.checkpoint()

    def _close_roles(self, roles):
        self.ctx = None
        if "J" in roles:
            self.selector = None
        closed, failures = {}, []
        for role in roles:
            try:
                closed[role] = shutdown_engine_checked(self.engines[role], float(self.p["ENGINE_CLOSE_TIMEOUT_S"]))
                if not closed[role].get("cleanup_confirmed"):
                    raise RuntimeError("engine close was not confirmed")
                del self.engines[role]
                self.event("engine_cleanup", role=role, **closed[role])
            except Exception as exc:
                failures.append(f"{role}: {exc}")
        if failures:
            self.phase_cleanup_confirmed = False
            raise RuntimeError("engine cleanup failed; " + "; ".join(failures))
        self.phase_cleanup_confirmed = True
        return closed

    def _completed_keys(self):
        rows = read_jsonl(self.f["predictions"])
        counts = Counter(r["resume_key"] for r in rows)
        coverage = self.manifest["expected_coverage"]
        if any(n != 1 for n in counts.values()) or any(
                r.get("config_hash") != self.chash or r.get("run_name") != self.run_name
                or r.get("condition") not in coverage.get("conditions", [self.cond])
                or r.get("problem_id") not in coverage["problem_ids"] or r.get("seed") not in coverage["seeds"]
                or r["resume_key"] != self._prediction_key(r) for r in rows):
            raise ValueError("duplicate or identity-mismatched prediction completion markers")
        return set(counts)

    def _prediction_key(self, row):
        return self._key(row["problem_id"], row["seed"])

    def _orphan_keys(self, done):
        return {r["resume_key"] for name, path in self.f.items() if name not in ("predictions", "warmup")
                for r in read_jsonl(path) if r.get("resume_key")} - done

    def checkpoint(self, final=False):
        self._assert_unpublished()
        coverage = self.manifest["expected_coverage"]
        predictions = read_jsonl(self.f["predictions"])
        write_json(self.dir / "progress.json", {"done": len({r["resume_key"] for r in predictions}),
                   "total": coverage["n_predictions"], "updated_utc": utc_now(),
                   "condition": self.cond, "config_hash": self.chash})
        self.manifest["updated_utc"] = utc_now()
        self.manifest["preflight_full"] = "preflight/summary.json"
        if not final:
            self.save_manifest()
        path = self.artifact_root / f"{self.run_name}_{'final' if final else 'checkpoint'}.zip"
        tmp = self.artifact_root / f".{path.name}.{uuid.uuid4()}.tmp"
        with zipfile.ZipFile(tmp, "x", zipfile.ZIP_DEFLATED) as archive:
            for file in self.dir.rglob("*"):
                if file.is_file():
                    if final and file == self.dir / "manifest.json":
                        continue
                    archive.write(file, f"v3/{self.cond}/{self.run_name}/{file.relative_to(self.dir)}")
            if final:
                archive.writestr(f"v3/{self.cond}/{self.run_name}/manifest.json", json.dumps(self.manifest, indent=2))
        if final:
            # link is atomic and exclusive: readers never see a half-written final ZIP.
            os.link(tmp, path)
            self._published = True
            write_json(self.dir / "manifest.json", self.manifest)
            try:
                os.unlink(tmp)
            except OSError:
                log.warning("Published final ZIP; could not remove unique temporary archive %s", tmp)
        else:
            os.replace(tmp, path)
        if self.p["PERSIST_TO_DRIVE"]:
            dest = Path(self.p["DRIVE_DIR"])
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / path.name
            if final:
                staged = dest / f".{path.name}.{uuid.uuid4()}.tmp"
                shutil.copy2(path, staged)
                try:
                    os.link(staged, target)
                except FileExistsError:
                    if sha256_file(target) != sha256_file(path):
                        raise FileExistsError("different final bytes already published to Drive")
                finally:
                    staged.unlink()
            else:
                shutil.copy2(path, target)
        return str(path)

    def finalize(self):
        self._assert_unpublished()
        self.manifest["counts"] = {k: len(read_jsonl(v)) for k, v in self.f.items()}
        predictions = read_jsonl(self.f["predictions"])
        coverage = self.manifest["expected_coverage"]
        expected_keys = {self._key(pid, seed) for pid in coverage["problem_ids"] for seed in coverage["seeds"]}
        actual_keys = {r["resume_key"] for r in predictions}
        self._completed_keys()
        self.manifest["coverage_complete"] = actual_keys == expected_keys and len(predictions) == len(expected_keys)
        self.manifest["missing_predictions"] = len(expected_keys - actual_keys)
        return self._publish_final()

    def _publish_final(self):
        self._check_authorization()
        self._assert_engines_usable()
        if self.p["SPLIT"] != "dev" and self.manifest.get("coverage_complete") is not True:
            raise ValueError("non-development final publication requires exact complete coverage")
        done = self._completed_keys()
        for name in ("metrics", "memory", "latency_metrics"):
            if name in self.f and Counter(r["resume_key"] for r in read_jsonl(self.f[name])) != Counter({key: 1 for key in done}):
                raise ValueError(f"final {name} multiplicity differs from committed predictions")
        if self._orphan_keys(done):
            raise ValueError("cannot publish uncommitted durable case traces")
        with self.stage("final_cleanup"):
            self.manifest["final_cleanup"] = self._close_roles(list(self.engines))
        # Write the finished marker only after the final archive is atomically published.
        self.manifest["finished_utc"] = utc_now()
        try:
            return self.checkpoint(final=True)
        except Exception:
            if not self._published:
                self.manifest.pop("finished_utc", None)
            raise

    def shutdown(self):
        try:
            self._close_roles(list(self.engines))
        finally:
            if self.sampler:
                self.sampler.stop()

    def run_all(self):
        try:
            self._check_authorization()
            self._validate_dataset()
            self.capture_environment()
            self.prepare_models()
            self.preflight_before_engines()
            self.start_engines()
            self.preflight_after_engines()
            self.warmup()
            self.run()
            return self.finalize()
        except Exception:
            if self._published:
                raise
            append_jsonl(self.f["failures"], {"ts": utc_now(), "kind": "run_aborted",
                                              **getattr(self, "_active_case", {}),
                                               "traceback": traceback.format_exc()})
            self.checkpoint()
            raise


def main(default_condition="JFINAL"):
    parser = argparse.ArgumentParser(description="Isolated v3 inference; smoke runs require --dev")
    parser.add_argument("--config-file", default=DEFAULTS["CONFIG_FILE"])
    parser.add_argument("--data-subdir", default=DEFAULTS["DATA_SUBDIR"])
    parser.add_argument("--root", default=DEFAULTS["ROOT"])
    parser.add_argument("--condition", choices=CONDITIONS + ("LATENCY",), default=default_condition)
    parser.add_argument("--split", choices=("dev", "pilot", "test"), default="test")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--run-tag", default="main")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--lat-n-items", type=int, default=None)
    parser.add_argument("--lat-repetitions", type=int, default=None)
    parser.add_argument("--lat-block-size", type=int, default=None)
    parser.add_argument("--approval-file", default="")
    parser.add_argument("--pilot-report-file", default="")
    parser.add_argument("--confirmatory-authorized", action="store_true")
    args = parser.parse_args()
    if args.condition == "LATENCY":
        from .latency_study import LatencyStudy

        cls = LatencyStudy
    else:
        cls = Experiment
    split = "dev" if args.dev else args.split
    seed_params = {"SEEDS": args.seeds} if args.seeds is not None else {}
    latency_params = {}
    if args.condition == "LATENCY":
        if split == "dev":
            latency_params = {"LAT_N_ITEMS": 2, "LAT_REPETITIONS": 1}
            seed_params.setdefault("SEEDS", "17")
        latency_params.update({key: value for key, value in (
            ("LAT_N_ITEMS", args.lat_n_items), ("LAT_REPETITIONS", args.lat_repetitions),
            ("LAT_BLOCK_SIZE", args.lat_block_size)) if value is not None})
    exp = cls(CONFIG_FILE=args.config_file, DATA_SUBDIR=args.data_subdir, ROOT=args.root,
              CONDITION=args.condition, SPLIT=split,
              RUN_TAG=args.run_tag, APPROVAL_FILE=args.approval_file, PILOT_REPORT_FILE=args.pilot_report_file,
              CONFIRMATORY_AUTHORIZED=args.confirmatory_authorized,
              **({"N_PROBLEMS": args.n_problems} if args.n_problems is not None else {}),
              **seed_params, **latency_params)
    try:
        exp.run_all()
    finally:
        exp.shutdown()


if __name__ == "__main__":
    main()
