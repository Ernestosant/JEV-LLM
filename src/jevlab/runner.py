"""Per-condition experiment runner used by notebooks 01-06.

Stages (each logged, each leaving artifacts):
  setup -> snapshots + verification -> tokenizers -> [native J reference] -> engines ->
  preflight -> warm-up -> main loop (resumable) -> manifest + zips.
Predictions never contain gold; grading happens offline in notebook 07.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path

from .common import (append_jsonl, config_hash, load_prompt, read_json, read_jsonl, sha256_file,
                     sha256_text, utc_now, write_json)
from .logutil import EventLog, setup_logging

log = logging.getLogger("jevlab.runner")

HYBRID = ("J64", "JSTEP", "JFINAL")
SINGLE_G = ("G_SINGLE",)
BASELINE = ("B13", "B13_GREEDY")
GPU_FOR = {"G_SINGLE": "L4|A100", "B13": "A100", "B13_GREEDY": "A100", "J64": "A100",
           "JSTEP": "A100", "JFINAL": "A100", "LATENCY": "A100"}

DEFAULTS = dict(
    CONDITION="J64", RUN_TAG="main", SPLIT="test", N_PROBLEMS=100, SEEDS="17",
    TEMPERATURE=0.7, TOP_P=0.9, TOP_K=0, REPETITION_PENALTY=1.0, N_CANDIDATES=4,
    MAX_OUTPUT_TOKENS=1024, CANDIDATE_BUDGET=4096, J64_BLOCK=64, JSTEP_STEP=128,
    JSTEP_MAX_ROUNDS=64, GB_CONTEXT=4096, J_MAX_INPUT=16384, MAX_PROMPT_TOKENS=512,
    TIMEOUT_S=600, ENFORCE_EAGER=False, KV_CACHE_GB_G=2.0, KV_CACHE_GB_J=4.0,
    KV_CACHE_GB_B=3.0, GPU_MEMORY_UTILIZATION=0.05, J_READOUT="logprob_token_ids",
    PREFLIGHT_MODE="full", ABORT_ON_PREFLIGHT_FAIL=True, RUN_R4_MEMORY_CHECK=False,
    WARMUP_N=3, CHECKPOINT_EVERY=5, NVML_HZ=10.0, HASH_WEIGHTS=True, REQUIRED_GPU="auto",
    PERSIST_TO_DRIVE=False, DRIVE_DIR="/content/drive/MyDrive/jev_llm_results",
    LOG_LEVEL="INFO", ROOT="/content/jev_llm", HF_CACHE="/content/hf_cache",
    CONFIG_FILE="config/experiment.json", DATA_SUBDIR="data",
)
# Parameters that do not change what is measured (excluded from config_hash).
NON_SEMANTIC = {"N_PROBLEMS", "PREFLIGHT_MODE", "ABORT_ON_PREFLIGHT_FAIL", "WARMUP_N",
                "CHECKPOINT_EVERY", "NVML_HZ", "HASH_WEIGHTS", "PERSIST_TO_DRIVE", "DRIVE_DIR",
                "LOG_LEVEL", "ROOT", "HF_CACHE", "REQUIRED_GPU", "RUN_R4_MEMORY_CHECK"}


class Experiment:
    def __init__(self, **params):
        p = {**DEFAULTS, **params}
        self.root = Path(p["ROOT"])
        for key in ("CONFIG_FILE", "DATA_SUBDIR"):
            path = Path(p[key])
            if path.anchor or ".." in path.parts:
                raise ValueError(f"{key} must be relative to ROOT without '..'")
        self.cfg = read_json(self.root / p["CONFIG_FILE"])
        if self.cfg.get("code_version") not in (None, _code_version()):
            raise ValueError("configuration code_version does not match installed jevlab")
        p = {**p, **{k: v for k, v in self.cfg.get("runtime_defaults", {}).items() if k not in params}}
        p["CONDITION"] = self.cfg.get("condition_aliases", {}).get(p["CONDITION"], p["CONDITION"])
        if p["SPLIT"] not in ("dev", "pilot", "test"):
            raise ValueError("SPLIT must be dev, pilot or test")
        if int(p["N_PROBLEMS"]) < 0:
            raise ValueError("N_PROBLEMS must be nonnegative")
        self.p = p
        self.cond = p["CONDITION"]
        assert self.cond in HYBRID + SINGLE_G + BASELINE + ("LATENCY",), self.cond
        self.data_dir = self.root / p["DATA_SUBDIR"]
        self.seeds = [int(s) for s in str(p["SEEDS"]).replace(" ", "").split(",") if s]
        sem = {k: v for k, v in p.items() if k not in NON_SEMANTIC}
        prompts = {n: load_prompt(self.root / "prompts" / f"{n}.txt")
                   for n in ("generador", "criterio_paso", "criterio_final")}
        self.prompts = prompts
        self.config = {"params": sem, "models": self.cfg["models"],
                       "jevk5_runtime": self.cfg["jevk5_runtime"],
                       "prompt_sha256": {k: sha256_text(v) for k, v in prompts.items()},
                       "data_sha256": self._data_hashes(), "profile": self.cfg["implementation_profile"],
                        "code_version": _code_version(),
                        "experiment_config_sha256": sha256_file(self.root / p["CONFIG_FILE"]),
                       "protocol_version": self.cfg["protocol_version"]}
        self.chash = config_hash(self.config)
        self.run_name = f"{self.cond}_{p['RUN_TAG']}_{self.chash[:8]}"
        self.dir = self.root / "results" / self.cond / self.run_name
        for sub in ("", "logs", "preflight"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        setup_logging(str(self.dir / "logs"), p["LOG_LEVEL"])
        # vLLM reads its logging config at import time (also in EngineCore subprocesses):
        # configure it before anything imports vllm so every engine logs to logs/vllm.log.
        from .engines import vllm_logging_config
        vllm_logging_config(str(self.dir / "logs"))
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.event = EventLog(str(self.dir / "events.jsonl"))
        self.f = {k: str(self.dir / f"{k}.jsonl") for k in
                  ("predictions", "candidates", "decisions", "rounds", "metrics", "memory",
                   "failures", "warmup")}
        self.engines = {}
        self.sampler = None
        self.preflight_results = {}
        self.manifest = {"run_name": self.run_name, "condition": self.cond, "config_hash": self.chash,
                          "condition_label": self.cfg.get("condition_labels", {}).get(self.cond, self.cond),
                         "config": self.config, "created_utc": utc_now(), "stages": {}}
        log.info("=" * 90)
        log.info("JEV-LLM | condition=%s | run=%s | config_hash=%s", self.cond, self.run_name, self.chash)
        log.info("results dir: %s", self.dir)
        log.info("=" * 90)
        self.event("init", condition=self.cond, config_hash=self.chash, params=p)

    # ------------------------------------------------------------------ helpers
    def _data_hashes(self) -> dict:
        d = self.root / self.p["DATA_SUBDIR"]
        out = {}
        for n in ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "schedule.json"):
            if (d / n).exists():
                out[n] = sha256_file(d / n)
        return out

    def stage(self, name: str):
        exp = self

        class _S:
            def __enter__(self_s):
                self_s.t0 = time.perf_counter()
                log.info(">>> stage %s: start", name)
                exp.event("stage_start", stage=name)
                return self_s

            def __exit__(self_s, et, ev, tb):
                dt = time.perf_counter() - self_s.t0
                ok = et is None
                exp.manifest["stages"][name] = {"seconds": round(dt, 2), "ok": ok,
                                                "error": None if ok else f"{et.__name__}: {ev}"}
                exp.event("stage_end", stage=name, seconds=dt, ok=ok,
                          error=None if ok else "".join(traceback.format_exception(et, ev, tb))[-4000:])
                (log.info if ok else log.error)("<<< stage %s: %s in %.1f s", name,
                                                "ok" if ok else "FAILED", dt)
                exp.save_manifest()
                return False
        return _S()

    def save_manifest(self):
        write_json(self.dir / "manifest.json", self.manifest)

    # ------------------------------------------------------------------ data
    def problems(self) -> list[dict]:
        split = self.p["SPLIT"]
        path = self.data_dir / f"{split}_inputs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if any(set(r) != {"id", "problem", "language"} for r in rows):
            raise ValueError("inference inputs must contain only id, problem, language (no gold)")
        items = {r["id"]: r for r in rows}
        if len(items) != len(rows):
            raise ValueError("duplicate input IDs")
        if split in ("test", "pilot"):
            order = read_json(self.data_dir / "schedule.json")["pilot_order" if split == "pilot" else "order"]
            if len(order) != len(items) or set(order) != set(items):
                raise ValueError(f"schedule does not match {split} inputs")
        else:
            order = sorted(items)
        sel = [items[i] for i in order][: int(self.p["N_PROBLEMS"])]
        return sel

    def dev_problems(self) -> list[dict]:
        path = self.data_dir / "dev_inputs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if any(set(r) != {"id", "problem", "language"} for r in rows):
            raise ValueError("dev inputs must contain only id, problem, language (no gold)")
        return rows

    # ------------------------------------------------------------------ environment
    def capture_environment(self):
        from . import hw

        with self.stage("environment"):
            gpus = hw.gpu_info()
            env = {"gpus": gpus, "cuda_driver_reported": hw.cuda_version_smi(), "host": hw.host_info(),
                   "packages": hw.package_versions()}
            (self.dir / "pip_freeze.txt").write_text("\n".join(hw.pip_freeze()))
            self.manifest["environment"] = env
            name = gpus[0]["name"] if gpus else "none"
            want = self.p["REQUIRED_GPU"]
            if want == "auto":
                want = self.cfg.get("hardware", {}).get("gpu", GPU_FOR[self.cond])
            log.info("GPU: %s | driver %s | CUDA %s | host %s (%s)", name,
                     gpus[0].get("driver_version") if gpus else "?", env["cuda_driver_reported"],
                     env["host"]["cpu_model"], env["host"]["cpu_count"])
            ok = want == "any" or any(w.lower() in name.lower() for w in want.split("|"))
            if "hardware" in self.cfg:
                lo, hi = self.cfg["hardware"]["memory_gib_range"]
                memory_gib = float(gpus[0].get("memory.total", 0)) / 1024 if gpus else 0
                ok &= "a100" in name.lower() and lo <= memory_gib <= hi
            self.manifest["gpu_requirement"] = {"required": want, "found": name, "ok": ok}
            prev = self.dir / "manifest_prev_gpu.json"
            if prev.exists():
                old = read_json(prev)
                if old.get("gpu") != name:
                    raise RuntimeError(f"resume on a different GPU ({old.get('gpu')} -> {name}); "
                                       "start a new RUN_TAG instead")
            write_json(prev, {"gpu": name})
            if not ok:
                msg = f"GPU '{name}' does not match required '{want}' for {self.cond}"
                if self.p["ABORT_ON_PREFLIGHT_FAIL"] or self.cfg.get("protocol_version") == "2":
                    raise RuntimeError(msg)
                log.warning(msg)
            self.sampler = hw.NvmlSampler(self.p["NVML_HZ"]).start()
            probe = hw.run_probe(str(self.dir / "preflight" / "hw_probe.json"))
            self.manifest["hw_probe"] = probe
            log.info("hw probe: %s", {k: (round(v, 2) if isinstance(v, float) else v)
                                      for k, v in probe.items()})

    # ------------------------------------------------------------------ models
    def roles(self) -> list[str]:
        if self.cond in HYBRID:
            return ["G", "J"]
        if self.cond in SINGLE_G:
            return ["G"]
        return ["B"]

    def prepare_models(self):
        from . import models
        from .prompts import load_tokenizer

        with self.stage("snapshots"):
            self.paths, self.toks, self.lock = {}, {}, {}
            for role in self.roles():
                m = self.cfg["models"][role]
                path = models.snapshot(m["repo"], m["revision"], self.p["HF_CACHE"])
                self.paths[role] = path
                log.info("[%s] verifying file hashes (weights=%s) ...", role, self.p["HASH_WEIGHTS"])
                rep = models.verify_snapshot(m["repo"], m["revision"], path, self.p["HASH_WEIGHTS"])
                write_json(self.dir / "preflight" / f"lock_{role}.json", rep)
                cfg = read_json(Path(path) / "config.json")
                tie = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings"))
                inv = models.param_inventory(path, bool(tie))
                self.lock[role] = {"repo": m["repo"], "revision": m["revision"], "ok": rep["ok"],
                                   "mismatches": rep["mismatches"], "architectures": cfg.get("architectures"),
                                   "params": inv}
                log.info("[%s] lock ok=%s | language params=%s (%.3fB) | vision/mtp excluded=%s",
                         role, rep["ok"], inv["language_params"], inv["language_params"] / 1e9,
                         inv["excluded_params_vision_mtp"])
                if not rep["ok"]:
                    raise RuntimeError(f"snapshot verification failed for {role}: {rep['mismatches']}")
                tok = load_tokenizer(path)
                self.toks[role] = tok
            self.manifest["model_lock"] = self.lock
            gen_role = "B" if self.cond in BASELINE else "G"
            tok = self.toks[gen_role]
            cache = Path(self.p["HF_CACHE"]) / f"newline_ids_{gen_role}_{self.cfg['models'][gen_role]['revision'][:10]}.json"
            if cache.exists():
                nl = set(json.loads(cache.read_text()))
            else:
                t0 = time.time()
                nl = models.newline_token_ids(tok)
                cache.write_text(json.dumps(sorted(nl)))
                log.info("newline token set built in %.1f s", time.time() - t0)
            self.newline_ids = nl
            self.eos_ids = models.eos_ids(tok)
            self.manifest["tokens"] = {"generator_role": gen_role, "eos_ids": self.eos_ids,
                                        "model_revision": self.cfg["models"][gen_role]["revision"],
                                        "tokenizer_sha256": sha256_file(Path(self.paths[gen_role]) / "tokenizer.json"),
                                        "newline_cache": str(cache),
                                       "special": models.special_ids(tok),
                                       "newline_set_size": len(nl),
                                       "newline_set_sha256": sha256_text(json.dumps(sorted(nl)))}
            log.info("eos ids=%s | newline token set: %d ids", self.eos_ids, len(nl))

    # ------------------------------------------------------------------ engines
    def start_engines(self):
        from .engines import Engine, vllm_logging_config
        from .selector import Selector

        p = self.p
        vllm_logging_config(str(self.dir / "logs"))
        with self.stage("engines"):
            mem0 = self.sampler.now_used() if self.sampler else None
            if "J" in self.roles():
                self.engines["J"] = Engine(
                    "J", self.paths["J"], self.toks["J"], max_model_len=p["J_MAX_INPUT"] + 16,
                    max_num_seqs=1, kv_cache_gb=p["KV_CACHE_GB_J"],
                    gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"], enforce_eager=p["ENFORCE_EAGER"],
                    eos_ids=[], logprobs_mode="raw_logits", max_logprobs=-1,
                    language_model_only=False)
            if "G" in self.roles():
                self.engines["G"] = Engine(
                    "G", self.paths["G"], self.toks["G"], max_model_len=p["GB_CONTEXT"],
                    max_num_seqs=self.cfg.get("hardware", {}).get("generator_max_num_seqs",
                        p["N_CANDIDATES"] if self.cond in HYBRID else 1),
                    kv_cache_gb=p["KV_CACHE_GB_G"], gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"],
                    enforce_eager=p["ENFORCE_EAGER"], eos_ids=self.eos_ids, newline_ids=self.newline_ids)
            if "B" in self.roles():
                self.engines["B"] = Engine(
                    "B", self.paths["B"], self.toks["B"], max_model_len=p["GB_CONTEXT"], max_num_seqs=1,
                    kv_cache_gb=p["KV_CACHE_GB_B"], gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"],
                    enforce_eager=p["ENFORCE_EAGER"], eos_ids=self.eos_ids, newline_ids=self.newline_ids)
            time.sleep(2)
            from .engines import parse_vllm_log
            self.manifest["memory_static"] = {
                "nvml_used_before_engines": mem0,
                "nvml_used_after_engines": self.sampler.now_used() if self.sampler else None,
                "processes": self.sampler.processes() if self.sampler else [],
                "vllm_log": parse_vllm_log(str(self.dir / "logs" / "vllm.log")),
                "engine_load_seconds": {k: e.load_seconds for k, e in self.engines.items()},
                "engine_kwargs": {k: {kk: vv for kk, vv in e.engine_kwargs.items()
                                      if kk not in ("model", "tokenizer")} for k, e in self.engines.items()},
                "note": "NVML samples are periodic observations, not exact peaks; KV cache is "
                        "sized explicitly with kv_cache_memory_bytes (reserved), see vllm_log."}
            log.info("static GPU memory after engines: %.2f GiB (NVML)",
                     (self.manifest["memory_static"]["nvml_used_after_engines"] or 0) / 2**30)
            if "J" in self.engines:
                jcfg = read_json(Path(self.paths["J"]) / "jevk5_config.json")
                T = float(jcfg["temperature"])
                exp_T = self.cfg["selector"]["calibration_temperature_expected"]
                if abs(T - exp_T) > 1e-9:
                    raise RuntimeError(f"J calibration temperature {T} != expected {exp_T}")
                self.selector = Selector(self.engines["J"], self.toks["J"], T, p["J_MAX_INPUT"],
                                         p["J_READOUT"])
                readout = self.selector.probe()
                self.manifest["selector"] = {"temperature": T, "letter_ids": self.selector.letter_ids[:4],
                                             "readout_requested": p["J_READOUT"], "readout_used": readout}
                log.info("selector ready: T=%.3f letters=%s readout=%s", T, self.selector.letter_ids[:4], readout)
            else:
                self.selector = None
        self.ctx = self._ctx()

    def _ctx(self):
        from .algorithms import Ctx

        p = self.p
        gen = self.engines["B"] if self.cond in BASELINE else self.engines["G"]
        gen_tok = self.toks["B"] if self.cond in BASELINE else self.toks["G"]
        return Ctx(gen=gen, tok=gen_tok, system_prompt=self.prompts["generador"],
                   sampling={"temperature": p["TEMPERATURE"], "top_p": p["TOP_P"], "top_k": p["TOP_K"],
                             "repetition_penalty": p["REPETITION_PENALTY"], "presence_penalty": 0.0,
                             "frequency_penalty": 0.0},
                   limits={"n_candidates": p["N_CANDIDATES"], "max_output_tokens": p["MAX_OUTPUT_TOKENS"],
                           "hybrid_candidate_budget": p["CANDIDATE_BUDGET"], "j64_block": p["J64_BLOCK"],
                           "jstep_step": p["JSTEP_STEP"], "jstep_max_rounds": p["JSTEP_MAX_ROUNDS"],
                           "gb_context": p["GB_CONTEXT"]},
                   eos_ids=self.eos_ids, newline_ids=self.newline_ids, selector=self.selector,
                   crit_step=self.prompts["criterio_paso"], crit_final=self.prompts["criterio_final"],
                   profile=self.cfg["implementation_profile"])

    # ------------------------------------------------------------------ preflight
    def preflight_before_engines(self):
        from . import preflight

        mode = self.p["PREFLIGHT_MODE"]
        with self.stage("preflight_pre_engines"):
            res = preflight.pre_engine_checks(self, mode)
            self.preflight_results.update(res)
            self._check_blocking(res)

    def preflight_after_engines(self):
        from . import preflight

        mode = self.p["PREFLIGHT_MODE"]
        with self.stage("preflight_post_engines"):
            res = preflight.post_engine_checks(self, mode)
            self.preflight_results.update(res)
            self._check_blocking(res)

    def _check_blocking(self, res: dict):
        write_json(self.dir / "preflight" / "summary.json", self.preflight_results)
        self.manifest["preflight"] = {k: {"ok": v.get("ok"), "blocking": v.get("blocking", False),
                                          "summary": v.get("summary")} for k, v in self.preflight_results.items()}
        bad = [k for k, v in res.items() if v.get("blocking") and not v.get("ok")]
        for k, v in res.items():
            (log.info if v.get("ok") else log.warning)("preflight %-28s ok=%s %s", k, v.get("ok"),
                                                       v.get("summary", ""))
        if bad:
            msg = f"blocking preflight failures: {bad}"
            if self.p["ABORT_ON_PREFLIGHT_FAIL"] or self.cfg.get("protocol_version") == "2":
                raise RuntimeError(msg)
            log.error(msg + " (continuing because ABORT_ON_PREFLIGHT_FAIL=False)")

    # ------------------------------------------------------------------ warm-up
    def warmup(self):
        from .algorithms import Case, run_case

        with self.stage("warmup"):
            dev = self.dev_problems()[: int(self.p["WARMUP_N"])]
            for i, it in enumerate(dev):
                out = run_case(Case(it["id"], it["problem"], 0, self.cond, generation=900 + i),
                               self.ctx, self.p["TIMEOUT_S"])
                r = out.run
                append_jsonl(self.f["warmup"], {k: v for k, v in r.items()
                                                if k not in ("raw_output", "public_output")})
                log.info("warm-up %d/%d %s status=%s T=%.2fs tokens=%s", i + 1, len(dev), it["id"],
                         r["status"], r["T_total"], r.get("candidate_tokens_total"))
            if self.selector is not None:
                t0 = time.perf_counter()
                long_txt = ("Step: compute the value carefully and check it.\n" * 300)
                self.selector.decide("Warm-up problem.", "", [long_txt[:6000], "a", "b", "c"],
                                     self.prompts["criterio_paso"], [0, 1, 2, 3])
                log.info("warm-up long selector call %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------ main loop
    def _key(self, pid: str, seed: int) -> str:
        return sha256_text(f"{self.chash}|{pid}|{seed}|{self.cond}|{self.cfg['implementation_profile']}")

    def run(self):
        from .algorithms import Case, run_case

        probs = self.problems()
        cases = [(s, it) for s in self.seeds for it in probs]
        done = {r["resume_key"] for r in read_jsonl(self.f["predictions"])}
        total = len(cases)
        log.info("main loop: %d cases (%d problems x %d seeds); %d already done (resume)",
                 total, len(probs), len(self.seeds), len([1 for s, it in cases if self._key(it["id"], s) in done]))
        t_loop = time.time()
        n_new, consecutive_infra = 0, 0
        with self.stage("main_loop"):
            for k, (seed, it) in enumerate(cases, 1):
                key = self._key(it["id"], seed)
                if key in done:
                    continue
                self._progress(k - 1, total, it["id"], t_loop, n_new)
                t_wall0 = utc_now()
                try:
                    out = run_case(Case(it["id"], it["problem"], seed, self.cond), self.ctx, self.p["TIMEOUT_S"])
                    consecutive_infra = 0
                except Exception as e:  # noqa: BLE001
                    tb = traceback.format_exc()
                    infra = any(s in (type(e).__name__ + str(e)) for s in
                                ("EngineDead", "out of memory", "CUDA error", "EngineCore", "Connection"))
                    append_jsonl(self.f["failures"], {"ts": utc_now(), "problem_id": it["id"], "seed": seed,
                                                      "condition": self.cond, "resume_key": key,
                                                      "kind": "infrastructure" if infra else "exception",
                                                      "error": f"{type(e).__name__}: {e}", "traceback": tb,
                                                      "gpu_used_bytes": self.sampler.now_used() if self.sampler else None})
                    log.error("case %s seed=%s FAILED (%s): %s", it["id"], seed,
                              "infra" if infra else "exception", e)
                    if infra:
                        consecutive_infra += 1
                        if consecutive_infra >= 2:
                            raise RuntimeError("repeated infrastructure failures; stopping (resume later)") from e
                        continue
                    rec = {"resume_key": key, "problem_id": it["id"], "seed": seed, "condition": self.cond,
                           "profile": self.cfg["implementation_profile"], "config_hash": self.chash,
                           "status": "exception", "public_output": "", "raw_output": "",
                           "error": f"{type(e).__name__}: {e}", "t_start_utc": t_wall0, "t_end_utc": utc_now()}
                    append_jsonl(self.f["predictions"], rec)
                    append_jsonl(self.f["metrics"], {k2: v for k2, v in rec.items() if k2 not in ("raw_output",)})
                    n_new += 1
                    continue
                self._persist(out, key, t_wall0)
                n_new += 1
                if n_new % int(self.p["CHECKPOINT_EVERY"]) == 0:
                    self.checkpoint()
            self._progress(total, total, None, t_loop, n_new)
        self.checkpoint()

    def _persist(self, out, key: str, t_wall0: str):
        r = out.run
        t_wall1 = utc_now()
        peak = self.sampler.window_peak(r["t_start_perf"], r["t_end_perf"]) if self.sampler else None
        common = {"resume_key": key, "config_hash": self.chash, "run_name": self.run_name}
        if "condition_labels" in self.cfg:
            common["condition_label"] = self.cfg["condition_labels"][self.cond]
        pred = {**common, "problem_id": r["problem_id"], "seed": r["seed"], "condition": r["condition"],
                "profile": r["profile"], "status": r["status"], "public_output": r["public_output"],
                "raw_output": r["raw_output"], "t_start_utc": t_wall0, "t_end_utc": t_wall1}
        metrics = {**common, **{k: v for k, v in r.items() if k not in ("raw_output", "public_output")},
                   "t_start_utc": t_wall0, "t_end_utc": t_wall1, "nvml_peak_used_bytes": peak,
                   "nvml_hz": self.p["NVML_HZ"], "rss_gib": _rss(),
                   "public_output_chars": len(r["public_output"])}
        for c in out.candidates:
            append_jsonl(self.f["candidates"], {**common, **c})
        for d in out.decisions:
            append_jsonl(self.f["decisions"], {**common, **d})
        for rd in out.rounds:
            append_jsonl(self.f["rounds"], {**common, **rd})
            log.debug("   round %s limit=%s sampled=%s accepted=%s winner=%s terminal=%s Tprop=%.2f Tsel=%.3f",
                      rd["round"], rd["limit"], rd["round_sampled"], rd["accepted_after"], rd["winner_branch"],
                      rd["winner_terminal"], rd["T_proposals"], rd["T_selector"] or 0)
        append_jsonl(self.f["metrics"], metrics)
        append_jsonl(self.f["memory"], {**common, "problem_id": r["problem_id"], "seed": r["seed"],
                                        "nvml_peak_used_bytes": peak, "nvml_now": self.sampler.now_used() if self.sampler else None})
        append_jsonl(self.f["predictions"], pred)  # last: marks the key as done
        log.info("[%s] %s seed=%s status=%-18s T=%6.2fs first=%s rounds=%s cand_tok=%s acc_tok=%s J_in=%s peak=%.2fGiB",
                 self.cond, r["problem_id"], r["seed"], r["status"], r["T_total"],
                 f"{r['T_first_accepted']:.2f}s" if r.get("T_first_accepted") else "-",
                 r.get("rounds"), r.get("candidate_tokens_total"), r.get("accepted_tokens"),
                 r.get("selector_input_tokens"), (peak or 0) / 2**30)

    def _progress(self, done: int, total: int, current, t_loop: float, n_new: int):
        el = time.time() - t_loop
        eta = (el / n_new * (total - done)) if n_new else None
        pr = {"condition": self.cond, "run_name": self.run_name, "done": done, "total": total,
              "current": current, "elapsed_s": round(el, 1), "eta_s": round(eta, 1) if eta else None,
              "updated_utc": utc_now(),
              "gpu_used_gib": round((self.sampler.now_used() or 0) / 2**30, 2) if self.sampler else None,
              "failures": len(read_jsonl(self.f["failures"]))}
        write_json(self.dir / "progress.json", pr)
        write_json(self.root / "results" / f"progress_{self.cond}.json", pr)
        if current:
            log.info("progress %d/%d (%.0f%%) elapsed=%.0fs eta=%s next=%s", done, total,
                     100 * done / max(total, 1), el, f"{eta:.0f}s" if eta else "?", current)

    # ------------------------------------------------------------------ packaging
    def checkpoint(self, final: bool = False):
        self.manifest["updated_utc"] = utc_now()
        self.manifest["preflight_full"] = "preflight/summary.json"
        self.save_manifest()
        tag = "final" if final else "checkpoint"
        zpath = self.root / "results" / f"{self.run_name}_{tag}.zip"
        tmp = str(zpath) + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in self.dir.rglob("*"):
                if f.is_file():
                    z.write(f, f"{self.cond}/{self.run_name}/{f.relative_to(self.dir)}")
        os.replace(tmp, zpath)
        log.info("%s zip written: %s (%.1f MB)", tag, zpath, zpath.stat().st_size / 1e6)
        if self.p["PERSIST_TO_DRIVE"]:
            try:
                dd = Path(self.p["DRIVE_DIR"])
                dd.mkdir(parents=True, exist_ok=True)
                shutil.copy2(zpath, dd / zpath.name)
                log.info("copied to Drive: %s", dd / zpath.name)
            except Exception as e:  # noqa: BLE001
                log.warning("Drive copy failed: %s", e)
        return str(zpath)

    def finalize(self):
        self.manifest["finished_utc"] = utc_now()
        self.manifest["counts"] = {k: len(read_jsonl(v)) for k, v in self.f.items()}
        if self.sampler:
            self.manifest["memory_final_processes"] = self.sampler.processes()
        z = self.checkpoint(final=True)
        log.info("DONE %s -> %s", self.run_name, z)
        return z

    def shutdown(self):
        failures = []
        try:
            for role, e in self.engines.items():
                try:
                    if self.p.get("LATENCY_SWAP_MODE") == "reload":
                        if e.engine is not None:
                            shutdown_engine_checked(e)
                    else:
                        e.shutdown()
                except Exception as exc:
                    failures.append(f"{role}: {exc}")
        finally:
            if self.sampler:
                self.sampler.stop()
        if failures:
            raise RuntimeError("engine cleanup failed; " + "; ".join(failures))

    # ------------------------------------------------------------------ one-call driver
    def run_all(self):
        try:
            self.capture_environment()
            self.prepare_models()
            self.preflight_before_engines()
            self.start_engines()
            self.preflight_after_engines()
            self.warmup()
            self.run()
            return self.finalize()
        except Exception:
            log.error("run aborted:\n%s", traceback.format_exc())
            append_jsonl(self.f["failures"], {"ts": utc_now(), "kind": "run_aborted",
                                              "traceback": traceback.format_exc()})
            self.checkpoint()
            raise


def _code_version() -> str:
    from . import __version__

    return __version__


def shutdown_engine_checked(engine, timeout_s: float = 60.0) -> dict:
    """Fail closed on the pinned local vLLM process contract before loading more weights."""
    core = engine.engine.engine_core
    manager = core.resources.engine_manager
    processes = list(manager.processes)
    if not processes:
        raise RuntimeError("cannot confirm engine cleanup without local process handles")
    pids = [p.pid for p in processes]
    started = time.perf_counter()
    core.shutdown(timeout=timeout_s)
    for process in processes:
        process.join(timeout=max(0.0, timeout_s - (time.perf_counter() - started)))
    if any(p.is_alive() or p.exitcode is None for p in processes):
        raise RuntimeError(f"engine processes still alive after shutdown: {pids}")
    engine.engine = None
    return {"pids": pids, "exitcodes": [p.exitcode for p in processes],
            "cleanup_confirmed": True, "seconds": time.perf_counter() - started}


def _rss():
    from .hw import rss_gib

    return rss_gib()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a pinned condition; use dev for smoke checks")
    parser.add_argument("--config-file", default="config/experiment.json")
    parser.add_argument("--data-subdir", default="data")
    parser.add_argument("--root", default="/content/jev_llm")
    parser.add_argument("--condition", default="J64")
    parser.add_argument("--split", choices=("dev", "pilot", "test"), default="test")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--run-tag", default="main")
    parser.add_argument("--seeds", default="17")
    args = parser.parse_args()
    exp = Experiment(CONFIG_FILE=args.config_file, DATA_SUBDIR=args.data_subdir, ROOT=args.root,
                     CONDITION=args.condition, SPLIT=args.split, N_PROBLEMS=args.n_problems,
                     RUN_TAG=args.run_tag, SEEDS=args.seeds)
    try:
        exp.run_all()
    finally:
        exp.shutdown()


if __name__ == "__main__":
    main()
