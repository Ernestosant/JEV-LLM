"""Notebook 08: same-GPU latency study (v3.1 confirmatory; v2 secondary/descriptive).

All arms run on the same GPU/host in counterbalanced blocks over an item set disjoint from
the test set: the 20 dev problems + synthetic prompts with controlled output-length bands.
B and G+J are swapped outside the timer (sleep for frozen v2; unload/reload amendment).
Wall-clock start/end and
perf_counter timings are persisted per run exactly as in notebooks 01-06.
"""

from __future__ import annotations

import json
import logging
import random
import time
import traceback
from pathlib import Path

from .common import append_jsonl, read_jsonl, sha256_text, utc_now, write_json
from .runner import Experiment, shutdown_engine_checked

log = logging.getLogger("jevlab.latency")

# Output-length bands (~8 tokens per line). n+1 lines always fit JSTEP's 64-round cap, so every
# arm can terminate normally (v0.1.1 used 10/40/120/300 plain integers: xlong hit max_rounds).
SYNTH_BANDS = {"short": 5, "medium": 15, "long": 30, "xlong": 60}
SYNTH_TEMPLATE = ("For each integer k from 1 to {n}, write one line of the form 'k squared is k*k' "
                  "with the value computed, for example '3 squared is 9'. Then write the line FINAL: {n}")


def build_items(dev: list[dict], n_dev: int, synth_per_band: int) -> list[dict]:
    if not 0 <= n_dev <= min(20, len(dev)) or not 0 <= synth_per_band <= 3:
        raise ValueError("latency counts require available 0-20 dev and 0-3 synthetics per band")
    items = [{"item_id": d["id"], "kind": "dev", "problem": d["problem"]} for d in dev[:n_dev]]
    for band, n in SYNTH_BANDS.items():
        for k in range(synth_per_band):
            m = n + k  # slightly different lengths inside a band
            items.append({"item_id": f"synth-{band}-{k}", "kind": f"synthetic_{band}",
                          "problem": SYNTH_TEMPLATE.format(n=m)})
    return items


class LatencyStudy(Experiment):
    CONDS_H = ["G_SINGLE", "J64", "JSTEP", "JFINAL"]
    CONDS_B = ["B13", "B13_GREEDY"]

    def __init__(self, **params):
        params = {"LAT_N_DEV": 20, "LAT_SYNTH_PER_BAND": 3, "LAT_BLOCK_SIZE": 8,
                  "LAT_CONDITIONS": ",".join(self.CONDS_H + self.CONDS_B), "SLEEP_LEVEL": 1,
                  "SLEEP_J_FOR_G_SINGLE": True, **params, "CONDITION": "LATENCY", "SPLIT": "dev"}
        if not 0 <= int(params["LAT_N_DEV"]) <= 20 or not 0 <= int(params["LAT_SYNTH_PER_BAND"]) <= 3:
            raise ValueError("LAT_N_DEV must be 0-20 and LAT_SYNTH_PER_BAND must be 0-3")
        if int(params["LAT_BLOCK_SIZE"]) < 1:
            raise ValueError("LAT_BLOCK_SIZE must be positive")
        super().__init__(**params)
        self.f.update({k: str(self.dir / f"{k}.jsonl") for k in ("latency_metrics", "swaps")})
        self.conds = [c.strip() for c in str(self.p.get("LAT_CONDITIONS", ",".join(self.CONDS_H + self.CONDS_B))).split(",") if c.strip()]
        self.conds = [self.cfg.get("condition_aliases", {}).get(c, c) for c in self.conds]
        if not self.conds or len(set(self.conds)) != len(self.conds) or any(c not in self.CONDS_H + self.CONDS_B for c in self.conds):
            raise ValueError("LAT_CONDITIONS must be unique known conditions")
        self.swap_mode = self.p.get("LATENCY_SWAP_MODE", "sleep")
        if self.swap_mode not in ("sleep", "reload"):
            raise ValueError("LATENCY_SWAP_MODE must be sleep or reload")
        self.phase, self.ctx, self.ctx_H, self.ctx_B, self.selector = None, None, None, None, None
        self._reload_epoch, self._warmed_reload = 0, set()
        if self.swap_mode == "sleep" and self.cfg.get("protocol_version") == "2" and int(self.p["SLEEP_LEVEL"]) != 1:
            raise ValueError("v2 requires SLEEP_LEVEL=1: level 2 discards weights without reloading them")
        log.info("latency study run=%s conditions=%s", self.run_name, self.conds)

    def roles(self):
        return ["G", "J", "B"]

    def prepare_models(self):
        from . import models

        super().prepare_models()
        self.nl, self.eos = {"G": self.newline_ids}, {"G": self.eos_ids}
        cache = Path(self.p["HF_CACHE"]) / f"newline_ids_B_{self.cfg['models']['B']['revision'][:10]}.json"
        if cache.exists():
            self.nl["B"] = set(json.loads(cache.read_text()))
        else:
            self.nl["B"] = models.newline_token_ids(self.toks["B"])
            cache.write_text(json.dumps(sorted(self.nl["B"])))
        self.eos["B"] = models.eos_ids(self.toks["B"])

    def start_engines(self):
        from .algorithms import Ctx
        from .engines import Engine, vllm_logging_config
        from .common import read_json
        from .selector import Selector

        p = self.p
        vllm_logging_config(str(self.dir / "logs"))
        if self.swap_mode == "reload":
            if set(self.paths) != set(self.roles()) or not all(self.lock[r]["ok"] for r in self.roles()):
                raise RuntimeError("all three snapshots must be verified before loading any latency engine")
            with self.stage("engines"):
                self._swap("H")
            return
        with self.stage("engines"):
            self.engines["B"] = Engine("B", self.paths["B"], self.toks["B"], max_model_len=p["GB_CONTEXT"],
                                       max_num_seqs=1, kv_cache_gb=p["KV_CACHE_GB_B"],
                                       gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"],
                                       enforce_eager=p["ENFORCE_EAGER"], eos_ids=self.eos["B"],
                                       newline_ids=self.nl["B"], sleep_mode=True)
            self.engines["B"].sleep(int(p.get("SLEEP_LEVEL", 1)))
            self.engines["J"] = Engine("J", self.paths["J"], self.toks["J"], max_model_len=p["J_MAX_INPUT"] + 16,
                                       max_num_seqs=1, kv_cache_gb=p["KV_CACHE_GB_J"],
                                       gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"],
                                       enforce_eager=p["ENFORCE_EAGER"], eos_ids=[], logprobs_mode="raw_logits",
                                       max_logprobs=-1, language_model_only=False, sleep_mode=True)
            self.engines["G"] = Engine("G", self.paths["G"], self.toks["G"], max_model_len=p["GB_CONTEXT"],
                                       max_num_seqs=p["N_CANDIDATES"], kv_cache_gb=p["KV_CACHE_GB_G"],
                                       gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"],
                                       enforce_eager=p["ENFORCE_EAGER"], eos_ids=self.eos["G"],
                                       newline_ids=self.nl["G"], sleep_mode=True)
            T = float(read_json(Path(self.paths["J"]) / "jevk5_config.json")["temperature"])
            if abs(T - self.cfg["selector"]["calibration_temperature_expected"]) > 1e-9:
                raise RuntimeError("selector calibration differs from pinned configuration")
            self.selector = Selector(self.engines["J"], self.toks["J"], T, p["J_MAX_INPUT"], p["J_READOUT"])
            self.selector.probe()
            common = dict(system_prompt=self.prompts["generador"],
                          sampling={"temperature": p["TEMPERATURE"], "top_p": p["TOP_P"], "top_k": p["TOP_K"],
                                    "repetition_penalty": p["REPETITION_PENALTY"], "presence_penalty": 0.0,
                                    "frequency_penalty": 0.0},
                          limits={"n_candidates": p["N_CANDIDATES"], "max_output_tokens": p["MAX_OUTPUT_TOKENS"],
                                  "hybrid_candidate_budget": p["CANDIDATE_BUDGET"], "j64_block": p["J64_BLOCK"],
                                  "jstep_step": p["JSTEP_STEP"], "jstep_max_rounds": p["JSTEP_MAX_ROUNDS"],
                                  "gb_context": p["GB_CONTEXT"]},
                          crit_step=self.prompts["criterio_paso"], crit_final=self.prompts["criterio_final"],
                          profile=self.cfg["implementation_profile"])
            self.ctx_H = Ctx(gen=self.engines["G"], tok=self.toks["G"], eos_ids=self.eos["G"],
                             newline_ids=self.nl["G"], selector=self.selector, **common)
            self.ctx_B = Ctx(gen=self.engines["B"], tok=self.toks["B"], eos_ids=self.eos["B"],
                             newline_ids=self.nl["B"], **common)
            self.phase = "H"
            self.ctx = self.ctx_H

    def _load_reload_roles(self, roles):
        from .engines import Engine
        from .common import read_json
        from .selector import Selector

        p = self.p
        for role in roles:
            if role in self.engines or (role == "B" and self.engines) or (role != "B" and "B" in self.engines):
                raise RuntimeError("reload would create duplicate weights or overlap B with G4+J")
            kw = dict(gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"], enforce_eager=p["ENFORCE_EAGER"],
                      kv_cache_gb=p[f"KV_CACHE_GB_{role}"], sleep_mode=False)
            if role == "J":
                kw.update(max_model_len=p["J_MAX_INPUT"] + 16, max_num_seqs=1, eos_ids=[],
                          logprobs_mode="raw_logits", max_logprobs=-1, language_model_only=False)
            else:
                kw.update(max_model_len=p["GB_CONTEXT"], max_num_seqs=4 if role == "G" else 1,
                          eos_ids=self.eos[role], newline_ids=self.nl[role])
            self.engines[role] = Engine(role, self.paths[role], self.toks[role], **kw)
            if role == "J":
                T = float(read_json(Path(self.paths["J"]) / "jevk5_config.json")["temperature"])
                if abs(T - self.cfg["selector"]["calibration_temperature_expected"]) > 1e-9:
                    raise RuntimeError("selector calibration differs from pinned configuration")
                self.selector = Selector(self.engines["J"], self.toks["J"], T, p["J_MAX_INPUT"], p["J_READOUT"])
                self.selector.probe()

    def _refresh_reload_contexts(self):
        from .algorithms import Ctx

        p = self.p
        common = dict(system_prompt=self.prompts["generador"],
                      sampling={"temperature": p["TEMPERATURE"], "top_p": p["TOP_P"], "top_k": p["TOP_K"],
                                "repetition_penalty": p["REPETITION_PENALTY"], "presence_penalty": 0.0,
                                "frequency_penalty": 0.0},
                      limits={"n_candidates": p["N_CANDIDATES"], "max_output_tokens": p["MAX_OUTPUT_TOKENS"],
                              "hybrid_candidate_budget": p["CANDIDATE_BUDGET"], "j64_block": p["J64_BLOCK"],
                              "jstep_step": p["JSTEP_STEP"], "jstep_max_rounds": p["JSTEP_MAX_ROUNDS"],
                              "gb_context": p["GB_CONTEXT"]},
                      crit_step=self.prompts["criterio_paso"], crit_final=self.prompts["criterio_final"],
                      profile=self.cfg["implementation_profile"])
        self.ctx_H = (Ctx(gen=self.engines["G"], tok=self.toks["G"], eos_ids=self.eos["G"],
                          newline_ids=self.nl["G"], selector=self.selector, **common) if "G" in self.engines else None)
        self.ctx_B = (Ctx(gen=self.engines["B"], tok=self.toks["B"], eos_ids=self.eos["B"],
                          newline_ids=self.nl["B"], **common) if "B" in self.engines else None)
        self.ctx = self.ctx_B if self.phase == "B" else self.ctx_H

    def _close_reload_roles(self, roles):
        import gc

        self.ctx, self.ctx_H, self.ctx_B = None, None, None
        if "J" in roles:
            self.selector = None
        closed, failures = {}, []
        for role in roles:
            try:
                closed[role] = shutdown_engine_checked(self.engines[role], float(self.p.get("ENGINE_CLOSE_TIMEOUT_S", 60)))
                self.event("engine_cleanup", role=role, **closed[role])
                del self.engines[role]
            except Exception as exc:
                failures.append(f"{role}: {type(exc).__name__}: {exc}")
                self.event("engine_cleanup", role=role, cleanup_confirmed=False, error=str(exc))
        gc.collect()
        if failures:
            raise RuntimeError("engine cleanup failed; " + "; ".join(failures))
        return closed

    def _reload(self, to: str, *, selector_only: bool = False, with_selector: bool = True):
        started, wall_start = time.perf_counter(), utc_now()
        previous = self.phase
        memory_before = self.sampler.now_used() if self.sampler else None
        closed, error, after_close = {}, None, None
        try:
            roles = list(self.engines) if not selector_only else (["J"] if not with_selector else [])
            if not selector_only:
                self.phase = None
            closed = self._close_reload_roles(roles)
            after_close = self.sampler.now_used() if self.sampler else None
            load = (["J"] if with_selector else []) if selector_only else (["B"] if to == "B" else ["J", "G"])
            self._load_reload_roles(load)
            self.phase = to
            self._reload_epoch += 1
            self._refresh_reload_contexts()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.phase = None
            raise
        finally:
            ended = time.perf_counter()
            append_jsonl(self.f["swaps"], {"ts": wall_start, "t_end_utc": utc_now(), "from": previous, "to": to,
                         "method": "reload", "selector_only": selector_only, "with_selector": with_selector,
                         "seconds": ended - started, "t_start_perf": started, "t_end_perf": ended,
                         "outside_T_total": True, "closed": closed, "ok": error is None, "error": error,
                         "nvml_before": memory_before, "nvml_after_close": after_close,
                         "nvml_used": self.sampler.now_used() if self.sampler else None,
                         "nvml_peak_used_bytes": self.sampler.window_peak(started, ended) if self.sampler else None,
                         "resident_roles": list(self.engines)})

    def _condition_context(self, condition):
        phase = "B" if condition in self.CONDS_B else "H"
        self._swap(phase)
        if self.swap_mode == "reload" and phase == "H":
            want_j = condition != "G_SINGLE" or not self.p["SLEEP_J_FOR_G_SINGLE"]
            if want_j != ("J" in self.engines):
                self._reload("H", selector_only=True, with_selector=want_j)
        return self.ctx_B if phase == "B" else self.ctx_H

    def _warm_condition(self, condition, ctx, items):
        from .algorithms import Case, run_case

        key = (condition, self._reload_epoch)
        if self.swap_mode == "reload" and key in self._warmed_reload:
            return
        for it in items[: int(self.p.get("WARMUP_N", 2))]:
            out = run_case(Case(it["item_id"], it["problem"], 0, condition, generation=900), ctx, self.p["TIMEOUT_S"])
            append_jsonl(self.f["warmup"], {"condition": condition, "item_id": it["item_id"],
                                          "T_total": out.run["T_total"], "status": out.run["status"]})
        if self.swap_mode == "reload":
            self._warmed_reload.add(key)

    def preflight_after_engines(self):
        from . import preflight

        if self.cfg.get("protocol_version") != "2":
            return super().preflight_after_engines()
        if self.swap_mode == "reload":
            self._swap("H")
            if "J" not in self.engines:
                self._reload("H", selector_only=True)
            self.cond = "JFINAL"
            old_mode = self.p["PREFLIGHT_MODE"]
            self.p["PREFLIGHT_MODE"] = "full"
            try:
                super().preflight_after_engines()
                self._swap("B")
                self.cond = "B13"
                res = {"stress_B": preflight._cached(self, "stress_B13", "full", lambda: preflight.stress_test(self))}
                self.preflight_results.update(res)
                self._check_blocking(res)
            finally:
                self.cond, self.p["PREFLIGHT_MODE"] = "LATENCY", old_mode
            self._swap("H")
            return
        self.cond = "JFINAL"
        selector = self.selector
        try:
            super().preflight_after_engines()
            self._swap("B")
            self.cond, self.ctx = "B13", self.ctx_B
            self.selector = None
            res = {"stress_B": preflight._cached(self, "stress_B13", "full", lambda: preflight.stress_test(self))}
            self.preflight_results.update(res)
            self._check_blocking(res)
        finally:
            self.selector = selector
            self.cond, self.ctx = "LATENCY", self.ctx_H
            self._swap("H")

    def _swap(self, to: str):
        if to not in ("B", "H"):
            raise ValueError("latency phase must be B or H")
        if self.phase == to:
            return
        if self.swap_mode == "reload":
            self._reload(to)
            return
        t0 = time.perf_counter()
        lvl = int(self.p.get("SLEEP_LEVEL", 1))
        if to == "B":
            self.engines["G"].sleep(lvl)
            self.engines["J"].sleep(lvl)
            self.engines["B"].wake()
        else:
            self.engines["B"].sleep(lvl)
            self.engines["G"].wake()
            self.engines["J"].wake()
        self.phase = to
        dt = time.perf_counter() - t0
        append_jsonl(self.f["swaps"], {"ts": utc_now(), "to": to, "seconds": dt,
                                       "nvml_used": self.sampler.now_used() if self.sampler else None})
        log.info("swapped to phase %s in %.1f s (outside timer)", to, dt)

    def run(self):
        from .algorithms import Case, run_case

        p = self.p
        items = build_items(self.dev_problems(), int(p.get("LAT_N_DEV", 20)), int(p.get("LAT_SYNTH_PER_BAND", 3)))
        rng = random.Random(self.cfg["seeds"]["planning"])
        rng.shuffle(items)
        bs = int(p.get("LAT_BLOCK_SIZE", 8))
        blocks = [items[i:i + bs] for i in range(0, len(items), bs)]
        write_json(self.dir / "latency_plan.json", {"items": items, "blocks": [[x["item_id"] for x in b] for b in blocks]})
        seeds = self.seeds
        done = {(r["item_id"], r["condition"], r["seed"]) for r in read_jsonl(self.f["latency_metrics"])}
        conds_H = [c for c in self.CONDS_H if c in self.conds]
        conds_B = [c for c in self.CONDS_B if c in self.conds]
        # warm-up both phases (outside timer, not recorded as results)
        with self.stage("warmup"):
            for ph, cs in (("H", conds_H), ("B", conds_B)):
                if not cs:
                    continue
                self._swap(ph)
                for c in cs:
                    ctx = self._condition_context(c)
                    self._warm_condition(c, ctx, items)
        total = len(items) * len(seeds) * (len(conds_H) + len(conds_B))
        n = len(done)
        t_loop = time.time()
        with self.stage("latency_loop"):
            for bi, block in enumerate(blocks):
                rot = bi % max(1, len(conds_H))
                order_H = conds_H[rot:] + conds_H[:rot]
                phases = [("B", conds_B), ("H", order_H)] if bi % 2 == 0 else [("H", order_H), ("B", conds_B)]
                log.info("block %d/%d: phase order %s", bi + 1, len(blocks), [f"{ph}:{cs}" for ph, cs in phases])
                for ph, cs in phases:
                    if not cs:
                        continue
                    self._swap(ph)
                    for c in cs:
                        ctx = self._condition_context(c)
                        if self.swap_mode == "reload":
                            self._warm_condition(c, ctx, items)
                        if self.swap_mode == "sleep" and c == "G_SINGLE" and p.get("SLEEP_J_FOR_G_SINGLE", True):
                            self.engines["J"].sleep(int(p.get("SLEEP_LEVEL", 1)))
                        for seed in seeds:
                            for it in block:
                                if (it["item_id"], c, seed) in done:
                                    continue
                                t_wall0 = utc_now()
                                try:
                                    out = run_case(Case(it["item_id"], it["problem"], seed, c), ctx, p["TIMEOUT_S"])
                                except Exception as e:  # noqa: BLE001
                                    append_jsonl(self.f["failures"], {"item_id": it["item_id"], "condition": c,
                                                                      "error": str(e), "traceback": traceback.format_exc()})
                                    log.error("latency case failed %s %s: %s", c, it["item_id"], e)
                                    continue
                                r = out.run
                                peak = self.sampler.window_peak(r["t_start_perf"], r["t_end_perf"]) if self.sampler else None
                                rec = {k: v for k, v in r.items() if k not in ("raw_output",)}
                                rec.update(item_id=it["item_id"], kind=it["kind"], block=bi, phase=ph,
                                           t_start_utc=t_wall0, t_end_utc=utc_now(), nvml_peak_used_bytes=peak,
                                           config_hash=self.chash)
                                if "condition_labels" in self.cfg:
                                    rec["condition_label"] = self.cfg["condition_labels"][c]
                                append_jsonl(self.f["latency_metrics"], rec)
                                n += 1
                                log.info("[LAT %s] %d/%d %s seed=%s status=%s T=%.2fs first=%s tokens=%s",
                                         c, n, total, it["item_id"], seed, r["status"], r["T_total"],
                                         f"{r['T_first_accepted']:.2f}s" if r.get("T_first_accepted") else "-",
                                         r.get("candidate_tokens_total"))
                        if self.swap_mode == "sleep" and c == "G_SINGLE" and p.get("SLEEP_J_FOR_G_SINGLE", True):
                            self.engines["J"].wake()
                write_json(self.dir / "progress.json", {"done": n, "total": total, "block": bi + 1,
                                                        "elapsed_s": time.time() - t_loop, "updated_utc": utc_now()})
                self.checkpoint()

    def shutdown(self):
        if self.swap_mode != "reload":
            return super().shutdown()
        try:
            self._close_reload_roles(list(self.engines))
            self.phase = None
        finally:
            if self.sampler:
                self.sampler.stop()

    def run_all(self):
        try:
            self.capture_environment()
            self.prepare_models()
            if self.cfg.get("protocol_version") == "2":
                self.preflight_before_engines()
            self.start_engines()
            if self.cfg.get("protocol_version") == "2":
                self.preflight_after_engines()
            self.run()
            return self.finalize()
        except Exception:
            log.error("latency study aborted:\n%s", traceback.format_exc())
            self.checkpoint()
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Same-GPU latency study; smoke: --n-dev 2 --synth-per-band 1")
    parser.add_argument("--config-file", default="config/experiment.json")
    parser.add_argument("--data-subdir", default="data")
    parser.add_argument("--root", default="/content/jev_llm")
    parser.add_argument("--run-tag", default="main")
    parser.add_argument("--n-dev", type=int, default=20)
    parser.add_argument("--synth-per-band", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--conditions", default=",".join(LatencyStudy.CONDS_H + LatencyStudy.CONDS_B))
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--swap-mode", choices=("sleep", "reload"), default=None)
    args = parser.parse_args()
    swap = {} if args.swap_mode is None else {"LATENCY_SWAP_MODE": args.swap_mode}
    exp = LatencyStudy(CONFIG_FILE=args.config_file, DATA_SUBDIR=args.data_subdir, ROOT=args.root,
                       RUN_TAG=args.run_tag, LAT_N_DEV=args.n_dev, LAT_SYNTH_PER_BAND=args.synth_per_band,
                       LAT_BLOCK_SIZE=args.block_size, LAT_CONDITIONS=args.conditions, SEEDS=args.seeds, **swap)
    try:
        exp.run_all()
    finally:
        exp.shutdown()


if __name__ == "__main__":
    main()
