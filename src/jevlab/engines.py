"""vLLM engine wrapper with a step loop (amendment v3.1 §A4).

Every arm is driven by the same loop: add requests -> engine.step() until done. This gives
uniform FINAL/EOS termination (a request is aborted as soon as its FINAL line is complete),
first-token timestamps, and a hard deadline (600 s per run). Each engine runs its EngineCore
in its own process (enable_multiprocessing=True), so two engines (G and J) can coexist on one
GPU and the notebook process never creates a CUDA context.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from .parsing import terminal_state

log = logging.getLogger("jevlab.engines")


@dataclass
class Req:
    rid: str
    prompt_ids: list[int]
    params: object                 # vllm.SamplingParams
    watch_final: bool = True       # abort as soon as a complete FINAL line appears


@dataclass
class ReqResult:
    rid: str
    prompt_len: int
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None   # stop | length | abort | final_abort | timeout
    stop_reason: object = None
    eos: bool = False
    t_add: float = 0.0
    t_first: float | None = None
    t_end: float | None = None
    logprobs: object = None
    steps_seen: int = 0


class Engine:
    """Thin wrapper around vllm.v1 LLMEngine."""

    def __init__(self, name: str, model_dir: str, tokenizer, *, max_model_len: int,
                 max_num_seqs: int, kv_cache_gb: float | None, gpu_memory_utilization: float,
                 enforce_eager: bool, eos_ids: list[int], newline_ids: set[int] | None = None,
                 logprobs_mode: str | None = None, max_logprobs: int = 20,
                 language_model_only: bool = True, sleep_mode: bool = False, seed: int = 0,
                 extra: dict | None = None):
        from vllm import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine

        self.name, self.model_dir, self.tok = name, model_dir, tokenizer
        self.eos_ids = list(eos_ids)
        self.newline_ids = newline_ids or set()
        kw = dict(model=model_dir, tokenizer=model_dir, dtype="bfloat16",
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  gpu_memory_utilization=gpu_memory_utilization,
                  enable_prefix_caching=False, generation_config="vllm", seed=seed,
                  enforce_eager=enforce_eager, max_logprobs=max_logprobs,
                  enable_sleep_mode=sleep_mode, disable_log_stats=False,
                  trust_remote_code=False)
        if kv_cache_gb:
            kw["kv_cache_memory_bytes"] = int(kv_cache_gb * (1 << 30))
        if logprobs_mode:
            kw["logprobs_mode"] = logprobs_mode
        if language_model_only:
            kw["language_model_only"] = True
        if extra:
            kw.update(extra)
        self.engine_kwargs = {k: v for k, v in kw.items()}
        t0 = time.perf_counter()
        log.info("[%s] starting vLLM engine: %s", name, {k: v for k, v in kw.items()
                                                        if k not in ("model", "tokenizer")})
        self.engine = LLMEngine.from_engine_args(EngineArgs(**kw), enable_multiprocessing=True)
        self.load_seconds = time.perf_counter() - t0
        self._n = 0
        self.sleeping = False
        log.info("[%s] engine ready in %.1f s", name, self.load_seconds)

    # ------------------------------------------------------------------ lifecycle
    def sleep(self, level: int = 1) -> float:
        t0 = time.perf_counter()
        self.engine.sleep(level)
        self.sleeping = True
        return time.perf_counter() - t0

    def wake(self) -> float:
        t0 = time.perf_counter()
        self.engine.wake_up()
        self.sleeping = False
        return time.perf_counter() - t0

    def shutdown(self) -> None:
        try:
            eng = self.engine
            if hasattr(eng, "engine_core") and hasattr(eng.engine_core, "shutdown"):
                eng.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] shutdown error: %s", self.name, e)
        self.engine = None

    # ------------------------------------------------------------------ running
    def new_id(self, tag: str) -> str:
        self._n += 1
        return f"{self.name}-{tag}-{self._n}"

    def run(self, reqs: list[Req], deadline: float | None = None) -> dict[str, ReqResult]:
        """Submit all requests at once (they share batches) and step until all finish.

        A request watching FINAL is aborted right after the step that completes its FINAL
        line; tokens delivered after that token are kept and counted as overproduction by
        the caller. Tokens the engine may have computed but never delivered after an abort
        are not observable (declared in the amendment)."""
        eng = self.engine
        res: dict[str, ReqResult] = {}
        watch = {}
        for r in reqs:
            rr = ReqResult(rid=r.rid, prompt_len=len(r.prompt_ids))
            rr.t_add = time.perf_counter()
            eng.add_request(r.rid, {"prompt_token_ids": list(r.prompt_ids)}, r.params)
            res[r.rid] = rr
            watch[r.rid] = r.watch_final
        live = set(res)
        while live:
            if deadline is not None and time.perf_counter() > deadline:
                eng.abort_request(list(live))
                t = time.perf_counter()
                for rid in live:
                    res[rid].finish_reason = "timeout"
                    res[rid].t_end = t
                live.clear()
                break
            outs = eng.step()
            t = time.perf_counter()
            for o in outs:
                rid = o.request_id
                if rid not in live:
                    continue
                rr = res[rid]
                co = o.outputs[0]
                toks = list(co.token_ids)
                new = toks[len(rr.token_ids):]
                if new and rr.t_first is None:
                    rr.t_first = t
                rr.token_ids = toks
                rr.steps_seen += 1
                if o.finished:
                    rr.finish_reason = co.finish_reason
                    rr.stop_reason = co.stop_reason
                    rr.logprobs = co.logprobs
                    rr.eos = (co.finish_reason == "stop" and
                              (co.stop_reason is None or co.stop_reason in self.eos_ids))
                    # vLLM delivers the EOS id inside token_ids: it is a real sampled token
                    # (counted by the caller via rr.eos) but never part of the text/prefix.
                    if rr.eos and rr.token_ids and rr.token_ids[-1] in self.eos_ids:
                        rr.token_ids = rr.token_ids[:-1]
                    rr.t_end = t
                    live.discard(rid)
                elif watch[rid] and self.newline_ids and any(x in self.newline_ids for x in new):
                    text = self.tok.decode(toks, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
                    if terminal_state(text, eos=False).terminal:
                        eng.abort_request([rid])
                        rr.finish_reason = "final_abort"
                        rr.t_end = t
                        live.discard(rid)
            if not outs and not eng.has_unfinished_requests():
                # engine reports nothing left (e.g. aborted internally)
                t = time.perf_counter()
                for rid in list(live):
                    res[rid].finish_reason = res[rid].finish_reason or "abort"
                    res[rid].t_end = t
                live.clear()
        return res


def sampling_params(*, temperature: float, top_p: float, top_k: int, max_tokens: int,
                    seed: int | None, stop_token_ids: list[int], repetition_penalty: float = 1.0,
                    presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                    ignore_eos: bool = False, **extra):
    from vllm import SamplingParams

    kw = dict(n=1, temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
              top_k=top_k if temperature > 0 else 0, max_tokens=max_tokens, seed=seed,
              stop_token_ids=list(stop_token_ids), repetition_penalty=repetition_penalty,
              presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
              ignore_eos=ignore_eos, detokenize=False, skip_special_tokens=False)
    kw.update(extra)
    return SamplingParams(**kw)


def vllm_logging_config(log_dir: str) -> str:
    """Write a logging config so every EngineCore subprocess also logs to a file we can
    parse (weights memory, KV cache size, graph capture). Returns the config path."""
    import json

    os.makedirs(log_dir, exist_ok=True)
    cfg = {
        "version": 1, "disable_existing_loggers": False,
        "formatters": {"v": {"format": "%(asctime)s %(levelname)s [pid %(process)d] %(name)s: %(message)s"}},
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "v", "level": "INFO",
                        "stream": "ext://sys.stdout"},
            "file": {"class": "logging.FileHandler", "formatter": "v", "level": "DEBUG",
                     "filename": os.path.join(log_dir, "vllm.log"), "mode": "a"},
        },
        "loggers": {"vllm": {"handlers": ["console", "file"], "level": "INFO", "propagate": False}},
    }
    path = os.path.join(log_dir, "vllm_logging.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = path
    return path


def parse_vllm_log(path: str) -> dict:
    """Best-effort extraction of memory facts from vllm.log."""
    import re

    out = {"model_loading": [], "kv_cache": [], "graph": [], "free_memory": [], "raw": []}
    if not os.path.exists(path):
        return out
    pats = {
        "model_loading": re.compile(r"Model loading took ([\d.]+) GiB(?: memory)?(?: and ([\d.]+) seconds)?"),
        "kv_cache": re.compile(r"(GPU KV cache size: [\d,]+ tokens|reserved ([\d.]+) GiB memory for KV Cache|Available KV cache memory: ([\d.]+) GiB)"),
        "graph": re.compile(r"Graph capturing finished in ([\d.]+) secs?, took ([\d.\-]+) GiB"),
        "free_memory": re.compile(r"Initial free memory ([\d.]+) GiB"),
    }
    with open(path, errors="replace") as f:
        for line in f:
            for k, p in pats.items():
                m = p.search(line)
                if m:
                    out[k].append(line.strip()[-300:])
            if "Maximum concurrency" in line or "max_num_batched_tokens" in line:
                out["raw"].append(line.strip()[-300:])
    return out
