"""Protocol §14 validations integrated in every condition notebook (amendment v3.1 §A6).

Each check returns {"ok": bool, "blocking": bool, "summary": str, ...details}. Heavy checks
are cached per software/hardware stack under ROOT/preflight_cache/<stack_hash>/ and reused
in "light" mode. In v2, tokenizer, native selector equivalence and stress are always mandatory;
"off" cannot bypass them, and graph disagreement is published without blocking.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

from .common import config_hash, read_json, sha256_file, sha256_text, write_json
from .parsing import extract_final, parse_strict_answer, terminal_state

log = logging.getLogger("jevlab.preflight")


def _stack_hash(exp) -> str:
    env = exp.manifest.get("environment", {})
    gpu = env.get("gpus", [{}])[0].get("name") if env.get("gpus") else None
    return config_hash({"packages": env.get("packages"), "gpu": gpu, "models": exp.cfg["models"],
                        "config": exp.config, "condition": exp.cond,
                        "eager": exp.p["ENFORCE_EAGER"], "readout": exp.p["J_READOUT"]})[:16]


def _cached(exp, name: str, mode: str, fn, blocking: bool = True):
    cdir = Path(exp.p["ROOT"]) / "preflight_cache" / _stack_hash(exp)
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{name}.json"
    if mode == "light" and f.exists():
        r = read_json(f)
        r["from_cache"] = str(f)
        return r
    if mode == "off":
        return {"ok": True, "blocking": False, "summary": "skipped (PREFLIGHT_MODE=off)", "skipped": True}
    t0 = time.perf_counter()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        r = {"ok": False, "blocking": blocking, "summary": f"exception: {type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}
    r["seconds"] = round(time.perf_counter() - t0, 2)
    write_json(f, r)
    return r


# ------------------------------------------------------------------ cheap checks
def parser_selftest() -> dict:
    cases = [("FINAL: 1/2\n", "1/2"), ("FINAL: 0.5\n", "1/2"), ("FINAL: 2/4\n", "1/2"),
             ("FINAL: -3\n", "-3"), ("FINAL: \u22123\n", "-3"), ("FINAL: 2+2\n", None),
             ("FINAL: 5 apples\n", None), ("FINAL: 1/0\n", None), ("no final", None),
             ("FINAL:  7 \n", "7"), ("FINAL: .25\n", "1/4")]
    bad = []
    for text, want in cases:
        st, raw, v = extract_final(text)
        got = None if v is None else (str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}")
        if got != want:
            bad.append({"text": text, "want": want, "got": got, "status": st})
    t_cases = [("a\nFINAL: 3", False, False), ("a\nFINAL: 3", True, True), ("x\nFINAL: 3\n", False, True),
               ("FINAL:", False, False), ("text", True, True)]
    for text, eos, want_terminal in t_cases:
        if terminal_state(text, eos).terminal != want_terminal:
            bad.append({"terminal_case": text, "eos": eos})
    return {"ok": not bad, "blocking": True, "summary": f"{len(cases) + len(t_cases)} cases, {len(bad)} failures",
            "failures": bad}


def template_check(exp) -> dict:
    from .prompts import THINK_OFF_SUFFIX, render_generator_prompt

    out, ok = {}, True
    for role, tok in exp.toks.items():
        if role == "J":
            continue
        txt = render_generator_prompt(tok, exp.prompts["generador"], "What is 1+1?")
        out[role] = {"sha256": sha256_text(txt), "ends_with_nothink": txt.endswith(THINK_OFF_SUFFIX),
                     "rendered": txt}
        ok &= txt.endswith(THINK_OFF_SUFFIX)
        (exp.dir / "preflight" / f"rendered_template_{role}.txt").write_text(txt, encoding="utf-8")
    return {"ok": ok, "blocking": True, "summary": f"non-thinking prefix rendered for {list(out)}", "roles": out}


def tokenizer_check(exp) -> dict:
    from . import models

    rows = {}
    for role in (r for r in exp.toks if r != "J"):
        tok = exp.toks[role]
        revision = exp.cfg["models"][role]["revision"]
        cache = Path(exp.p["HF_CACHE"]) / f"newline_ids_{role}_{revision[:10]}.json"
        expected = models.newline_token_ids(tok)
        cached = set(read_json(cache)) if cache.exists() else set()
        eos = models.eos_ids(tok)
        special = models.special_ids(tok)
        actual_eos = exp.eos[role] if hasattr(exp, "eos") else exp.eos_ids
        actual_nl = exp.nl[role] if hasattr(exp, "nl") else exp.newline_ids
        rows[role] = {"revision": revision, "tokenizer_sha256": sha256_file(Path(exp.paths[role]) / "tokenizer.json"),
                      "special": special, "eos_ids": eos, "newline_cache": str(cache),
                      "newline_set_sha256": sha256_text(json.dumps(sorted(expected))),
                      "newline_set_size": len(expected),
                      "ok": bool(eos) and tok.pad_token_id is not None and bool(expected)
                            and cached == expected == actual_nl and eos == actual_eos}
    return {"ok": bool(rows) and all(r["ok"] for r in rows.values()), "blocking": True,
            "summary": "revision-specific tokenizer/EOS/PAD/newline validation", "roles": rows}


def prompt_length_check(exp) -> dict:
    from .prompts import generator_prompt_ids

    roles = ("G", "B") if exp.cond == "LATENCY" else ("B" if exp.cond.startswith("B13") else "G",)
    lens = {}
    items = exp.problems()
    if exp.cond == "LATENCY":
        from .latency_study import build_items
        items += [{"id": it["item_id"], "problem": it["problem"]} for it in
                  build_items([], 0, int(exp.p["LAT_SYNTH_PER_BAND"]))]
    for role in roles:
        for it in items:
            ids, _ = generator_prompt_ids(exp.toks[role], exp.prompts["generador"], it["problem"])
            lens[f"{role}:{it['id']}" if len(roles) > 1 else it["id"]] = len(ids)
    over = {k: v for k, v in lens.items() if v > exp.p["MAX_PROMPT_TOKENS"]}
    return {"ok": not over, "blocking": True, "summary": f"max={max(lens.values()) if lens else 0} "
            f"over_limit={len(over)}", "over": over, "max": max(lens.values()) if lens else 0}


def sampling_semantics_check() -> dict:
    from vllm import SamplingParams

    sp = SamplingParams(top_k=0, temperature=0.7, seed=2**63 - 2)
    ok = sp.top_k in (0, -1)
    return {"ok": ok, "blocking": True, "summary": f"top_k=0 stored as {sp.top_k} (disabled); seed<2**63 accepted"}


def lock_check(exp) -> dict:
    ok = set(exp.lock) == set(exp.roles()) and all(v["ok"] for v in exp.lock.values())
    arch = {k: v["architectures"] for k, v in exp.lock.items()}
    return {"ok": ok, "blocking": True, "summary": f"hash verification ok={ok}; architectures={arch}",
            "params": {k: v["params"] for k, v in exp.lock.items()}}


# ------------------------------------------------------------------ selector reference
def selector_items(exp) -> list[dict]:
    """Synthetic decisions (no gold, no model output): short, mid-sentence, long, duplicate."""
    from .prompts import selector_question, selector_state

    dev = exp.dev_problems()
    items = []
    frag = ["First, find the total number of items.\n", "Let x be the unknown quantity, so 3x + 5 = ",
            "FINAL: 12\n", "We have 2 * 7 = 14.\nThen 14 - 3 = 11.\n"]
    for i, it in enumerate(dev[:6]):
        cands = frag[i % 4:] + frag[: i % 4]
        prefix = "" if i % 2 == 0 else "We start by reading the problem.\n"
        crit = exp.prompts["criterio_final"] if i % 3 == 0 else exp.prompts["criterio_paso"]
        items.append({"state": selector_state(it["problem"], prefix), "question": selector_question(crit, cands)})
    long_c = "Compute each partial sum and verify it step by step.\n" * 250
    items.append({"state": selector_state(dev[0]["problem"], "Step one.\n"),
                  "question": selector_question(exp.prompts["criterio_paso"], [long_c, "x", long_c[:500], "y"])})
    items.append({"state": selector_state(dev[1]["problem"], ""),
                  "question": selector_question(exp.prompts["criterio_final"], ["FINAL: 3\n"] * 4)})
    return items


def native_reference(exp) -> dict:
    items = selector_items(exp)
    d = exp.dir / "preflight"
    inp, out = d / "native_ref_in.json", d / "native_ref_out.json"
    inp.write_text(json.dumps(items))
    cmd = [sys.executable, "-m", "jevlab.native_ref", str(inp), str(out), exp.paths["J"]]
    log.info("running native JevK5 reference in a subprocess (%d decisions) ...", len(items))
    from .common import subprocess_env

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=subprocess_env())
    (d / "native_ref_stderr.txt").write_text(r.stdout[-20000:] + "\n---\n" + r.stderr[-20000:])
    if r.returncode != 0:
        return {"ok": False, "blocking": True, "summary": f"native runtime failed rc={r.returncode}",
                "stderr_tail": r.stderr[-3000:]}
    res = json.loads(out.read_text())
    ids_ok = len(res.get("items", [])) == len(items) and all(x["ids_equal_prompt_text"] for x in res["items"])
    return {"ok": ids_ok, "blocking": True, "summary": f"native ok; ids==prompt_text: {ids_ok}; "
            f"T={res['temperature']}", "native": res}


def selector_equivalence(exp, native: dict) -> dict:
    """vLLM readout vs native runtime on identical token ids. Tolerance for bf16: argmax must
    match unless the native top-2 raw-logit margin is <= 2 bf16 ulps at that magnitude."""
    import hashlib

    from .prompts import selector_prompt_ids

    if not native or not native.get("native"):
        return {"ok": False, "blocking": True, "summary": "no native reference available"}
    items = selector_items(exp)
    reference = native["native"]
    if (not native.get("ok") or len(reference.get("items", [])) != len(items) or not items
            or abs(reference.get("temperature", 0) - exp.cfg["selector"]["calibration_temperature_expected"]) > 1e-9):
        return {"ok": False, "blocking": True, "summary": "incomplete/invalid native reference or calibration mismatch"}
    rows, n_bad = [], 0
    for it, nat in zip(items, native["native"]["items"]):
        ids, _ = selector_prompt_ids(exp.toks["J"], it["state"], it["question"])
        same_ids = hashlib.sha256(json.dumps(ids).encode()).hexdigest() == nat["ids_sha256"]
        n = len(it["question"]["criteria"])
        logits, dt = exp.selector.letter_logits_from_ids(ids, n)
        probs = exp.selector.probabilities(logits)
        nl, npb = nat["raw_logits"], nat["probabilities"]
        if (any(len(v) != n for v in (logits, probs, nl, npb))
                or not all(math.isfinite(x) for v in (logits, probs, nl, npb) for x in v)
                or any(x < 0 or x > 1 for v in (probs, npb) for x in v)
                or any(abs(sum(v) - 1) > 1e-5 for v in (probs, npb))):
            rows.append({"same_ids": same_ids, "ok": False, "error": "invalid logits/probabilities"})
            n_bad += 1
            continue
        am_v, am_n = probs.index(max(probs)), npb.index(max(npb))
        srt = sorted(nl, reverse=True)
        margin = srt[0] - srt[1] if len(srt) > 1 else 99
        e = math.floor(math.log2(max(abs(srt[0]), 1e-6)))
        ulp = 2.0 ** (e - 7)  # bf16 has 8 significant bits
        tie_tolerated = margin <= 2 * ulp and max(nl) - nl[am_v] <= 2 * ulp
        ok_row = same_ids and (am_v == am_n or tie_tolerated)
        n_bad += int(not ok_row)
        rows.append({"n_ids": len(ids), "same_ids": same_ids, "vllm_logits": logits, "native_logits": nl,
                     "vllm_probs": probs, "native_probs": npb, "max_abs_prob_diff":
                     max(abs(a - b) for a, b in zip(probs, npb)), "argmax_vllm": am_v, "argmax_native": am_n,
                     "native_top2_margin": margin, "bf16_ulp": ulp, "tie_tolerated": tie_tolerated,
                     "vllm_seconds": dt, "native_seconds": nat["seconds"]})
    maxdiff = max((r.get("max_abs_prob_diff", 0) for r in rows), default=0)
    return {"ok": n_bad == 0, "blocking": True,
            "summary": f"{len(rows)} decisions, argmax mismatches beyond bf16 tolerance={n_bad}, "
                       f"max |dp|={maxdiff:.4f}", "rows": rows}


# ------------------------------------------------------------------ generator checks
def continuity_test(exp) -> dict:
    """Greedy continuation from prefix ids must agree with one long greedy request."""
    from .engines import Req, sampling_params
    from .prompts import generator_prompt_ids

    eng, tok = exp.ctx.gen, exp.ctx.tok
    it = exp.dev_problems()[0]
    ids, _ = generator_prompt_ids(tok, exp.prompts["generador"], it["problem"])
    sp = lambda m: sampling_params(temperature=0.0, top_p=1.0, top_k=0, max_tokens=m, seed=None,  # noqa: E731
                                   stop_token_ids=[], ignore_eos=True)
    rid = eng.new_id("cont")
    full = eng.run([Req(rid, ids, sp(160), watch_final=False)])[rid].token_ids
    rows = []
    for k in (1, 63, 64, 65, 127, 128):
        if k + 16 > len(full):
            continue
        rid = eng.new_id("cont")
        cont = eng.run([Req(rid, ids + full[:k], sp(16), watch_final=False)])[rid].token_ids
        ref = full[k:k + 16]
        agree = sum(a == b for a, b in zip(cont, ref))
        first_div = next((i for i, (a, b) in enumerate(zip(cont, ref)) if a != b), None)
        rows.append({"prefix": k, "agree_16": agree, "first_divergence": first_div})
    rate = sum(r["agree_16"] for r in rows) / max(1, 16 * len(rows))
    return {"ok": rate >= 0.75, "blocking": False,
            "summary": f"greedy agreement {rate:.3f} over prefixes {[r['prefix'] for r in rows]} "
                       "(bf16 prefill vs decode kernels may legitimately diverge)", "rows": rows}


def batching_audit(exp) -> dict:
    """4 requests one-by-one vs together (10 dev prefixes, 64 tokens)."""
    from .engines import Req, sampling_params
    from .prompts import generator_prompt_ids

    eng, tok = exp.ctx.gen, exp.ctx.tok
    dev = exp.dev_problems()[:10]
    seq_t, bat_t = [], []
    for i, it in enumerate(dev):
        ids, _ = generator_prompt_ids(tok, exp.prompts["generador"], it["problem"])
        mk = lambda b: Req(eng.new_id("aud"), ids, sampling_params(  # noqa: E731
            temperature=0.7, top_p=0.9, top_k=0, max_tokens=64, seed=1000 + 10 * i + b,
            stop_token_ids=[], ignore_eos=True), watch_final=False)
        t0 = time.perf_counter()
        for b in range(4):
            eng.run([mk(b)])
        seq_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        eng.run([mk(b) for b in range(4)])
        bat_t.append(time.perf_counter() - t0)
    s, b = sum(seq_t), sum(bat_t)
    return {"ok": True, "blocking": False, "summary": f"sequential {s:.2f}s vs batched {b:.2f}s "
            f"-> batching speedup x{s / b:.2f}", "sequential_s": seq_t, "batched_s": bat_t,
            "speedup": s / b}


def stress_test(exp) -> dict:
    """Max-length generation (ignore_eos) with all candidates, plus a J input near 16k."""
    from .engines import Req, sampling_params
    from .prompts import generator_prompt_ids

    eng, tok = exp.ctx.gen, exp.ctx.tok
    longest = max(exp.dev_problems(), key=lambda x: len(generator_prompt_ids(tok, exp.prompts["generador"], x["problem"])[0]))
    ids, _ = generator_prompt_ids(tok, exp.prompts["generador"], longest["problem"])
    n = exp.p["N_CANDIDATES"] if exp.cond in ("J64", "JSTEP", "JFINAL") else 1
    m = exp.p["MAX_OUTPUT_TOKENS"]
    if len(ids) + m > exp.p["GB_CONTEXT"]:
        return {"ok": False, "blocking": True, "summary": "stress prompt plus full output exceeds context"}
    stress_start = time.perf_counter()
    t0 = stress_start
    reqs = [Req(eng.new_id("stress"), ids, sampling_params(temperature=0.7, top_p=0.9, top_k=0, max_tokens=m,
                                                           seed=77 + b, stop_token_ids=[], ignore_eos=True),
                watch_final=False) for b in range(n)]
    res = eng.run(reqs)
    gen_s = time.perf_counter() - t0
    out = {"generator": {"n": n, "max_tokens": m, "tokens": [len(r.token_ids) for r in res.values()],
                         "seconds": gen_s}}
    ok = len(res) == n and all(len(r.token_ids) == m for r in res.values())
    if exp.selector is not None:
        from .prompts import selector_prompt_ids, selector_question, selector_state

        target = exp.p["J_MAX_INPUT"] - 64
        body = "Check the arithmetic of this line and keep the running total consistent.\n"
        lo, hi = 0, 5000
        ids_j = []
        while lo <= hi:
            k = (lo + hi) // 2
            q = selector_question(exp.prompts["criterio_paso"], [body * k, "a", "b", "c"])
            ids_j, _ = selector_prompt_ids(exp.toks["J"], selector_state("Stress.", ""), q)
            if target <= len(ids_j) <= exp.p["J_MAX_INPUT"]:
                break
            if len(ids_j) < target:
                lo = k + 1
            else:
                hi = k - 1
        if not target <= len(ids_j) <= exp.p["J_MAX_INPUT"]:
            return {"ok": False, "blocking": True, "summary": "could not construct near-limit selector input", **out}
        t0 = time.perf_counter()
        logits, dt = exp.selector.letter_logits_from_ids(ids_j, 4)
        out["selector"] = {"input_tokens": len(ids_j), "seconds": dt,
                            "finite": len(logits) == 4 and all(math.isfinite(x) for x in logits)}
        ok &= out["selector"]["finite"]
    peak = exp.sampler.window_peak(stress_start, time.perf_counter()) if exp.sampler else None
    out["nvml_peak_gib"] = peak / 2**30 if peak is not None else None
    return {"ok": ok, "blocking": True, "summary": f"max-length generation ok={ok}; "
            f"J16k={out.get('selector', {}).get('input_tokens')} tokens; NVML observed peak {out['nvml_peak_gib']} GiB",
            **out}


def eager_vs_graph(exp) -> dict:
    """Greedy tokens with CUDA graphs vs an eager G engine (3 dev prompts x 64 tokens)."""
    from .engines import Engine, Req, sampling_params
    from .prompts import generator_prompt_ids

    if exp.cond.startswith("B13"):
        return {"ok": True, "blocking": False, "summary": "skipped for B (memory)", "skipped": True}
    if exp.cfg.get("protocol_version") == "2":
        return eager_vs_graph_v2(exp)
    tok = exp.toks["G"]
    eg = Engine("G_eager", exp.paths["G"], tok, max_model_len=exp.p["GB_CONTEXT"], max_num_seqs=1,
                kv_cache_gb=1.0, gpu_memory_utilization=exp.p["GPU_MEMORY_UTILIZATION"],
                enforce_eager=True, eos_ids=exp.eos_ids)
    rows = []
    try:
        for it in exp.dev_problems()[:3]:
            ids, _ = generator_prompt_ids(tok, exp.prompts["generador"], it["problem"])
            sp = sampling_params(temperature=0.0, top_p=1.0, top_k=0, max_tokens=64, seed=None,
                                 stop_token_ids=[], ignore_eos=True)
            a = exp.engines["G"].run([Req(exp.engines["G"].new_id("eg"), ids, sp, False)])
            b = eg.run([Req(eg.new_id("eg"), ids, sp, False)])
            ta, tb = list(a.values())[0].token_ids, list(b.values())[0].token_ids
            rows.append({"agree": sum(x == y for x, y in zip(ta, tb)), "n": len(ta),
                         "first_divergence": next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)})
    finally:
        eg.shutdown()
    rate = sum(r["agree"] for r in rows) / max(1, sum(r["n"] for r in rows))
    return {"ok": rate >= 0.75, "blocking": False, "summary": f"greedy token agreement graph vs eager {rate:.3f}",
             "rows": rows}


def eager_vs_graph_v2(exp) -> dict:
    """Swap G engines outside measurement; never hold two physical G copies on GPU."""
    from .engines import Engine, Req, sampling_params
    from .prompts import generator_prompt_ids

    if exp.p["ENFORCE_EAGER"]:
        return {"ok": True, "blocking": False, "summary": "eager replica: no graph engine to compare", "skipped": True}
    dev = exp.dev_problems()[:20]
    if len(dev) != 20:
        raise ValueError("v2 graph validation requires 20 development prompts available")
    p, tok = exp.p, exp.toks["G"]
    graph = exp.engines["G"]
    sleep_mode = graph.engine_kwargs.get("enable_sleep_mode", False)
    kwargs = dict(max_model_len=p["GB_CONTEXT"], max_num_seqs=4, kv_cache_gb=p["KV_CACHE_GB_G"],
                  gpu_memory_utilization=p["GPU_MEMORY_UTILIZATION"], eos_ids=exp.eos_ids,
                  newline_ids=exp.newline_ids, sleep_mode=sleep_mode)

    def tokens(engine, it):
        ids, _ = generator_prompt_ids(tok, exp.prompts["generador"], it["problem"])
        sp = sampling_params(temperature=0.0, top_p=1.0, top_k=0, max_tokens=64, seed=None,
                             stop_token_ids=[], ignore_eos=True)
        rid = engine.new_id("eg")
        return engine.run([Req(rid, ids, sp, False)])[rid].token_ids

    a = [tokens(graph, it) for it in dev[:3]]
    cleanup = []
    reload_mode = exp.p.get("LATENCY_SWAP_MODE") == "reload"
    if reload_mode:
        from .runner import shutdown_engine_checked
        cleanup.append(shutdown_engine_checked(graph))
        del exp.engines["G"]
        exp.ctx.gen = None
    else:
        graph.shutdown()
    eager = None
    expanded = False
    try:
        eager = Engine("G_eager", exp.paths["G"], tok, enforce_eager=True, **kwargs)
        if reload_mode:
            exp.engines["G"] = eager
        b = [tokens(eager, it) for it in dev[:3]]
        initial_rate = sum(sum(x == y for x, y in zip(ta, tb)) for ta, tb in zip(a, b)) / (3 * 64)
        expanded = initial_rate < 0.75
        if expanded:
            b.extend(tokens(eager, it) for it in dev[3:])
    finally:
        if eager is not None:
            if reload_mode:
                cleanup.append(shutdown_engine_checked(eager))
                del exp.engines["G"]
            else:
                eager.shutdown()
        # A failed constructor exposes no process handles: do not load another G in reload mode.
        if eager is not None or not reload_mode:
            exp.engines["G"] = Engine("G", exp.paths["G"], tok, enforce_eager=False, **kwargs)
            exp.ctx.gen = exp.engines["G"]
            if reload_mode and getattr(exp, "ctx_H", None) is not None:
                exp.ctx_H.gen = exp.engines["G"]
    if expanded:
        a.extend(tokens(exp.engines["G"], it) for it in dev[3:])
    rows = [{"problem_id": it["id"], "graph_tokens": ta, "eager_tokens": tb,
             "agree": sum(x == y for x, y in zip(ta, tb)), "n": 64,
             "complete": len(ta) == len(tb) == 64}
            for it, ta, tb in zip(dev, a, b)]
    rate = sum(r["agree"] for r in rows) / (64 * len(rows))
    result = {"ok": rate >= 0.75 and all(r["complete"] for r in rows), "blocking": False,
              "summary": f"graph/eager greedy agreement {rate:.3f}; {len(rows)} dev prompts; expanded={expanded}",
              "agreement": rate, "initial_agreement": initial_rate, "threshold": 0.75,
              "expanded_to_20": expanded, "rows": rows}
    if reload_mode:
        result["process_cleanup"] = cleanup
    write_json(exp.dir / "preflight" / "eager_vs_graph.json", result)
    return result


def r4_memory_check(exp) -> dict:
    """Memory-only reference for the protocol's R4 profile: three extra G engines are started
    next to the running G+J (4 physical G copies + J), NVML is read, and they are shut down."""
    from .engines import Engine

    extra = []
    t0 = time.perf_counter()
    try:
        for i in range(3):
            extra.append(Engine(f"G_r4_{i}", exp.paths["G"], exp.toks["G"], max_model_len=exp.p["GB_CONTEXT"],
                                max_num_seqs=1, kv_cache_gb=exp.p["KV_CACHE_GB_G"],
                                gpu_memory_utilization=exp.p["GPU_MEMORY_UTILIZATION"],
                                enforce_eager=exp.p["ENFORCE_EAGER"], eos_ids=exp.eos_ids))
        time.sleep(2)
        used = exp.sampler.now_used() if exp.sampler else None
        procs = exp.sampler.processes() if exp.sampler else []
    finally:
        for e in extra:
            e.shutdown()
    return {"ok": True, "blocking": False, "summary": f"R4 (4xG + J resident) NVML used = {(used or 0) / 2**30:.2f} GiB",
            "nvml_used_bytes": used, "processes": procs, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------ orchestration
def pre_engine_checks(exp, mode: str) -> dict:
    res = {"parser_selftest": parser_selftest(), "lock": lock_check(exp),
           "chat_template": template_check(exp), "prompt_length": prompt_length_check(exp),
           "sampling_semantics": sampling_semantics_check()}
    v2 = exp.cfg.get("protocol_version") == "2"
    if v2:
        res["tokenizer"] = tokenizer_check(exp)
    if "J" in exp.roles():
        res["selector_native_reference"] = _cached(exp, "selector_native_reference", "full" if v2 else mode,
                                                   lambda: native_reference(exp))
    return res


def post_engine_checks(exp, mode: str) -> dict:
    res = {}
    v2 = exp.cfg.get("protocol_version") == "2"
    if exp.selector is not None:
        nat = exp.preflight_results.get("selector_native_reference")
        res["selector_equivalence"] = (selector_equivalence(exp, nat) if v2 or (mode != "off" and nat and not nat.get("skipped"))
                                       else {"ok": True, "blocking": False, "summary": "skipped"})
    res["continuity"] = _cached(exp, f"continuity_{exp.cond[:3]}", mode, lambda: continuity_test(exp))
    if exp.cond in ("J64", "JSTEP", "JFINAL"):
        res["batching_audit"] = _cached(exp, "batching_audit", mode, lambda: batching_audit(exp))
    if (v2 or mode == "full") and not exp.cond.startswith("B13"):
        res["eager_vs_graph"] = _cached(exp, "eager_vs_graph", "full" if v2 else mode,
                                        lambda: eager_vs_graph(exp), blocking=exp.p.get("LATENCY_SWAP_MODE") == "reload")
    res["stress"] = _cached(exp, f"stress_{exp.cond}", "full" if v2 else mode, lambda: stress_test(exp))
    if not v2 and exp.p.get("RUN_R4_MEMORY_CHECK") and exp.cond in ("J64", "JSTEP", "JFINAL"):
        res["r4_memory"] = _cached(exp, "r4_memory", mode, lambda: r4_memory_check(exp))
    return res
