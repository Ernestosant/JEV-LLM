"""The six arms (protocol §4, §7) on the SHARED4-VLLM-GRAPH profile.

G_SINGLE / B13 / B13_GREEDY : one solution, no selector.
J64    : 4 blocks of <=64 tokens per round, J chooses, winner's ids appended.
JSTEP  : 4 newline-delimited steps (<=128 tokens) per round, max 64 rounds.
JFINAL : 4 complete solutions, one J decision.

Budget rule before each J64/JSTEP round (§7.4):
    limit = min(mode_limit, 1024 - accepted, floor((4096 - candidates_total)/4))
further capped by the remaining context (4096 - prompt - accepted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .engines import Engine, Req, sampling_params
from .parsing import terminal_state
from .prompts import generator_prompt_ids
from .seeds import branch_seed, option_order_seed, permutation


@dataclass
class Ctx:
    """Everything an arm needs; built once per notebook."""
    gen: Engine                 # G engine (G_SINGLE / hybrids) or B engine (B13*)
    tok: object                 # tokenizer of the generator
    system_prompt: str
    sampling: dict
    limits: dict
    eos_ids: list[int]
    newline_ids: set[int]
    selector: object = None     # Selector for hybrids
    crit_step: str = ""
    crit_final: str = ""
    profile: str = "SHARED4-VLLM-GRAPH"


@dataclass
class Case:
    problem_id: str
    problem: str
    seed: int
    condition: str
    generation: int = 0


@dataclass
class Outcome:
    run: dict
    candidates: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    rounds: list = field(default_factory=list)


def _decode(tok, ids) -> str:
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _delta_text(tok, prefix_ids, ids) -> str:
    """Candidate text as it appears after the accepted prefix (robust to multi-byte chars
    split across block boundaries)."""
    return _decode(tok, list(prefix_ids) + list(ids))[len(_decode(tok, prefix_ids)):]


def finalize_candidate(ctx: Ctx, accepted_ids: list[int], rr, mode_limit: int | None,
                       step_mode: bool) -> dict:
    """Terminality, truncation at the terminal token, overproduction and boundary flags."""
    tok = ctx.tok
    ids = list(rr.token_ids)
    n_sampled = len(ids) + (1 if rr.eos else 0)
    full_text = _decode(tok, accepted_ids + ids)
    ts = terminal_state(full_text, eos=rr.eos)
    kept, over_tok, over_chars = ids, 0, 0
    if ts.terminal and ts.reason == "final_line":
        # earliest token index k whose inclusion completes the FINAL line
        k_cut = len(ids)
        for k in range(1, len(ids) + 1):
            if ids[k - 1] in ctx.newline_ids or k == len(ids):
                if terminal_state(_decode(tok, accepted_ids + ids[:k]), eos=False).terminal:
                    k_cut = k
                    break
        kept = ids[:k_cut]
        over_tok = len(ids) - k_cut
        text_k = _decode(tok, accepted_ids + kept)
        over_chars = len(text_k) - ts.end_char if ts.end_char is not None else 0
    rec = {
        "n_tokens_delivered": len(ids), "n_sampled": n_sampled, "eos": rr.eos,
        "finish_reason": rr.finish_reason, "stop_reason": rr.stop_reason,
        "terminal": ts.terminal, "has_final": ts.has_final, "terminal_reason": ts.reason,
        "kept_ids": kept, "overproduction_tokens": over_tok, "overproduction_chars": max(0, over_chars),
        "text": _delta_text(tok, accepted_ids, kept), "raw_text": _delta_text(tok, accepted_ids, ids),
        "t_first": rr.t_first, "t_add": rr.t_add, "t_end": rr.t_end,
    }
    rec["replacement_chars"] = rec["raw_text"].count("\ufffd")
    if step_mode:
        stopped_nl = rr.finish_reason == "stop" and rr.stop_reason in ctx.newline_ids
        last = ids[-1] if ids else None
        overshoot = False
        if stopped_nl and last is not None:
            ttxt = _decode(tok, [last])
            overshoot = "\n" in ttxt and len(ttxt.split("\n", 1)[1]) > 0
        rec.update(boundary=stopped_nl, boundary_overshoot=overshoot,
                   forced_boundary=(rr.finish_reason == "length" and mode_limit is not None
                                    and len(ids) >= mode_limit and not ts.terminal),
                   empty_step=stopped_nl and rec["text"].strip() == "")
    return rec


def _gen_params(ctx: Ctx, condition: str, max_tokens: int, seed: int, step_mode: bool):
    s = ctx.sampling
    greedy = condition == "B13_GREEDY"
    stops = list(ctx.eos_ids) + (sorted(ctx.newline_ids) if step_mode else [])
    return sampling_params(temperature=0.0 if greedy else s["temperature"], top_p=s["top_p"],
                           top_k=s["top_k"], max_tokens=max_tokens, seed=None if greedy else seed,
                           stop_token_ids=stops, repetition_penalty=s["repetition_penalty"],
                           presence_penalty=s["presence_penalty"],
                           frequency_penalty=s["frequency_penalty"])


def _base_run(case: Case, ctx: Ctx) -> dict:
    return {"problem_id": case.problem_id, "seed": case.seed, "condition": case.condition,
            "profile": ctx.profile, "generation": case.generation}


def _finish(run: dict, ctx: Ctx, accepted_ids: list[int], end_char_hint, status: str,
            t_start: float, t_first_accept: float | None, times: dict, counters: dict) -> None:
    raw = _decode(ctx.tok, accepted_ids)
    ts = terminal_state(raw, eos=status in ("final", "eos_invalid"))
    public = raw[: ts.end_char] if ts.has_final and ts.end_char is not None else raw
    t_end = time.perf_counter()
    run.update(status=status, raw_output=raw, public_output=public,
               public_overproduction_chars=len(raw) - len(public),
               accepted_tokens=len(accepted_ids), T_total=t_end - t_start,
               T_first_accepted=(t_first_accept - t_start) if t_first_accept else None,
               t_end_perf=t_end, **times, **counters)
    run["T_control"] = max(0.0, run["T_total"] - run.get("T_proposals", 0.0) - run.get("T_selector", 0.0))


# ----------------------------------------------------------------------- single-solution arms
def run_single(case: Case, ctx: Ctx, t_start: float, deadline: float) -> Outcome:
    run = _base_run(case, ctx)
    ids_prompt, _ = generator_prompt_ids(ctx.tok, ctx.system_prompt, case.problem)
    lim = min(ctx.limits["max_output_tokens"], ctx.limits["gb_context"] - len(ids_prompt))
    seed = branch_seed(case.seed, case.problem_id, case.condition, 0, 0, case.generation)
    rid = ctx.gen.new_id(case.condition)
    t0 = time.perf_counter()
    rr = ctx.gen.run([Req(rid, ids_prompt, _gen_params(ctx, case.condition, lim, seed, False))],
                     deadline)[rid]
    t1 = time.perf_counter()
    c = finalize_candidate(ctx, [], rr, None, False)
    status = ("timeout" if rr.finish_reason == "timeout" else
              "final" if c["terminal"] and c["has_final"] else
              "eos_invalid" if c["terminal"] else "truncated")
    cand = {**_base_run(case, ctx), "round": 0, "branch": 0, "seed_branch": seed, **c}
    cand.pop("kept_ids")
    times = {"T_proposals": t1 - t0, "T_selector": 0.0, "T_prefill": (rr.t_first - rr.t_add) if rr.t_first else None,
             "T_cache_sync": 0.0}
    counters = {"prompt_tokens": len(ids_prompt), "candidate_tokens_total": c["n_sampled"],
                "overproduction_tokens": c["overproduction_tokens"], "rounds": 1, "selector_calls": 0,
                "selector_input_tokens": 0, "processed_prompt_tokens": len(ids_prompt),
                "seed_branch": seed, "tokenizer": "B" if case.condition.startswith("B13") else "G"}
    _finish(run, ctx, c["kept_ids"], None, status, t_start, rr.t_first, times, counters)
    return Outcome(run, [cand], [], [])


# ----------------------------------------------------------------------- hybrid arms
def _round(ctx: Ctx, case: Case, ids_prompt, accepted, rnd, limit, step_mode, deadline):
    reqs, seeds = [], []
    for b in range(ctx.limits["n_candidates"]):
        s = branch_seed(case.seed, case.problem_id, case.condition, rnd, b, case.generation)
        seeds.append(s)
        reqs.append(Req(ctx.gen.new_id(f"{case.condition}-r{rnd}b{b}"), ids_prompt + accepted,
                        _gen_params(ctx, case.condition, limit, s, step_mode),
                        watch_final=not step_mode))
    t0 = time.perf_counter()
    res = ctx.gen.run(reqs, deadline)
    t1 = time.perf_counter()
    rrs = [res[r.rid] for r in reqs]
    return rrs, seeds, t0, t1


def run_hybrid(case: Case, ctx: Ctx, t_start: float, deadline: float) -> Outcome:
    mode = case.condition
    L = ctx.limits
    n = L["n_candidates"]
    step_mode = mode == "JSTEP"
    mode_limit = {"J64": L["j64_block"], "JSTEP": L["jstep_step"], "JFINAL": L["max_output_tokens"]}[mode]
    max_rounds = L["jstep_max_rounds"] if mode == "JSTEP" else (1 if mode == "JFINAL" else 10**9)
    run = _base_run(case, ctx)
    ids_prompt, _ = generator_prompt_ids(ctx.tok, ctx.system_prompt, case.problem)
    accepted: list[int] = []
    cand_total, rnd = 0, 0
    T_prop = T_sel = T_pre = 0.0
    t_first_accept = None
    cands_out, decs_out, rounds_out = [], [], []
    status = None
    sel_tokens = 0
    processed = 0
    dup_rounds = 0
    over_total = 0
    while status is None:
        if rnd >= max_rounds:
            status = "max_rounds"
            break
        limit = min(mode_limit, L["max_output_tokens"] - len(accepted),
                    (L["hybrid_candidate_budget"] - cand_total) // n,
                    L["gb_context"] - len(ids_prompt) - len(accepted))
        if limit <= 0:
            status = "truncated"
            break
        rrs, seeds, t0, t1 = _round(ctx, case, ids_prompt, accepted, rnd, limit, step_mode, deadline)
        T_prop += t1 - t0
        firsts = [r.t_first for r in rrs if r.t_first]
        if firsts:
            T_pre += min(firsts) - t0
        processed += n * (len(ids_prompt) + len(accepted))
        if any(r.finish_reason == "timeout" for r in rrs):
            status = "timeout"
            break
        cs = [finalize_candidate(ctx, accepted, r, mode_limit, step_mode) for r in rrs]
        round_sampled = sum(c["n_sampled"] for c in cs)
        cand_total += round_sampled
        over_total += sum(c["overproduction_tokens"] for c in cs)
        texts = [c["text"] for c in cs]
        n_unique = len(set(texts))
        dup_rounds += int(n_unique < n)
        perm = permutation(n, option_order_seed(case.seed, case.problem_id, mode, rnd, case.generation))
        prefix_text = _decode(ctx.tok, accepted) if mode != "JFINAL" else ""
        crit = ctx.crit_final if mode == "JFINAL" else ctx.crit_step
        ts0 = time.perf_counter()
        try:
            dec = ctx.selector.decide(case.problem, prefix_text, texts, crit, perm, deadline)
        except TimeoutError:
            T_sel += time.perf_counter() - ts0
            status = "timeout"
            break
        T_sel += time.perf_counter() - ts0
        sel_tokens += dec["input_tokens"]
        for b, c in enumerate(cs):
            rec = {**_base_run(case, ctx), "round": rnd, "branch": b, "seed_branch": seeds[b],
                   "limit": limit, "shown_position": perm.index(b), **c}
            rec.pop("kept_ids")
            rec["chosen"] = dec.get("winner_branch") == b
            cands_out.append(rec)
        decs_out.append({**_base_run(case, ctx), "round": rnd,
                         "perm_seed": option_order_seed(case.seed, case.problem_id, mode, rnd, case.generation),
                         "prefix_tokens": len(accepted), "n_unique_candidates": n_unique,
                         **{k: v for k, v in dec.items()}})
        if dec["status"] != "ok":
            status = dec["status"]
            break
        w = dec["winner_branch"]
        accepted = accepted + cs[w]["kept_ids"]
        if t_first_accept is None:
            t_first_accept = time.perf_counter()
        rounds_out.append({**_base_run(case, ctx), "round": rnd, "limit": limit,
                           "round_sampled": round_sampled, "accepted_after": len(accepted),
                           "candidates_total_after": cand_total, "T_proposals": t1 - t0,
                           "T_selector": dec.get("wall_s"), "winner_branch": w,
                           "winner_terminal": cs[w]["terminal"]})
        if cs[w]["terminal"]:
            status = "final" if cs[w]["has_final"] else "eos_invalid"
            break
        if mode == "JFINAL":
            status = "truncated"
            break
        rnd += 1
    times = {"T_proposals": T_prop, "T_selector": T_sel, "T_prefill": T_pre, "T_cache_sync": 0.0}
    counters = {"prompt_tokens": len(ids_prompt), "candidate_tokens_total": cand_total,
                "overproduction_tokens": over_total, "rounds": len(rounds_out),
                "selector_calls": len(decs_out), "selector_input_tokens": sel_tokens,
                "processed_prompt_tokens": processed, "duplicate_rounds": dup_rounds,
                "tokenizer": "G",
                "empty_steps": sum(1 for c in cands_out if c.get("empty_step")),
                "forced_boundaries": sum(1 for c in cands_out if c.get("forced_boundary")),
                "boundary_overshoots": sum(1 for c in cands_out if c.get("boundary_overshoot"))}
    _finish(run, ctx, accepted, None, status, t_start, t_first_accept, times, counters)
    run["discarded_fraction"] = ((cand_total - len(accepted)) / cand_total) if cand_total else None
    return Outcome(run, cands_out, decs_out, rounds_out)


def run_case(case: Case, ctx: Ctx, timeout_s: float) -> Outcome:
    t_start = time.perf_counter()
    deadline = t_start + timeout_s
    if case.condition in ("G_SINGLE", "B13", "B13_GREEDY"):
        out = run_single(case, ctx, t_start, deadline)
    else:
        out = run_hybrid(case, ctx, t_start, deadline)
    out.run["t_start_perf"] = t_start
    return out
