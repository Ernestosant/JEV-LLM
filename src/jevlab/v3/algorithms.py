"""Independent v3 reference runs and durable four-proposal final selection."""

from __future__ import annotations

import time
import traceback

from ..algorithms import Case, Ctx, Outcome, _decode, _finish, finalize_candidate
from ..common import config_hash
from ..engines import Req, sampling_params
from ..parsing import extract_final
from ..prompts import generator_prompt_ids
from ..seeds import branch_seed, option_order_seed, permutation

CONDITIONS = ("G_SINGLE", "G_GREEDY", "JFINAL", "B13", "B13_GREEDY", "Q9_GREEDY")
GREEDY = ("G_GREEDY", "B13_GREEDY", "Q9_GREEDY")


def run_case(case: Case, ctx: Ctx, timeout_s: float, *, proposal_callback=None,
             provenance=None) -> Outcome:
    """The callback durably persists one four-candidate pool before any selection."""
    if case.condition not in CONDITIONS:
        raise ValueError(f"unsupported v3 condition: {case.condition}")
    if ctx.limits["n_candidates"] != 4:
        raise ValueError("v3 requires n_candidates=4")
    start = time.perf_counter()
    deadline = start + timeout_s
    hybrid = case.condition == "JFINAL"
    greedy = case.condition in GREEDY
    provenance_hash = config_hash({} if provenance is None else provenance)
    base = dict(problem_id=case.problem_id, seed=case.seed, condition=case.condition,
                generation=case.generation, profile=ctx.profile,
                provenance_hash=provenance_hash, effective_seed=None if greedy else case.seed,
                clock_scope="candidate_generation_inclusive")
    seed_condition = "G4_FINAL4_V3" if hybrid else case.condition
    seeds = [branch_seed(case.seed, case.problem_id, seed_condition, 0, b,
                         case.generation) for b in range(4 if hybrid else 1)]
    identity = dict(problem_id=case.problem_id, problem=case.problem,
                    provenance_hash=provenance_hash)
    base["candidate_set_id"] = config_hash(dict(
        identity, proposal_seeds=[None] if greedy else seeds,
        proposal_condition=seed_condition))
    if not hybrid:
        # Greedy references identify the problem/configuration, not a latency repetition.
        ref = dict(identity, condition=case.condition)
        if not greedy:
            ref["proposal_seeds"] = seeds
        base["reference_id"] = config_hash(ref)
    run = dict(base, t_start_perf=start)
    out = Outcome(run)
    perm_seed = option_order_seed(case.seed, case.problem_id, "JFINAL", 0,
                                  case.generation) if hybrid else None
    perm = permutation(4, perm_seed) if hybrid else None
    prompt, _ = generator_prompt_ids(ctx.tok, ctx.system_prompt, case.problem)
    limit = min(ctx.limits["max_output_tokens"], ctx.limits["gb_context"] - len(prompt))
    if hybrid:
        limit = min(limit, ctx.limits["hybrid_candidate_budget"] // 4)
    times = dict(T_proposals=0.0, T_selector=0.0, T_prefill=None, T_cache_sync=0.0,
                 T_proposal_persist=0.0)
    counters = dict(prompt_tokens=len(prompt), candidate_tokens_total=0,
                    overproduction_tokens=0, rounds=0, selector_calls=0,
                    selector_input_tokens=0, processed_prompt_tokens=0,
                    tokenizer="B" if case.condition.startswith("B13") else
                    "O" if case.condition == "Q9_GREEDY" else "G")
    accepted = []
    first_accept = None
    status = "truncated"
    if limit > 0:
        reqs = []
        s = ctx.sampling
        for b, seed in enumerate(seeds):
            params = sampling_params(
                temperature=0.0 if greedy else s["temperature"],
                top_p=1.0 if greedy else s["top_p"],
                top_k=0 if greedy else s["top_k"],
                max_tokens=limit, seed=None if greedy else seed,
                stop_token_ids=list(ctx.eos_ids),
                repetition_penalty=s["repetition_penalty"],
                presence_penalty=s["presence_penalty"],
                frequency_penalty=s["frequency_penalty"])
            reqs.append(Req(ctx.gen.new_id(f"{case.condition}-b{b}"), prompt, params))
        t0 = time.perf_counter()
        error = None
        try:
            results = ctx.gen.run(reqs, deadline)
        except Exception as exc:
            # Engine.run exposes no partial results on exceptions. Never invent them.
            results = {}
            error = exc
            run.update(error_traceback=traceback.format_exc(),
                       engine_cleanup=getattr(exc, "engine_cleanup", None),
                       engine_usable=getattr(ctx.gen, "usable", None))
        times["T_proposals"] = time.perf_counter() - t0
        counters["processed_prompt_tokens"] = len(reqs) * len(prompt)
        for b, req in enumerate(reqs):
            if req.rid not in results:
                continue
            rr = results[req.rid]
            c = finalize_candidate(ctx, [], rr, None, False)
            cstatus = ("timeout" if rr.finish_reason == "timeout" else
                       "final" if c["terminal"] and c["has_final"] else
                       "eos_invalid" if c["terminal"] else "truncated")
            row = dict(base, round=0, branch=b, seed_branch=None if greedy else seeds[b],
                       limit=limit, request_id=req.rid, token_ids=list(rr.token_ids), **c)
            if hybrid:
                row["shown_position"] = perm.index(b)
            _finish(row, ctx, c["kept_ids"], None, cstatus, start, rr.t_first, {}, {})
            fmt, answer, value = extract_final(row["public_output"])
            row.update(format_status=fmt, answer_raw=answer,
                       answer_value=str(value) if value is not None else None,
                       eligible=cstatus == "final" and fmt == "ok")
            out.candidates.append(row)
            counters["candidate_tokens_total"] += c["n_sampled"]
            counters["overproduction_tokens"] += c["overproduction_tokens"]
        if hybrid and proposal_callback is not None:
            t_persist = time.perf_counter()
            partial = error is not None or len(out.candidates) != 4 or any(
                c["status"] == "timeout" for c in out.candidates)
            proposal_callback({**base, "status": "no_candidates" if not out.candidates else
                               "partial" if partial else "complete", "planned_candidate_slots": 4,
                               "candidates": [dict(c) for c in out.candidates],
                               "seed_branches": seeds, "proposal_namespace": "G4_FINAL4_V3",
                               "error": str(error) if error is not None else None})
            times["T_proposal_persist"] = time.perf_counter() - t_persist
        counters["rounds"] = 1
        firsts = [c["t_first"] for c in out.candidates if c["t_first"] is not None]
        times["T_prefill"] = min(firsts) - t0 if firsts else None
        if error is not None or len(out.candidates) != len(reqs):
            status = "timeout" if isinstance(error, TimeoutError) else "engine_error"
            run["error"] = str(error) if error else "engine returned incomplete results"
        elif any(c["status"] == "timeout" for c in out.candidates):
            status = "timeout"
        elif not hybrid:
            chosen = out.candidates[0]
            accepted, status = chosen["kept_ids"], chosen["status"]
            first_accept = chosen["t_first"]
            chosen["chosen"] = True
        else:
            decision = dict(base, round=0, perm_seed=perm_seed, permutation=perm,
                            prefix_tokens=0, n_unique_candidates=len(set(
                                c["public_output"] for c in out.candidates)))
            counters["selector_calls"] = 1
            ts = time.perf_counter()
            try:
                dec = ctx.selector.decide(case.problem, "", [c["public_output"]
                    for c in out.candidates], ctx.crit_final, perm, deadline)
                decision.update(dec)
                winner = dec.get("winner_branch")
                if dec["status"] == "ok" and (type(winner) is not int or winner not in range(4)):
                    raise ValueError("selector returned invalid winner_branch")
            except Exception as exc:
                decision.update(status="timeout" if isinstance(exc, TimeoutError)
                                else "selector_error", error=str(exc),
                                error_traceback=traceback.format_exc(),
                                engine_cleanup=getattr(exc, "engine_cleanup", None),
                                engine_usable=getattr(getattr(ctx.selector, "engine", None), "usable", None))
                decision.pop("winner_branch", None)
                decision.pop("winner_position", None)
            times["T_selector"] = time.perf_counter() - ts
            decision.setdefault("wall_s", times["T_selector"])
            if decision["status"] != "ok":
                decision.pop("winner_branch", None)
                decision.pop("winner_position", None)
            out.decisions.append(decision)
            counters["selector_input_tokens"] = decision.get("input_tokens", 0)
            status = decision["status"]
            if "error" in decision:
                run["error"] = decision["error"]
                for field in ("error_traceback", "engine_cleanup", "engine_usable"):
                    if field in decision:
                        run[field] = decision[field]
            if status == "ok":
                chosen = out.candidates[decision["winner_branch"]]
                accepted, status = chosen["kept_ids"], chosen["status"]
                first_accept = time.perf_counter()
                run["winner_branch"] = chosen["branch"]
                for c in out.candidates:
                    c["chosen"] = c["branch"] == chosen["branch"]
        if hybrid:
            out.rounds.append(dict(base, round=0, limit=limit, status=status,
                                   round_sampled=counters["candidate_tokens_total"],
                                   accepted_after=len(accepted),
                                   candidates_total_after=counters["candidate_tokens_total"],
                                   winner_branch=run.get("winner_branch"), **times))
    counters["discarded_tokens"] = counters["candidate_tokens_total"] - len(accepted)
    if hybrid and limit <= 0 and proposal_callback is not None:
        t_persist = time.perf_counter()
        proposal_callback({**base, "status": "no_candidates", "planned_candidate_slots": 4,
                           "candidates": [], "seed_branches": seeds,
                           "proposal_namespace": "G4_FINAL4_V3", "error": "no available context/budget"})
        times["T_proposal_persist"] = time.perf_counter() - t_persist
    _finish(run, ctx, accepted, None, status, start, first_accept, times, counters)
    total = counters["candidate_tokens_total"]
    run["discarded_fraction"] = counters["discarded_tokens"] / total if total else None
    return out
