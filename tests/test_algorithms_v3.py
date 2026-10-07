"""v3 contract checks with observable fake results only, no GPU dependencies."""

import json
import sys

import pytest

from jevlab.algorithms import Case, Ctx, Outcome
from jevlab.common import append_jsonl, config_hash
from jevlab.engines import ReqResult
from jevlab.parsing import extract_final
from jevlab.seeds import branch_seed, option_order_seed, permutation
from jevlab.v3 import algorithms as a


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) if i != 1000 else "\nextra" for i in ids)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.outputs = [("FINAL: 4\n", "final_abort", False)] * 4
        self.error = None
        self.n = 0

    def new_id(self, tag):
        self.n += 1
        return f"{tag}-{self.n}"

    def run(self, reqs, deadline):
        self.calls.append(reqs)
        if self.error:
            raise self.error
        return {r.rid: ReqResult(r.rid, len(r.prompt_ids),
                    list(text) if isinstance(text, list) else list(map(ord, text)),
                    finish_reason=finish, eos=eos, t_add=1.0, t_first=2.0, t_end=3.0)
                for r, (text, finish, eos) in zip(reqs, self.outputs)}


class FakeSelector:
    def __init__(self):
        self.calls = []
        self.error = None
        self.check = lambda: None
        self.winner = 0

    def decide(self, *args):
        self.check()
        self.calls.append(args)
        if self.error:
            raise self.error
        return dict(status="ok", winner_branch=self.winner, input_tokens=23)


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setattr(a, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(a, "generator_prompt_ids", lambda *args: ([1, 2], "prompt"))
    return Ctx(FakeEngine(), Tokenizer(), "sys",
               dict(temperature=.7, top_p=.8, top_k=40, repetition_penalty=1.,
                    presence_penalty=0., frequency_penalty=0.),
               dict(n_candidates=4, max_output_tokens=2048,
                    hybrid_candidate_budget=8192, gb_context=16384),
               [999], {10, 1000}, FakeSelector(), crit_final="criterion")


def case(condition="JFINAL", seed=42, generation=0):
    return Case("p1", "Compute 2+2", seed, condition, generation)


@pytest.mark.parametrize("condition", a.CONDITIONS)
def test_dispatch_and_params(ctx, condition):
    out = a.run_case(case(condition), ctx, 60)
    assert isinstance(out, Outcome)
    assert len(ctx.gen.calls) == 1
    reqs = ctx.gen.calls[0]
    assert len(reqs) == (4 if condition == "JFINAL" else 1)
    assert len(ctx.selector.calls) == (1 if condition == "JFINAL" else 0)
    for req in reqs:
        assert req.params["max_tokens"] == 2048
        if condition in a.GREEDY:
            assert {k: req.params[k] for k in ("temperature", "seed", "top_p", "top_k")} == {
                "temperature": 0., "seed": None, "top_p": 1., "top_k": 0}
        else:
            assert req.params["temperature"] == .7
    assert out.run["public_output"] == out.candidates[0]["public_output"]
    assert all(c["candidate_set_id"] == out.run["candidate_set_id"] for c in out.candidates)


def test_seeds_provenance_and_independent_single(ctx):
    p = {"revision": "abc", "sampling": {"top_p": .8}, "prompt_hash": "xyz"}
    out = a.run_case(case(generation=3), ctx, 60, provenance=p)
    seeds = [branch_seed(42, "p1", "G4_FINAL4_V3", 0, b, 3) for b in range(4)]
    assert [c["seed_branch"] for c in out.candidates] == seeds
    assert [r.params["seed"] for r in ctx.gen.calls[0]] == seeds
    assert out.run["provenance_hash"] == config_hash(p)
    again = a.run_case(case(generation=3), ctx, 60, provenance=dict(reversed(list(p.items()))))
    assert again.run["candidate_set_id"] == out.run["candidate_set_id"]
    assert a.run_case(case(generation=4), ctx, 60, provenance=p).run["candidate_set_id"] != out.run["candidate_set_id"]
    assert a.run_case(case(generation=3), ctx, 60, provenance={}).run["candidate_set_id"] != out.run["candidate_set_id"]
    single = a.run_case(case("G_SINGLE", generation=3), ctx, 60, provenance=p)
    assert single.candidates[0]["seed_branch"] == branch_seed(42, "p1", "G_SINGLE", 0, 0, 3)
    assert single.candidates[0]["seed_branch"] != seeds[0]
    ps = option_order_seed(42, "p1", "JFINAL", 0, 3)
    assert out.decisions[0]["perm_seed"] == ps
    assert out.decisions[0]["permutation"] == permutation(4, ps)


def test_durable_callback_before_selector(ctx, tmp_path):
    path = tmp_path / "proposals.jsonl"
    def persist(row):
        append_jsonl(path, row)
    def check():
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["status"] == "complete"
        assert [r["branch"] for r in rows[0]["candidates"]] == list(range(4))
        assert all(r["status"] == "final" for r in rows[0]["candidates"])
    ctx.selector.check = check
    a.run_case(case(), ctx, 60, proposal_callback=persist)
    check()


def test_callback_failure_propagates(ctx):
    def fail(row):
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        a.run_case(case(), ctx, 60, proposal_callback=fail)
    assert not ctx.selector.calls


def test_invalid_selector_winner_is_failure(ctx):
    ctx.selector.winner = 7
    out = a.run_case(case(), ctx, 60)
    assert out.run["status"] == "selector_error"
    assert len(out.candidates) == 4
    assert "winner_branch" not in out.decisions[0]
    assert out.run["public_output"] == ""


def test_generation_timeout_keeps_four_partials(ctx):
    ctx.gen.outputs = [("FINAL: 4\n", "timeout", False), ("partial", "timeout", False),
                       ("FINAL: 2\n", "final_abort", False), ("", "timeout", False)]
    saved = []
    out = a.run_case(case(), ctx, 60, proposal_callback=saved.append)
    assert len(out.candidates) == len(saved[0]["candidates"]) == 4
    assert len(saved) == 1 and saved[0]["status"] == "partial"
    assert out.candidates[0]["status"] == "timeout"
    assert not out.candidates[0]["eligible"]
    assert out.candidates[1]["raw_output"] == "partial"
    assert out.run["status"] == "timeout"
    assert out.run["public_output"] == ""
    assert out.run["accepted_tokens"] == 0
    assert out.run["discarded_tokens"] == out.run["candidate_tokens_total"]
    assert not ctx.selector.calls


@pytest.mark.parametrize("error,status", [(TimeoutError("late"), "timeout"),
                                          (RuntimeError("broken"), "selector_error")])
def test_selector_failure_preserves_candidates(ctx, error, status):
    ctx.selector.error = error
    saved = []
    out = a.run_case(case(), ctx, 60, proposal_callback=saved.append)
    assert len(out.candidates) == len(saved[0]["candidates"]) == 4
    assert len(saved) == 1 and saved[0]["status"] == "complete"
    assert out.run["status"] == out.decisions[0]["status"] == status
    assert out.decisions[0]["error"] == str(error)
    assert out.run["public_output"] == out.run["raw_output"] == ""
    assert "winner_branch" not in out.decisions[0]
    assert not any(c.get("chosen") for c in out.candidates)
    assert out.run["selector_calls"] == 1


@pytest.mark.parametrize("text,finish,eos,status,public,eligible", [
    ("FINAL: 4", "stop", True, "final", "FINAL: 4", True),
    ("FINAL: 4", "length", False, "truncated", "FINAL: 4", False),
    ("no answer", "stop", True, "eos_invalid", "no answer", False),
    ("FINAL: 4 units\n", "final_abort", False, "final", "FINAL: 4 units\n", False),
    ("FINAL: 1/0\n", "final_abort", False, "final", "FINAL: 1/0\n", False),
    ("FINAL: .5\nextra", "final_abort", False, "final", "FINAL: .5\n", True),
])
def test_terminal_and_strict_rules(ctx, text, finish, eos, status, public, eligible):
    ctx.gen.outputs = [(text, finish, eos)] * 4
    out = a.run_case(case(), ctx, 60)
    c = out.candidates[0]
    assert c["status"] == out.run["status"] == status
    assert c["public_output"] == out.run["public_output"] == public
    assert c["raw_text"] == text
    assert c["eligible"] is eligible
    assert c["n_sampled"] == len(text) + int(eos)
    assert c["format_status"] == extract_final(public)[0]


def test_token_overshoot_full_trace_and_chosen_output(ctx):
    ids = list(map(ord, "FINAL: 4")) + [1000] + list(map(ord, "tail"))
    ctx.gen.outputs = [(ids, "final_abort", False)] * 4
    ctx.selector.winner = 2
    out = a.run_case(case(), ctx, 60)
    c = out.candidates[2]
    assert c["token_ids"] == ids
    assert c["raw_text"] == "FINAL: 4\nextratail"
    assert c["raw_output"] == "FINAL: 4\nextra"
    assert c["public_output"] == out.run["public_output"] == "FINAL: 4\n"
    assert c["overproduction_tokens"] == 4
    assert c["overproduction_chars"] == 5
    assert out.run["candidate_tokens_total"] == 4 * len(ids)
    assert out.run["accepted_tokens"] == len(ids) - 4


@pytest.mark.parametrize("condition", a.GREEDY)
def test_reference_id_stable_no_gold(ctx, condition):
    first = a.run_case(case(condition), ctx, 60, provenance={"revision": "a"})
    second = a.run_case(case(condition, seed=999, generation=8), ctx, 60,
                        provenance={"revision": "a"})
    assert first.run["reference_id"] == second.run["reference_id"]
    assert a.run_case(case(condition), ctx, 60, provenance={"revision": "b"}).run["reference_id"] != first.run["reference_id"]
    assert all(c["seed_branch"] is None for c in first.candidates)
    assert "gold" not in json.dumps(first.run)


def test_invalid_dispatch_and_candidate_count(ctx):
    with pytest.raises(ValueError, match="unsupported"):
        a.run_case(case("JSTEP"), ctx, 60)
    ctx.limits["n_candidates"] = 3
    with pytest.raises(ValueError, match="n_candidates"):
        a.run_case(case(), ctx, 60)
    assert not ctx.gen.calls and not ctx.selector.calls


def test_engine_exception_does_not_invent_completions(ctx):
    ctx.gen.error = RuntimeError("engine died")
    saved = []
    out = a.run_case(case(), ctx, 60, proposal_callback=saved.append)
    assert out.run["status"] == "engine_error"
    assert not out.candidates and not ctx.selector.calls
    assert len(saved) == 1 and saved[0]["status"] == "no_candidates"
    assert saved[0]["candidates"] == [] and saved[0]["error"] == "engine died"
    assert out.run["public_output"] == ""


@pytest.mark.parametrize("confirmed", [True, False])
def test_engine_failure_retains_cleanup_and_trace_without_quality_retry(ctx, confirmed):
    error = RuntimeError("accepted then failed")
    error.engine_cleanup = {"attempted_ids": ["one", "two"], "cleanup_confirmed": confirmed,
                            "unfinished": False if confirmed else None}
    ctx.gen.error, ctx.gen.usable = error, confirmed
    out = a.run_case(case(), ctx, 60)
    assert out.run["status"] == "engine_error" and not out.candidates
    assert out.run["engine_cleanup"] == error.engine_cleanup
    assert out.run["engine_usable"] is confirmed and "accepted then failed" in out.run["error_traceback"]
    assert len(ctx.gen.calls) == 1 and not ctx.selector.calls


def test_budget_and_context_caps(ctx):
    ctx.limits["hybrid_candidate_budget"] = 400
    a.run_case(case(), ctx, 60)
    assert all(r.params["max_tokens"] == 100 for r in ctx.gen.calls[-1])
    ctx.limits["gb_context"] = 12
    a.run_case(case(), ctx, 60)
    assert all(r.params["max_tokens"] == 10 for r in ctx.gen.calls[-1])
    ctx.limits["gb_context"] = 2
    before = len(ctx.gen.calls)
    out = a.run_case(case(), ctx, 60)
    assert len(ctx.gen.calls) == before
    assert out.run["status"] == "truncated" and not out.candidates


def test_generation_zero_repeats_and_problem_identity(ctx):
    first = a.run_case(case(), ctx, 60)
    second = a.run_case(case(), ctx, 10)
    assert first.run["candidate_set_id"] == second.run["candidate_set_id"]
    assert [c["seed_branch"] for c in first.candidates] == [c["seed_branch"] for c in second.candidates]
    changed = Case("p1", "Different question", 42, "JFINAL")
    assert a.run_case(changed, ctx, 60).run["candidate_set_id"] != first.run["candidate_set_id"]


def test_incomplete_engine_results_preserve_only_observed(ctx):
    ctx.gen.outputs = [("partial", "timeout", False)]
    saved = []
    out = a.run_case(case(), ctx, 60, proposal_callback=saved.append)
    assert len(saved) == len(out.candidates) == 1
    assert saved[0]["status"] == "partial" and len(saved[0]["candidates"]) == 1
    assert out.candidates[0]["raw_output"] == "partial"
    assert out.run["status"] == "engine_error"
    assert not ctx.selector.calls


def test_no_vllm_import():
    assert "vllm" not in sys.modules
