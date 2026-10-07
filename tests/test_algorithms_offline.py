"""Candidate finalisation with the real G tokenizer (CPU, no vLLM)."""

from pathlib import Path

import pytest

from jevlab.algorithms import Ctx, finalize_candidate
from jevlab.engines import ReqResult
from jevlab.prompts import THINK_OFF_SUFFIX, load_fast_tokenizer_file, render_generator_prompt

META = Path(__file__).resolve().parents[1] / "data" / "raw" / "model_meta"


@pytest.fixture(scope="module")
def tok():
    if not (META / "G" / "tokenizer.json").exists():
        pytest.skip("model meta not downloaded")
    return load_fast_tokenizer_file(str(META / "G"))


@pytest.fixture(scope="module")
def ctx(tok):
    nl = set()
    for i in range(tok.vocab_size):
        s = tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "\n" in s:
            nl.add(i)
    return Ctx(gen=None, tok=tok, system_prompt="", sampling={}, limits={}, eos_ids=[], newline_ids=nl)


def _rr(ids, finish="length", eos=False, stop=None):
    r = ReqResult(rid="x", prompt_len=0, token_ids=ids, finish_reason=finish, eos=eos, stop_reason=stop)
    return r


def test_templates_render_non_thinking():
    for role in ("G", "B"):
        t = load_fast_tokenizer_file(str(META / role))
        txt = render_generator_prompt(t, "SYS", "What is 2+2?")
        assert txt.endswith(THINK_OFF_SUFFIX)
        assert "<|im_start|>system\nSYS<|im_end|>" in txt and "What is 2+2?" in txt


def test_final_cut_and_overproduction(ctx, tok):
    ids = tok.encode("2+2=4\nFINAL: 4\nThanks, more text here", add_special_tokens=False)
    c = finalize_candidate(ctx, [], _rr(ids), None, False)
    assert c["terminal"] and c["has_final"]
    assert c["text"].endswith("FINAL: 4\n") or "FINAL: 4\n" in c["text"]
    assert c["overproduction_tokens"] > 0
    assert c["n_sampled"] == len(ids)


def test_eos_without_final_is_terminal_invalid(ctx, tok):
    ids = tok.encode("I do not know", add_special_tokens=False)
    c = finalize_candidate(ctx, [], _rr(ids, finish="stop", eos=True), None, False)
    assert c["terminal"] and not c["has_final"] and c["n_sampled"] == len(ids) + 1


def test_final_across_blocks(ctx, tok):
    acc = tok.encode("x=3\nFINAL: 1", add_special_tokens=False)
    ids = tok.encode("2\n", add_special_tokens=False)
    c = finalize_candidate(ctx, acc, _rr(ids), 64, False)
    assert c["terminal"] and c["has_final"]


def test_step_flags(ctx, tok):
    ids = tok.encode("First we add.\n", add_special_tokens=False)
    c = finalize_candidate(ctx, [], _rr(ids, finish="stop", stop=ids[-1]), 128, True)
    assert c["boundary"] and not c["forced_boundary"] and not c["empty_step"]
    nl = tok.encode("\n", add_special_tokens=False)
    e = finalize_candidate(ctx, [], _rr(nl, finish="stop", stop=nl[-1]), 128, True)
    assert e["empty_step"]
    long = tok.encode("a " * 200, add_special_tokens=False)[:128]
    f = finalize_candidate(ctx, [], _rr(long, finish="length"), 128, True)
    assert f["forced_boundary"]
