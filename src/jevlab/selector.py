"""JevK5-9B selector on vLLM (protocol §6, amendment v3.1 §A3).

The prompt token ids are built exactly like jevk5.runtime.JevK5.encode (native prompt,
thinking off). One prefill; the next-token *raw logits* of the option letters A..D are read
through vLLM (`logprobs_mode="raw_logits"` + `logprob_token_ids`), divided by the published
calibration temperature and softmaxed. The decision is the argmax (ties -> first position in
the permuted order). The single sampled token is discarded: no letter is "generated" as the
decision, no reasoning is requested.
"""

from __future__ import annotations

import logging
import math
import time

from .engines import Engine, Req
from .prompts import selector_prompt_ids, selector_question, selector_state

log = logging.getLogger("jevlab.selector")
LETTERS = "ABCDEFGHIJKLMNOP"


class Selector:
    def __init__(self, engine: Engine, j_tok, temperature: float, max_input: int = 16384,
                 readout: str = "logprob_token_ids"):
        self.engine, self.tok, self.temperature, self.max_input = engine, j_tok, temperature, max_input
        slots = [j_tok.encode(ch, add_special_tokens=False) for ch in LETTERS]
        if any(len(s) != 1 for s in slots):
            raise ValueError("every answer letter must be a single token")
        self.letter_ids = [s[0] for s in slots]
        self.readout = readout

    def _params(self, n: int):
        from vllm import SamplingParams

        if self.readout == "logprob_token_ids":
            return SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                                  logprob_token_ids=self.letter_ids[:n], seed=0)
        # fallback: full-vocabulary raw logits, pick the letters
        return SamplingParams(max_tokens=1, temperature=0.0, detokenize=False, logprobs=-1, seed=0)

    def letter_logits_from_ids(self, ids: list[int], n: int, deadline=None) -> tuple[list[float], float]:
        rid = self.engine.new_id("sel")
        t0 = time.perf_counter()
        res = self.engine.run([Req(rid, ids, self._params(n), watch_final=False)], deadline)[rid]
        dt = time.perf_counter() - t0
        if res.finish_reason == "timeout":
            raise TimeoutError("selector timeout")
        lp = res.logprobs[0] if res.logprobs else None
        if lp is None:
            raise RuntimeError("selector returned no logprobs")
        vals = []
        for t in self.letter_ids[:n]:
            if t not in lp:
                raise RuntimeError(f"letter token {t} missing from logprobs (readout={self.readout})")
            vals.append(float(lp[t].logprob))
        return vals, dt

    def probe(self) -> str:
        """Check that the chosen readout returns all four letter logits; otherwise fall back
        to the full-vocabulary raw logits (identical values, slower transfer)."""
        ids = self.tok.encode("Answer with a letter: A, B, C or D.", add_special_tokens=False)
        try:
            self.letter_logits_from_ids(ids, 4)
        except RuntimeError as e:
            if self.readout != "full_vocab":
                log.warning("readout %s failed (%s); falling back to full_vocab", self.readout, e)
                self.readout = "full_vocab"
                self.letter_logits_from_ids(ids, 4)
            else:
                raise
        return self.readout

    def probabilities(self, logits: list[float]) -> list[float]:
        z = [v / self.temperature for v in logits]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def decide(self, problem: str, prefix_text: str, cand_texts: list[str], criterion: str,
               perm: list[int], deadline=None) -> dict:
        """perm[pos] = original branch index displayed at option position pos."""
        shown = [cand_texts[b] for b in perm]
        state = selector_state(problem, prefix_text)
        q = selector_question(criterion, shown)
        t0 = time.perf_counter()
        ids, _ = selector_prompt_ids(self.tok, state, q)
        t_tok = time.perf_counter() - t0
        rec = {"input_tokens": len(ids), "permutation": perm,
               "option_chars": [len(s) for s in shown], "tokenize_s": t_tok}
        if len(ids) > self.max_input:
            rec.update(status="selector_context_limit", wall_s=time.perf_counter() - t0)
            return rec
        logits, dt = self.letter_logits_from_ids(ids, len(shown), deadline)
        probs = self.probabilities(logits)
        best = max(probs)
        pos = probs.index(best)  # first maximal position (ties -> earliest permuted slot)
        rec.update(status="ok", logits=logits, probabilities=probs, winner_position=pos,
                   winner_branch=perm[pos], exact_tie=probs.count(best) > 1,
                   engine_s=dt, wall_s=time.perf_counter() - t0)
        return rec
