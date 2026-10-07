"""Prompt construction for G/B (generators) and J (selector).

G and B receive the same instruction content through their own published chat templates
(system = protocol prompt, user = the problem statement only), rendered with
enable_thinking=False (§5.1). J's prompt is built exactly as the JevK5 runtime does.
"""

from __future__ import annotations

import json
from pathlib import Path

THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generator_messages(system_prompt: str, problem: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}]


def load_tokenizer(model_dir: str):
    """AutoTokenizer from a local snapshot. The published chat_template.jinja is attached
    explicitly so the rendering does not depend on tokenizer_config defaults."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tpl = Path(model_dir) / "chat_template.jinja"
    if tpl.exists():
        tok.chat_template = tpl.read_text(encoding="utf-8")
    return tok


def load_fast_tokenizer_file(model_dir: str):
    """Fallback that only needs tokenizer.json (used locally when the installed
    transformers does not know the model class)."""
    from transformers import PreTrainedTokenizerFast

    cfg = json.loads((Path(model_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_dir) / "tokenizer.json"),
                                  eos_token=cfg.get("eos_token"), pad_token=cfg.get("pad_token"))
    tok.chat_template = (Path(model_dir) / "chat_template.jinja").read_text(encoding="utf-8")
    return tok


def render_generator_prompt(tok, system_prompt: str, problem: str) -> str:
    text = tok.apply_chat_template(generator_messages(system_prompt, problem), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("chat template did not render the non-thinking assistant prefix")
    return text


def generator_prompt_ids(tok, system_prompt: str, problem: str) -> tuple[list[int], str]:
    text = render_generator_prompt(tok, system_prompt, problem)
    ids = tok.encode(text, add_special_tokens=False)
    return ids, text


# ---------------------------------------------------------------------------------------
# Selector (JevK5) prompt: exactly jevk5.runtime.JevK5.encode.
# ---------------------------------------------------------------------------------------

def selector_question(criterion: str, option_texts: list[str]) -> dict:
    """`choice` question with neutral ids option_0..option_{n-1} (protocol §6)."""
    return {"type": "choice", "instructions": criterion,
            "criteria": {f"option_{i}": t for i, t in enumerate(option_texts)}}


def selector_state(problem: str, accepted_prefix: str) -> dict:
    return {"problem": problem, "accepted_prefix": accepted_prefix}


def selector_prompt_ids(j_tok, state: dict, question: dict) -> tuple[list[int], str]:
    """Same as JevK5.encode(state, question['instructions'], [text for _, text in options])."""
    from jevk5.prompt import decision_options, messages

    options = decision_options(question)
    msgs = messages(state, question["instructions"], [text for _, text in options])
    text = j_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = j_tok.encode(text, add_special_tokens=False)
    return ids, text
