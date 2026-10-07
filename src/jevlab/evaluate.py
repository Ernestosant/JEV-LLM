"""Offline grading (protocol §11): runs only after predictions are saved; never online."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .common import read_jsonl
from .parsing import extract_final

FAIL_STATUS = {"truncated": "truncated", "max_rounds": "truncated", "timeout": "timeout",
               "eos_invalid": "eos_without_final", "selector_context_limit": "selector_context_limit",
               "exception": "exception"}


def load_gold(path) -> dict:
    return {g["id"]: g for g in read_jsonl(path)}


def grade_text(public_output: str, gold: dict) -> dict:
    st, raw, v = extract_final(public_output or "")
    gv = Fraction(gold["gold_numerator"], gold["gold_denominator"])
    correct = st == "ok" and v == gv
    return {"format_status": st, "answer_raw": raw,
            "answer_normalized": None if v is None else (str(v.numerator) if v.denominator == 1
                                                         else f"{v.numerator}/{v.denominator}"),
            "valid_format": st == "ok", "correct": bool(correct)}


def grade_prediction(pred: dict, gold: dict) -> dict:
    g = grade_text(pred.get("public_output", ""), gold)
    status = pred.get("status")
    if status != "final":
        category = FAIL_STATUS.get(status, status or "unknown")
        g["correct"] = False
    elif not g["valid_format"]:
        category = g["format_status"]
    else:
        category = "correct" if g["correct"] else "wrong_answer"
    return {"problem_id": pred["problem_id"], "seed": pred["seed"], "condition": pred["condition"],
            "profile": pred.get("profile"), "run_name": pred.get("run_name"), "status": status,
            "category": category, "domain": gold["domain"], "difficulty": gold["difficulty"],
            "template_group": gold.get("template_group"), "gold_answer": gold["gold_answer"], **g}


def collect(results_root) -> dict:
    """Find every run directory (…/<COND>/<run_name>/predictions.jsonl)."""
    out = {}
    for p in Path(results_root).rglob("predictions.jsonl"):
        run_dir = p.parent
        out[str(run_dir)] = {"predictions": read_jsonl(p),
                             "metrics": read_jsonl(run_dir / "metrics.jsonl"),
                             "candidates": read_jsonl(run_dir / "candidates.jsonl"),
                             "decisions": read_jsonl(run_dir / "decisions.jsonl"),
                             "failures": read_jsonl(run_dir / "failures.jsonl"),
                             "manifest": (run_dir / "manifest.json")}
    return out
