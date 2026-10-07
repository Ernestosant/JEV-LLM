import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"


def _jl(name):
    return [json.loads(x) for x in (D / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def test_sizes_and_quota():
    ti, tg, di, dg = _jl("test_inputs.jsonl"), _jl("test_gold.jsonl"), _jl("dev_inputs.jsonl"), _jl("dev_gold.jsonl")
    assert len(ti) == len(tg) == 100 and len(di) == len(dg) == 20
    c = Counter((g["domain"], g["difficulty"]) for g in tg)
    for dom in ("arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"):
        assert c[(dom, "easy")] == 8 and c[(dom, "medium")] == 8 and c[(dom, "hard")] == 4


def test_inputs_have_no_gold_fields():
    for r in _jl("test_inputs.jsonl") + _jl("dev_inputs.jsonl"):
        assert set(r) == {"id", "problem", "language"} and r["language"] == "en"


def test_disjoint_and_unique():
    t = {g["template_group"] for g in _jl("test_gold.jsonl")}
    d = {g["template_group"] for g in _jl("dev_gold.jsonl")}
    assert not (t & d)
    probs = [r["problem"] for r in _jl("test_inputs.jsonl") + _jl("dev_inputs.jsonl")]
    assert len(set(probs)) == len(probs)


def test_gold_exact_and_bounded():
    for g in _jl("test_gold.jsonl") + _jl("dev_gold.jsonl"):
        v = Fraction(g["gold_numerator"], g["gold_denominator"])
        assert g["gold_denominator"] > 0 and abs(v) <= 999
        assert max(g["prompt_tokens"].values()) <= 512
        assert g["reviewed"] is False


def test_schedule_covers_test_once():
    s = json.loads((D / "schedule.json").read_text())
    ids = {r["id"] for r in _jl("test_inputs.jsonl")}
    assert sorted(s["order"]) == sorted(ids) and len(s["blocks"]) == 10
    gold = {g["id"]: g["domain"] for g in _jl("test_gold.jsonl")}
    for b in s["blocks"]:
        assert len(b) == 10 and len({gold[i] for i in b}) == 5  # balanced by domain


def test_sha256sums():
    import hashlib

    for line in (D / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        assert hashlib.sha256((D / name.strip()).read_bytes()).hexdigest() == h, name
