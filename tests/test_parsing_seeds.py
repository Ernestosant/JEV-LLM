from fractions import Fraction

import pytest

from jevlab.parsing import extract_final, parse_latex_gold, parse_strict_answer, terminal_state
from jevlab.seeds import branch_seed, derive, option_order_seed, permutation


@pytest.mark.parametrize("s,want", [
    ("1/2", Fraction(1, 2)), ("0.5", Fraction(1, 2)), ("2/4", Fraction(1, 2)), (" 7 ", Fraction(7)),
    ("-3", Fraction(-3)), ("\u22123", Fraction(-3)), (".25", Fraction(1, 4)), ("+4", Fraction(4)),
    ("3.", Fraction(3)), ("-1/3", Fraction(-1, 3)),
    ("2+2", None), ("1/0", None), ("5 apples", None), ("$5", None), ("50%", None), ("1,200", None),
    ("\\frac{1}{2}", None), ("", None), ("x", None)])
def test_strict(s, want):
    assert parse_strict_answer(s) == want


@pytest.mark.parametrize("s,want", [
    ("\\frac{3}{4}", Fraction(3, 4)), ("\\dfrac{3}{4}", Fraction(3, 4)), ("-\\frac{1}{2}", Fraction(-1, 2)),
    ("\\frac12", Fraction(1, 2)), ("12", Fraction(12)), ("0.25", Fraction(1, 4)),
    ("\\sqrt{2}", None), ("2\\pi", None), ("5\\%", None), ("\\$5", None), ("1,000", None),
    ("x=3", None), ("(1,2)", None), ("10^\\circ", None), ("\\text{Yes}", None), ("3_5", None)])
def test_latex_gold(s, want):
    assert parse_latex_gold(s) == want


def test_extract_final():
    assert extract_final("step\nFINAL: 1/2\n")[2] == Fraction(1, 2)
    assert extract_final("step\nFINAL: 1/2")[0] == "ok"
    assert extract_final("no answer")[0] == "no_final"
    assert extract_final("FINAL: 2\nFINAL: 3\n")[0] == "multiple_final"
    assert extract_final("FINAL: two\n")[0] == "invalid_format"


def test_terminal_rules():
    assert not terminal_state("a\nFINAL:", eos=False).terminal         # mere 'FINAL:' is not enough
    assert not terminal_state("a\nFINAL: 3", eos=False).terminal       # line not finished
    t = terminal_state("a\nFINAL: 3\nextra", eos=False)
    assert t.terminal and t.has_final and "a\nFINAL: 3\n"[: t.end_char] == "a\nFINAL: 3\n"
    assert terminal_state("a\nFINAL: 3", eos=True).reason == "final_at_eos"
    e = terminal_state("a\nb", eos=True)
    assert e.terminal and not e.has_final and e.reason == "eos_no_final"
    assert not terminal_state("I think FINAL: 3\n", eos=False).terminal  # must start a line


def test_seed_derivation_matches_spec():
    import hashlib

    s = "17|jev-t-001|J64|0|2|0"
    h = hashlib.sha256(s.encode()).digest()
    want = int.from_bytes(h[:8], "little") % (2**63 - 1)
    assert branch_seed(17, "jev-t-001", "J64", 0, 2, 0) == want == derive(17, "jev-t-001", "J64", 0, 2, 0)
    assert option_order_seed(17, "jev-t-001", "J64", 0) == derive(17, "jev-t-001", "J64", 0, 0, "option_order")
    assert 0 <= want < 2**63 - 1


def test_permutation_deterministic():
    p1, p2 = permutation(4, 12345), permutation(4, 12345)
    assert p1 == p2 and sorted(p1) == [0, 1, 2, 3]
    assert len({tuple(permutation(4, s)) for s in range(200)}) > 10
