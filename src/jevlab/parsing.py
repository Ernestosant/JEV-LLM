"""Answer parsing and FINAL-line handling (protocol §7.4 and §11).

Pure standard library: used by the dataset builder, the online terminal detector and the
offline evaluator, so the three agree on the same rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

# Strict answer grammar (§11): optional sign, integer | p/q | finite decimal. Outer spaces
# allowed, Unicode minus normalised. Nothing else (no units, no expressions, no LaTeX).
_INT = re.compile(r"^[+-]?\d+$")
_DEC = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)$")
_FRAC = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_MINUS = {"\u2212": "-", "\u2013": "-", "\u2014": "-", "\ufe63": "-", "\uff0d": "-"}


def normalize_minus(s: str) -> str:
    for k, v in _MINUS.items():
        s = s.replace(k, v)
    return s


def parse_strict_answer(raw: str) -> Fraction | None:
    """Parse the text after 'FINAL:' with the strict grammar. None if invalid."""
    if raw is None:
        return None
    s = normalize_minus(raw).strip()
    if not s:
        return None
    if _INT.match(s):
        return Fraction(int(s))
    if _DEC.match(s):
        return Fraction(s)  # Fraction parses decimal strings exactly
    m = _FRAC.match(s)
    if m:
        p, q = int(m.group(1)), int(m.group(2))
        if q == 0:
            return None
        return Fraction(p, q)
    return None


# ---------------------------------------------------------------------------------------
# Gold-answer parsing for dataset construction (LaTeX from MATH \boxed{...}).
# ---------------------------------------------------------------------------------------

def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) with balanced braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


_LATEX_FRAC = re.compile(r"^([+-]?)\\[dt]?frac\{([+-]?\d+)\}\{(\d+)\}$")
_LATEX_FRAC_SHORT = re.compile(r"^([+-]?)\\[dt]?frac(\d)(\d)$")


def parse_latex_gold(ans: str) -> Fraction | None:
    """Accept only exact int / fraction / finite-decimal gold answers; reject everything else
    (radicals, pi, units, text, percent, money, lists, intervals, thousands separators)."""
    if ans is None:
        return None
    s = ans.strip()
    for tok in ("\\!", "\\,", "\\;", "\\ ", " "):
        s = s.replace(tok, "")
    s = s.strip("$")
    if not s:
        return None
    if any(bad in s for bad in ("\\$", "\\%", "%", "^\\circ", "\\text", "\\mbox", ",", "\\sqrt",
                                "\\pi", "i", "x", "\\infty", "(", "[", "\\le", "\\ge", "=",
                                "_", "^", "\\cdot", "\\times", "e")):
        return None
    m = _LATEX_FRAC.match(s) or _LATEX_FRAC_SHORT.match(s)
    if m:
        sign, p, q = m.group(1), int(m.group(2)), int(m.group(3))
        if q == 0:
            return None
        v = Fraction(p, q)
        return -v if sign == "-" else v
    return parse_strict_answer(s)


def fraction_to_str(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# ---------------------------------------------------------------------------------------
# FINAL-line terminal detection (§7.4) and offline extraction (§11).
# ---------------------------------------------------------------------------------------

_FINAL_LINE = re.compile(r"(?m)^[ \t]*FINAL:[^\n]*\n")
_FINAL_AT_END = re.compile(r"(?m)^[ \t]*FINAL:[^\n]*\Z")


@dataclass
class TerminalInfo:
    terminal: bool          # candidate is terminal (FINAL complete, or EOS)
    has_final: bool         # a FINAL line was delivered
    end_char: int | None    # index in text where the public output ends (after FINAL line)
    reason: str             # "final_line" | "final_at_eos" | "eos_no_final" | "open"


def terminal_state(text: str, eos: bool) -> TerminalInfo:
    """Decide terminality of a (possibly partial) generation.

    - A complete FINAL line (FINAL: ... + newline) marks terminal. The mere string 'FINAL:'
      is not enough while generation is open.
    - On EOS, a FINAL line at the very end of the buffer without newline is admitted.
    - EOS without FINAL -> terminal but invalid.
    """
    m = _FINAL_LINE.search(text)
    if m:
        return TerminalInfo(True, True, m.end(), "final_line")
    if eos:
        m2 = _FINAL_AT_END.search(text)
        if m2:
            return TerminalInfo(True, True, len(text), "final_at_eos")
        return TerminalInfo(True, False, None, "eos_no_final")
    return TerminalInfo(False, False, None, "open")


def extract_final(public_text: str) -> tuple[str, str | None, Fraction | None]:
    """Offline evaluator: returns (format_status, raw_answer, value).

    format_status in {"ok", "no_final", "multiple_final", "invalid_format"}.
    Exactly one delivered FINAL line is required (§11).
    """
    lines = [ln for ln in public_text.split("\n")]
    finals = [ln for ln in lines if ln.strip().startswith("FINAL:")]
    if not finals:
        return "no_final", None, None
    if len(finals) > 1:
        raw = finals[-1].strip()[len("FINAL:"):]
        return "multiple_final", raw, None
    raw = finals[0].strip()[len("FINAL:"):]
    v = parse_strict_answer(raw)
    if v is None:
        return "invalid_format", raw, None
    return "ok", raw, v
