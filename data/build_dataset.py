"""Build the frozen evaluation dataset (protocol §3 as amended by v3.1).

Sources (pinned revisions, downloaded into data/raw):
  * openai/gsm8k (test)                      -> arithmetic / ratios-percentages
  * EleutherAI/hendrycks_math (test)         -> algebra, number theory, counting/probability,
                                                prealgebra (ratios/percentages)
  * AI-MO/aimo-validation-amc (AMC12 22-23)  -> hard tier
  * AIME 2024 / 2025 (local copies from math_agent, hashed) -> hard tier

Design: 5 domains x (8 easy, 8 medium, 4 hard) = 100 test; dev = 5 x (2 easy, 1 medium,
1 hard) = 20, disjoint. Every choice is deterministic (seed below). Nothing in this script
looks at model outputs. Usage:  python data/build_dataset.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevlab.common import load_prompt, sha256_file, write_json  # noqa: E402
from jevlab.parsing import fraction_to_str, last_boxed, parse_latex_gold  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
SEED = 20260927
MAX_PROMPT_TOKENS = 512
MAX_ABS_ANSWER = 999  # answers >= 1000 are excluded (thousands separators, amendment v3.1)

SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              ["main/test-00000-of-00001.parquet"]),
    "math": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3",
             [f"{c}/test-00000-of-00001.parquet" for c in
              ("algebra", "number_theory", "counting_and_probability", "prealgebra")]),
    "amc": ("AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316",
            ["data/train-00000-of-00001.parquet"]),
}
AIME_FILES = ["aime/aime_2024_problems.jsonl", "aime/aime_2025_problems.jsonl"]

DOMAINS = ["arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"]
TEST_QUOTA = {"easy": 8, "medium": 8, "hard": 4}
DEV_QUOTA = {"easy": 2, "medium": 1, "hard": 1}

# ---------------------------------------------------------------- text filters
MONEY = re.compile(r"\$|dollar|\bcents?\b|\\\$", re.I)
PCT_WORDS = re.compile(r"percent|%|\bratio\b|proportion|\brate\b", re.I)
ASKS_PERCENT = re.compile(r"(what|which)\s+(is the\s+)?(percent|percentage)|as a percent|"
                          r"in percent|percentage (chance|of|is|was|did|does|increase|decrease)|"
                          r"by what percent", re.I)
COMP_MONEY = re.compile(r"dollar|\bcents?\b|\\\$|\$\$\d", re.I)
AIME_BOILERPLATE = re.compile(r"relatively prime positive integers|where \$?m\$? and \$?n\$? are "
                              r"relatively prime[^.]*|find \$?m\s*\+\s*n\$?", re.I)
MATH_EXCLUDE = re.compile(
    r"\[asy\]|nearest|mixed number|dollars?|\\\$|cents|degrees|\^\\circ|radical|base\s*\d|"
    r"in base|scientific notation|interval notation|ordered pair|in terms of|"
    r"express your answer as a percent|as a decimal to|significant", re.I)
COMP_EXCLUDE = re.compile(r"which of the following|\(A\)|\\textbf\{\(A\)", re.I)
GEOMETRY = re.compile(
    r"triangle|circle|angle|polygon|square|rectangle|hexagon|pentagon|radius|diameter|"
    r"perimeter|area|volume|cube|sphere|cylinder|cone|parallel|perpendicular|tangent|"
    r"coordinate|plane|segment|\bline\b|lattice|trapezoid|rhombus|quadrilateral|chord|"
    r"inscribed|circumscribed|vertex|vertices|\\overline|\\angle|\\triangle|octahedron|"
    r"tetrahedron|grid|hypotenuse|diagonal|arc|sector|prism|pyramid|ellipse|parabola|"
    r"point|distance", re.I)
CP_WORDS = re.compile(r"probabilit|how many ways|arrange|choose|random|expected|permutation|"
                      r"combination|subsets?|sequences? of|strings?|paths?|committee|dice|"
                      r"\bdie\b|coins?|cards?|selected|distinct|how many (?:positive )?"
                      r"(?:integers|numbers|ordered)", re.I)
NT_WORDS = re.compile(r"divisib|divides|divisor|remainder|prime|gcd|lcm|greatest common|"
                      r"least common|modulo|\bmod\b|digits?|factorial|multiple of|integer "
                      r"solutions|perfect square|\\pmod", re.I)
ALG_WORDS = re.compile(r"polynomial|equation|roots?|real numbers?|function|\\log|log_|"
                       r"sequence|\bsum\b|solve|system|quadratic|variable|f\(x\)|x\^", re.I)


def clean(s: str) -> str:
    return s.replace("\r\n", "\n").strip()


def download_sources() -> dict:
    from huggingface_hub import hf_hub_download

    local = {}
    for key, (repo, rev, files) in SOURCES.items():
        local[key] = [hf_hub_download(repo, f, repo_type="dataset", revision=rev, cache_dir=RAW)
                      for f in files]
    return local


def gsm8k_items(paths) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(paths[0])
    items = []
    for i, row in df.iterrows():
        q, sol = clean(row["question"]), row["answer"]
        ans_s = sol.split("####")[-1].strip().replace(",", "")
        items.append({"source": "openai/gsm8k", "source_split": "test", "source_index": int(i),
                      "problem": q, "gold_raw": ans_s, "solution": sol,
                      "steps": sol.count("<<"), "level": None, "subject": "gsm8k"})
    return items


def math_items(paths) -> list[dict]:
    import pandas as pd

    items = []
    for p in paths:
        cfg = Path(p).parent.name
        df = pd.read_parquet(p)
        for i, row in df.iterrows():
            level = row["level"]
            lv = int(level.split()[-1]) if level.split()[-1].isdigit() else None
            items.append({"source": "EleutherAI/hendrycks_math", "source_split": f"{cfg}/test",
                          "source_index": int(i), "problem": clean(row["problem"]),
                          "gold_raw": last_boxed(row["solution"]), "solution": row["solution"],
                          "steps": None, "level": lv, "subject": cfg})
    return items


def competition_items(amc_paths) -> list[dict]:
    import pandas as pd

    items = []
    df = pd.read_parquet(amc_paths[0])
    for i, row in df.iterrows():  # row position is unique; the 'id' column restarts per year
        a = float(row["answer"])
        items.append({"source": "AI-MO/aimo-validation-amc", "source_split": "train",
                      "source_index": int(i), "problem": clean(row["problem"]),
                      "gold_raw": str(int(a)) if a.is_integer() else None,
                      "solution": row["url"], "steps": None, "level": None, "subject": "amc12"})
    for f in AIME_FILES:
        for line in (RAW / f).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            items.append({"source": f"AIME {r['year']} ({r['exam']}) via math_agent",
                          "source_split": f"{r['year']}-{r['exam']}",
                          "source_index": int(r["problem_number"]), "problem": clean(r["problem"]),
                          "gold_raw": str(int(r["answer"])), "solution": "AoPS wiki (official key)",
                          "steps": None, "level": None, "subject": "aime"})
    return items


# Manual domain corrections for competition items whose keyword classification was wrong,
# decided by reading the statement only (never model outputs). None = exclude.
# key: exact prefix of the problem statement.
DOMAIN_OVERRIDES: dict[str, str | None] = {
    "Let $h_n$ and $k_n$ be the unique relatively prime": "number_theory",
    "A data set consists of $6$ (not distinct)": "algebra",
    "On Halloween $31$ children": None,                      # logic puzzle, no domain
    "Define $x\\diamond y$ to be $|x-y|$": "arithmetic",
    "How many $4 \\times 4$ arrays": "counting_probability",
    "Let $x_0,x_1,x_2,\\dotsc$ be a sequence of numbers, where each": "number_theory",
    "For how many values of the constant $k$ will the polynomial": "algebra",
    "The sequence $a_0,a_1,a_2,\\cdots$ is a strictly increasing": "algebra",
    "How many ordered pairs of positive real numbers $(a,b)$": "algebra",
    "There is a unique sequence of integers $a_1, a_2, \\cdots a_{2023}$": "algebra",
    "For how many ordered pairs $(a,b)$ of integers does the polynomial": "algebra",
    "For how many integers $n$ does the expression": "algebra",
    "Suppose $a$ is a real number such that the equation": None,  # trigonometry
    "Define $f(x)=||x|-\\tfrac{1}{2}|$": None,                    # graphs of trig functions
    "Let $k$ be a real number such that the system $|25+20i-z|=5$": None,  # complex-plane geometry
    "A sequence is defined by $x_1 = \\frac{25}{11}$": "algebra",
}


def gold_value(it) -> Fraction | None:
    if it["gold_raw"] is None:
        return None
    return parse_latex_gold(it["gold_raw"])


def classify_competition(text: str) -> str | None:
    if COMP_EXCLUDE.search(text) or GEOMETRY.search(text) or COMP_MONEY.search(text):
        return None
    text = AIME_BOILERPLATE.sub(" ", text)
    if PCT_WORDS.search(text):
        return "ratios_percentages"
    if CP_WORDS.search(text):
        return "counting_probability"
    if NT_WORDS.search(text):
        return "number_theory"
    if ALG_WORDS.search(text):
        return "algebra"
    return None


def assign(it) -> tuple[str | None, str | None, str]:
    """(domain, difficulty, rule) or (None, None, exclusion_reason)."""
    t = it["problem"]
    v = it.get("gold")
    if v is None:
        return None, None, "gold_not_exact_number"
    if abs(v) > MAX_ABS_ANSWER:
        return None, None, "gold_ge_1000"
    if it["subject"] == "gsm8k":
        if MONEY.search(t):
            return None, None, "money"
        if ASKS_PERCENT.search(t):
            return None, None, "asks_percent"
        dom = "ratios_percentages" if PCT_WORDS.search(t) else "arithmetic"
        s = it["steps"]
        diff = "easy" if s in (2, 3) else "medium" if s in (4, 5) else "hard" if s >= 7 else None
        if diff is None:
            return None, None, "gsm8k_steps_out_of_bands"
        return dom, diff, f"gsm8k steps={s}"
    if it["subject"] in ("amc12", "aime"):
        key = next((k for k in DOMAIN_OVERRIDES if t.startswith(k)), None)
        dom = DOMAIN_OVERRIDES[key] if key is not None else classify_competition(t)
        if key is not None and dom is None:
            return None, None, "competition_manual_exclusion"
        if dom is None:
            return None, None, "competition_unclassified_or_geometry_or_choices"
        return dom, "hard", f"{it['subject']} keyword-domain"
    # MATH
    if MATH_EXCLUDE.search(t):
        return None, None, "math_format_or_units"
    if MONEY.search(t):
        return None, None, "money"
    if ASKS_PERCENT.search(t):
        return None, None, "asks_percent"
    lv = it["level"]
    diff = "easy" if lv in (1, 2) else "medium" if lv == 3 else "hard" if lv in (4, 5) else None
    if diff is None:
        return None, None, "math_level_missing"
    subj = it["subject"]
    if subj == "prealgebra":
        if not PCT_WORDS.search(t):
            return None, None, "prealgebra_not_ratio_percent"
        return "ratios_percentages", diff, f"math prealgebra L{lv}"
    dom = {"algebra": "algebra", "number_theory": "number_theory",
           "counting_and_probability": "counting_probability"}[subj]
    return dom, diff, f"math {subj} L{lv}"


def token_counter():
    """Rendered-prompt token counts with the G and B tokenizers and templates (§5)."""
    from jevlab.prompts import load_fast_tokenizer_file, render_generator_prompt

    system = load_prompt(ROOT / "prompts" / "generador.txt")
    toks = {k: load_fast_tokenizer_file(str(RAW / "model_meta" / k)) for k in ("G", "B")}

    def count(problem: str) -> dict:
        return {k: len(t.encode(render_generator_prompt(t, system, problem),
                                add_special_tokens=False)) for k, t in toks.items()}

    return count


def norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def near_duplicate(a: str, b: str, n: int = 8, thr: float = 0.5) -> bool:
    wa, wb = norm_words(a), norm_words(b)
    ga = {tuple(wa[i:i + n]) for i in range(max(0, len(wa) - n + 1))}
    gb = {tuple(wb[i:i + n]) for i in range(max(0, len(wb) - n + 1))}
    if not ga or not gb:
        return False
    return len(ga & gb) / min(len(ga), len(gb)) >= thr


# Hard-tier source preference per domain (amendment v3.1): competition first, then the
# declared fallback. A tier is filled in this order until the quota is met.
HARD_PREF = {
    "arithmetic": ["competition", "gsm8k"],
    "algebra": ["competition", "math"],
    "ratios_percentages": ["competition", "math"],
    "number_theory": ["competition", "math"],
    "counting_probability": ["competition", "math"],
}
# Easy/medium: ratios/percentages alternate GSM8K and MATH prealgebra; others single source.


def family(it) -> str:
    if it["subject"] in ("amc12", "aime"):
        return "competition"
    return "gsm8k" if it["subject"] == "gsm8k" else "math"


def main() -> None:
    rng = random.Random(SEED)
    paths = download_sources()
    pool = gsm8k_items(paths["gsm8k"]) + math_items(paths["math"]) + competition_items(paths["amc"])
    excluded = Counter()
    eligible = []
    for it in pool:
        it["gold"] = gold_value(it)
        dom, diff, rule = assign(it)
        if dom is None:
            excluded[rule] += 1
            continue
        it.update(domain=dom, difficulty=diff, rule=rule)
        eligible.append(it)

    count_tokens = token_counter()
    buckets = defaultdict(list)
    for it in sorted(eligible, key=lambda x: (x["source"], x["source_split"], x["source_index"])):
        buckets[(it["domain"], it["difficulty"], family(it))].append(it)
    for k in buckets:
        rng.shuffle(buckets[k])

    chosen: list[dict] = []

    def take(dom: str, diff: str, fams: list[str], n: int, split: str) -> None:
        # hard tier: exhaust families in preference order; easy/medium: round-robin families
        got = 0
        queues = [buckets[(dom, diff, f)] for f in fams]
        qi = 0
        while got < n and any(queues):
            if diff == "hard":
                q = next((q for q in queues if q), None)
            else:
                q = queues[qi % len(queues)]
                qi += 1
                if not q:
                    continue
            it = q.pop(0)
            if any(near_duplicate(it["problem"], c["problem"]) for c in chosen):
                excluded["near_duplicate"] += 1
                continue
            tc = count_tokens(it["problem"])
            if max(tc.values()) > MAX_PROMPT_TOKENS:
                excluded["prompt_gt_512_tokens"] += 1
                continue
            it.update(split=split, prompt_tokens=tc)
            chosen.append(it)
            got += 1
        if got < n:
            raise SystemExit(f"not enough items for {dom}/{diff}: {got}/{n}")

    for split, quota in (("test", TEST_QUOTA), ("dev", DEV_QUOTA)):
        for dom in DOMAINS:
            for diff, n in quota.items():
                if diff == "hard":
                    fams = HARD_PREF[dom]
                elif dom == "ratios_percentages":
                    fams = ["gsm8k", "math"]
                elif dom == "arithmetic":
                    fams = ["gsm8k"]
                else:
                    fams = ["math"]
                take(dom, diff, fams, n, split)

    # Opaque, shuffled ids: nothing in the id reveals source, domain or difficulty.
    for split in ("test", "dev"):
        items = [c for c in chosen if c["split"] == split]
        rng.shuffle(items)
        for k, it in enumerate(items, 1):
            it["id"] = f"jev-{split[0]}-{k:03d}"

    def gold_rec(it):
        v = it["gold"]
        return {"id": it["id"], "gold_numerator": v.numerator, "gold_denominator": v.denominator,
                "gold_answer": fraction_to_str(v), "gold_raw": it["gold_raw"],
                "reference_solution": it["solution"], "domain": it["domain"],
                "difficulty": it["difficulty"], "difficulty_rule": it["rule"],
                "template_group": f"{it['source']}::{it['source_split']}::{it['source_index']}",
                "source": it["source"], "source_split": it["source_split"],
                "source_index": it["source_index"], "prompt_tokens": it["prompt_tokens"],
                "reviewed": False}

    files = {}
    for split in ("test", "dev"):
        items = sorted([c for c in chosen if c["split"] == split], key=lambda x: x["id"])
        fi, fg = OUT / f"{split}_inputs.jsonl", OUT / f"{split}_gold.jsonl"
        with open(fi, "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps({"id": it["id"], "problem": it["problem"], "language": "en"},
                                   ensure_ascii=False) + "\n")
        with open(fg, "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(gold_rec(it), ensure_ascii=False) + "\n")
        files[fi.name], files[fg.name] = fi, fg

    # Schedule: 10 blocks of 10, balanced by domain (round-robin), block order and within-block
    # order shuffled with the planning seed. Contains ids only (no gold) -> ships with inputs.
    test = sorted([c for c in chosen if c["split"] == "test"], key=lambda x: x["id"])
    by_dom = defaultdict(list)
    for it in test:
        by_dom[it["domain"]].append(it["id"])
    prng = random.Random(SEED)
    for d in DOMAINS:
        prng.shuffle(by_dom[d])
    blocks = [[] for _ in range(10)]
    for d in DOMAINS:
        for k, pid in enumerate(by_dom[d]):
            blocks[k % 10].append(pid)
    for b in blocks:
        prng.shuffle(b)
    prng.shuffle(blocks)
    schedule = {"planning_seed": SEED, "blocks": blocks,
                "order": [pid for b in blocks for pid in b]}
    write_json(OUT / "schedule.json", schedule)
    files["schedule.json"] = OUT / "schedule.json"

    # Pre-registered blind secondary review: 4 test ids per domain (protocol §11).
    rrng = random.Random(SEED + 11)
    review_ids = []
    for d in DOMAINS:
        ids = sorted(by_dom[d])
        review_ids += sorted(rrng.sample(ids, 4))
    write_json(OUT / "blind_review_ids.json", {"seed": SEED + 11, "ids": review_ids,
                                               "note": "chosen before any model output"})
    files["blind_review_ids.json"] = OUT / "blind_review_ids.json"

    with open(OUT / "review_sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "domain", "difficulty", "source", "problem", "gold_answer",
                    "unique_answer_ok", "statement_ok", "solution_ok", "difficulty_ok",
                    "reviewer", "notes"])
        for it in sorted(chosen, key=lambda x: x["id"]):
            w.writerow([it["id"], it["split"], it["domain"], it["difficulty"], it["source"],
                        it["problem"], fraction_to_str(it["gold"]), "", "", "", "", "", ""])

    manifest = {
        "seed": SEED, "sources": {k: {"repo": v[0], "revision": v[1], "files": v[2]}
                                  for k, v in SOURCES.items()},
        "aime_local_files": {f: sha256_file(RAW / f) for f in AIME_FILES},
        "quota": {"test": TEST_QUOTA, "dev": DEV_QUOTA}, "domains": DOMAINS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS, "max_abs_answer": MAX_ABS_ANSWER,
        "pool_size": len(pool), "eligible": len(eligible), "exclusions": dict(excluded),
        "counts": {split: Counter(f"{c['domain']}/{c['difficulty']}/{family(c)}"
                                  for c in chosen if c["split"] == split)
                   for split in ("test", "dev")},
        "answer_types": Counter("int" if c["gold"].denominator == 1 else
                                "fraction_or_decimal" for c in chosen),
        "prompt_tokens_max": max(max(c["prompt_tokens"].values()) for c in chosen),
    }
    write_json(OUT / "dataset_manifest.json", manifest)
    files["dataset_manifest.json"] = OUT / "dataset_manifest.json"
    with open(OUT / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        for name in sorted(files):
            f.write(f"{sha256_file(files[name])}  {name}\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
