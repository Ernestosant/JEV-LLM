"""Prepare v3 candidates; freeze ONLY separately evidenced, actual agent reviews.

python -B data/build_dataset_v3.py --prepare
python -B data/build_dataset_v3.py --freeze
See data/v3/REVIEW_CONTRACT.md. Preparation never writes final split files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as previous

ROOT = previous.ROOT
OUT = ROOT / "data" / "v3"
RAW = ROOT / "data" / "raw" / "v3"
SEED = 20261002
LATENCY_SEED = 20261003
DOMAINS = previous.DOMAINS
TEST_QUOTA = {"easy": 30, "medium": 40, "hard": 30}
PILOT_QUOTA = {"easy": 3, "medium": 4, "hard": 3}
LATENCY_QUOTA = {"easy": 6, "medium": 8, "hard": 6}
EXPERIMENT_SEEDS = [17, 29, 43]
REVIEW_POLICY_ID = "jev-v3-source-sampling-tier-agent-review-20261004"
QUOTA_AXIS = "source_sampling_tier"
SCOPE_RULE_ID = "assigned-input-integrity-v1"
SCOPE_LEDGER = "review_scope_incidents.jsonl"
OPENSTAX_REPO = "openstax/osbooks-college-algebra-bundle"
OPENSTAX_REVISION = "463991614337632b0e02cbb6c76223cbb0d423d3"
OPENSTAX_MODULES = ("m49448", "m49450", "m51254", "m51279", "m51280", "m51281")
CNX = "{http://cnx.rice.edu/cnxml}"
MATHML = "{http://www.w3.org/1998/Math/MathML}"
MAX_ANSWER = 10**9
MAX_TOKENS = 1024
META_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MODELS = {
    "G4": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "B13": ("DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking",
            "717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe"),
    "O9": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}
ASDIV_REPO = "chaochun/nlu-asdiv-dataset"
ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
# Independently obtained from the official HF revision API with blobs=true.
O9_FILE_PINS = {
    "tokenizer.json": ("sha256", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "tokenizer_config.json": ("git_blob", "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1"),
    "chat_template.jinja": ("git_blob", "a585dec894e63da457d9440ec6aa7caa16d20860"),
    "config.json": ("git_blob", "273ce437e01baf96a07cd9eb3d5f48bac8d7c657"),
    "README.md": ("git_blob", "0f3972cb2c995a86bde9ac92440da530fe2b2c68"),
    "LICENSE": ("git_blob", "f938136e3adacfd92be087f6e113b5d6d97f678f"),
}
FORMAT_EXCLUDE = re.compile(
    r"\[asy\]|<img|\\includegraphics|diagram|nearest|round(?:ed|ing)?\s+to|"
    r"approximate|significant figures|mixed number|interval notation|"
    r"in terms of|scientific notation|which of the following|\\textbf\{\(A\)|"
    r"\(A\).*\(B\)", re.I)
RATIO = re.compile(r"percent|%|\bratio\b|proportion|\brate\b", re.I)
SYMBOLIC = re.compile(
    r"equation|variable|solve for|polynomial|function|\\log|\\sqrt|"
    r"\$[^$]*\b[a-z]\b[^$]*\$|\b[a-z]\s*[=^]|\b[a-z]\s*\+\s*\d", re.I)
COUNTING_CONTENT = re.compile(
    r"probabilit|expected value|random(?:ly)?|how many[^?\.]{0,70}(?:ways|arrangements|ordered pairs)|"
    r"number of (?:ways|arrangements|subsets)|permutations?|combinations?|anagrams?|"
    r"arrang(?:e|ed|ements?)|subsets?|strings? of|committee|dice|\bdie\b|toss(?:ed|ing)?", re.I)
NT_CONTENT = re.compile(
    r"divisib|divisors?|remainder|greatest common|least common|\bgcd\b|\blcm\b|"
    r"modulo|\\pmod|\bcongruen|prime (?:numbers?|integers?|factors?)|relatively prime|"
    r"(?:decimal|binary|base[- ]\d+) (?:representation|expansion)|terminating decimal|"
    r"(?:sum|product) of (?:the )?digits|(?:last|first|units|tens|hundreds)[- ]digits?|"
    r"decimal digits?|\d+(?:st|nd|rd|th)[- ](?:decimal )?digits?|digits? of (?:the )?(?:fraction|number)", re.I)
FORMAL_ALGEBRA = re.compile(
    r"polynomial|quadratic|discriminant|(?<!square )(?<!cube )(?<!third )(?<!fourth )\broots?\b|logarithm|\\log|"
    r"complex numbers?|recursive|recurrence|arithmetic sequence|geometric series|"
    r"composition|inverse functions?|inverse of|radicals?", re.I)
QUANTITIES = re.compile(
    r"\b(?:students?|boys?|girls?|children|people|persons?|workers?|painters?|family|families|"
    r"marbles?|coffee|milk|water|sugar|mixture|solution|money|price|cost|income|tax|"
    r"invest(?:ment|ments|ed|ing)?|interest|savings|weigh(?:s|ed|ing)?|weights?|pounds?|ounces?|gallons?|liters?|litres?|"
    r"hours?|minutes?|seconds?|days?|weeks?|years?|miles?|meters?|metres?|"
    r"speed|distance|momentum|errors?|voltage|current|resistance|mass|force|population|"
    r"amounts?|quantit(?:y|ies)|jobs?|tasks?|work|walk|run|travel|bounc(?:e|es|ing)|rebound(?:s|ed|ing)?|recipe|flour|"
    r"pay|paid|win|won|loss|lost|games?|races?|cyclists?|runners?|fuel|robots?)\b", re.I)
NON_SCALAR_REQUEST = re.compile(
    r"(?:construct|write|derive|give)\s+(?:(?:a|an|the)\s+)?(?:probability\s+)?"
    r"(?:model|recursive formula|formula|equation|function|expression|polynomial|graph|table)|"
    r"(?:find|determine)\s+(?:(?:a|an|the)\s+)?(?:inverse|recursive formula|function|equation)|"
    r"(?:and then|and also)\s+(?:find|determine|calculate|write)|"
    r"and\s+(?:determine whether|decide whether|state whether|whether)|"
    r"(?:find|determine)\s+(?:the\s+)?values? of\s+\$?[a-z]\$?\s+and\s+\$?[a-z]\$?\b", re.I)


def geometric_reasoning(text):
    """Actual geometric construction/measurement, not substring homonyms."""
    if re.search(r"shadow|similar triangles|\\angle|\\triangle|\bangles?\b|hypotenuse|circumference|"
                 r"inscribed|circumscribed|perpendicular|tangent|radius|diameter", text, re.I):
        return True
    discrete = (COUNTING_CONTENT.search(text) and re.search(r"integer|lattice|grid|arrays?|arrang|ways", text, re.I))
    shape_text = re.sub(r"\bperfect (?:squares?|cubes?)\b|\b(?:square|cube) roots?\b|"
                        r"\b(?:square|cube) of (?:\$?\d+|\$?[a-z]\$?\b|(?:a|the) (?:number|integer))|"
                        r"\bsquare (?:miles?|meters?|metres?|kilometers?|kilometres?|feet|foot|inches|yards?)\b|"
                        r"\b(?:six-sided )?number cubes?\b", " ", text, flags=re.I)
    shapes = re.search(r"\b(?:triangles?|triangular|circles?|circular|polygons?|rectangles?|rectangular|squares?|cubes?|cubical|cuboids?|hexagons?|pentagons?|"
                       r"spheres?|cylinders?|cones?|trapezoids?|rhombi?|quadrilaterals?|"
                       r"tetrahedrons?|pyramids?|ellipses?|prisms?)\b|"
                       r"(?:side|diagonal|area|perimeter) of (?:a|the) square|square with", shape_text, re.I)
    if shapes and not discrete:
        return True
    if shapes and re.search(r"area|volume|perimeter|angles?|length of|measure", shape_text, re.I):
        return True
    return bool(re.search(r"distance between[^.]*points?|coordinate distance|length of[^.]*segment", text, re.I))


def content_domain(item):
    """One pre-sampling rule for all subjects/tiers; never uses quotas or gold."""
    text = previous.AIME_BOILERPLATE.sub(" ", item["problem"])
    if geometric_reasoning(text):
        return None, "actual_geometric_reasoning"
    counting = COUNTING_CONTENT.search(text)
    number_theory = NT_CONTENT.search(text)
    if re.search(r"probabilit|expected value|random(?:ly)?|chance of|"
                 r"\bcoins?\b[^.]*(?:flip|toss|heads?|tails?|fair|biased|unfair|likely)|"
                 r"(?:flip|toss)[^.]*\bcoins?\b", text, re.I):
        return "counting_probability", "stochastic_event_not_a_ratio_answer_format"
    if number_theory and not (counting and re.search(r"arrang|permutation|combination|strings?|subsets", text, re.I)):
        return "number_theory", "substantive_divisibility_representation_or_modular_content"
    if FORMAL_ALGEBRA.search(text):
        return "algebra", "formal_polynomial_roots_or_nonlinear_algebra_not_incidental_ratio"
    if counting:
        return "counting_probability", "stochastic_or_combinatorial_content_not_fraction_format"
    formal = FORMAL_ALGEBRA.search(text)
    variation = re.search(r"varies? (?:directly|inversely|jointly)|(?:direct|inverse) variation|"
                          r"(?:directly|inversely) proportional", text, re.I)
    nonlinear_unknown = re.search(r"[a-z]\s*\^\s*\{?[2-9]|fourth root|compound interest rate|"
                                 r"annual interest rate[^.]*find|find[^.]*interest rate", text, re.I)
    averages = re.search(r"(?:average|mean)[^.]*\b(?:test|exam|quiz|score|grade)|"
                        r"(?:test|exam|quiz|score|grade)[^.]*(?:average|mean)", text, re.I)
    ratio_marker = re.search(r"percent|%|\bratio\b|proportion|\brate\b|scale", text, re.I)
    physical_rate = re.search(r"(?:mile|meter|metre|kilometer|kilometre|foot|feet|inch|gallon|liter|litre|"
                              r"furlong)s? (?:per|an?|each) (?:hour|minute|second|day|fortnight)|"
                              r"miles?-per-gallon|\bmph\b|\bkph\b|average speed|"
                              r"(?:painters?|workers?|pipes?|machines?)[^.]*(?:hours?|minutes?)|"
                              r"(?:hours?|minutes?)[^.]*(?:painters?|workers?|pipes?|machines?)|"
                              r"(?:speed|bounc|rebound|catch)[^.]*(?:time|hours?|minutes?|distance|height|miles?)", text, re.I)
    shares = re.search(r"\\[dt]?frac[^.]{0,45}\b(?:of|times)\b|"
                       r"\b(?:half|halves|third|fifth|portion|share)\b|\bquarter of\b|\btwice\b|"
                       r"\bdouble (?:the|his|her|its|their|as|in)\b", text, re.I)
    pure_numeric_ratio = re.search(r"\bratio\b[^.]*\d[^.]*\d", text, re.I) and not SYMBOLIC.search(text)
    percentage_change = re.search(r"percentage (?:increase|decrease)|by (?:what|how many) percent", text, re.I)
    proportional = (variation or pure_numeric_ratio or percentage_change
                    or (QUANTITIES.search(text) and (ratio_marker or physical_rate or shares)))
    if proportional and not formal and not averages and (variation or not nonlinear_unknown):
        return "ratios_percentages", "central_quantitative_shares_percentage_unit_rate_or_variation"
    subject = item["subject"]
    if subject in ("gsm8k", "asdiv"):
        if SYMBOLIC.search(text):
            return None, "natural_symbolic_not_genuine_arithmetic"
        return "arithmetic", "natural_exact_arithmetic_without_substantive_proportional_content"
    if subject == "prealgebra":
        if SYMBOLIC.search(text):
            return None, "prealgebra_symbolic_not_genuine_arithmetic"
        return "arithmetic", "prealgebra_exact_arithmetic_without_ratios_or_symbolic_structure"
    if subject in ("amc12", "aime"):
        if formal or previous.ALG_WORDS.search(text) or SYMBOLIC.search(text):
            return "algebra", "formal_competition_algebra_content"
        if previous.NT_WORDS.search(text):
            return "number_theory", "competition_integer_structure"
        return None, "competition_unclassified"
    if subject == "openstax":
        return None, "publisher_task_not_classified_by_global_content"
    return {"algebra": "algebra", "number_theory": "number_theory",
            "counting_and_probability": "counting_probability"}[subject], "source_subject_hint_when_no_content_override"


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def digest(path):
    return previous.sha256_file(path)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def source_id(item):
    return f"{item['source']}::{item['source_split']}::{item['source_index']}"


def candidate_sha(item):
    return digest_bytes(canonical({k: v for k, v in item.items()
                                   if k not in ("id", "candidate_sha256", "priority")}).encode("utf-8"))


def protected_inventory():
    paths = [p for p in (ROOT / "data").rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p != Path(__file__).resolve()
             and not p.is_relative_to(OUT) and not p.is_relative_to(RAW)]
    paths += list((ROOT / "results").glob("**/preparation/*rejected*"))
    return {relative(p): digest(p) for p in sorted(paths) if p.is_file()}


def verify_hashes(inventory):
    for name, expected in inventory.items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Protected/pinned artifact changed: {name}")


def verify_seal(directory):
    hashes = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        name = name.strip()
        path = directory / name
        if (name in hashes or Path(name).is_absolute() or ".." in Path(name).parts
                or not path.resolve().is_relative_to(directory.resolve())
                or path.is_symlink() or digest(path) != expected):
            raise ValueError(f"Invalid seal: {directory}/{name}")
        hashes[name] = expected
    return hashes


def seal(directory, paths, required_hashes=None):
    marker = directory / "SHA256SUMS"
    pending = directory / "SHA256SUMS.pending"
    if marker.exists():
        raise ValueError("Refusing to overwrite an existing seal")
    inventory = {path: digest(path) for path in sorted(paths)}
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        for path, expected in inventory.items():
            handle.write(f"{expected}  {path.relative_to(directory).as_posix()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    for path in [*paths, pending]:
        path.chmod(0o444)
    for path, expected in inventory.items():
        if digest(path) != expected:
            raise ValueError("Artifact changed while sealing; completed marker not published")
    if required_hashes:
        verify_hashes(required_hashes)
    pending.rename(marker)


def load_exclusions():
    """Used union and rejected candidate membership are intentionally separate."""
    for directory in (ROOT / "data", ROOT / "data" / "v2"):
        verify_seal(directory)
    used, extra, dev = {}, {}, None

    def ingest(inputs, gold, registry, origin):
        if len(inputs) != len(gold) or [i["id"] for i in inputs] != [g["id"] for g in gold]:
            raise ValueError(f"Unaligned previous records: {origin}")
        for i, g in zip(inputs, gold):
            sid = source_id(g)
            if sid != g["template_group"]:
                raise ValueError(f"Previous source identity mismatch: {origin}")
            row = registry.setdefault(sid, {"source_id": sid, "problem": i["problem"], "origins": []})
            if row["problem"] != i["problem"]:
                raise ValueError(f"Conflicting previous statement: {sid}")
            row["origins"].append({"artifact": origin, "id": i["id"]})

    for directory, splits in ((ROOT / "data", ("test", "dev")),
                              (ROOT / "data" / "v2", ("test", "pilot", "dev"))):
        for split in splits:
            ii, gg = (read_jsonl(directory / f"{split}_{kind}.jsonl") for kind in ("inputs", "gold"))
            ingest(ii, gg, used, relative(directory / f"{split}_inputs.jsonl"))
            if directory == ROOT / "data" and split == "dev":
                dev = (ii, gg)
    if len(used) != 260 or dev is None or len(dev[0]) != 20:
        raise ValueError(f"Expected union 260 used source IDs and 20 original dev, got {len(used)}")
    archives = []
    for path in sorted((ROOT / "results").glob("**/preparation/*rejected.zip")):
        audit_path = path.with_suffix(".json")
        audit = read_json(audit_path)
        if audit.get("archive_sha256") != digest(path):
            raise ValueError(f"Rejected archive changed: {path}")
        with zipfile.ZipFile(path) as archive:
            inventory = json.loads(archive.read("hash_inventory.json"))
            for name, expected in inventory.items():
                if digest_bytes(archive.read("data/v2/" + name)) != expected:
                    raise ValueError(f"Rejected archive member changed: {path}/{name}")
            count = 0
            identities = set()
            for split in ("test", "pilot"):
                ii, gg = ([json.loads(line) for line in archive.read(
                    f"data/v2/{split}_{kind}.jsonl").decode("utf-8").splitlines() if line.strip()]
                          for kind in ("inputs", "gold"))
                ingest(ii, gg, extra, relative(path) + ":" + split)
                for row in gg:
                    if source_id(row) in identities:
                        raise ValueError("Duplicate candidate source identity inside rejected archive")
                    identities.add(source_id(row))
                count += len(ii)
            if count != 140 or len(identities) != 140:
                raise ValueError(f"Expected 140 identifiable rejected candidates in {path}, got {count}")
            archives.append({"path": relative(path), "sha256": digest(path), "candidate_count": count})
    if len(archives) != 2:
        raise ValueError("Expected both identifiable 140-item pre-inference rejected archives")
    return list(used.values()), list(extra.values()), dev, archives


def words(text, numeric_template=False):
    text = unicodedata.normalize("NFKC", text).casefold()
    if numeric_template:
        text = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", " numtoken ", text)
    return tuple(re.findall(r"[a-z0-9]+", text))


def grams(tokens, n):
    return {tokens[i:i+n] for i in range(max(0, len(tokens)-n+1))}


def duplicate_reason(a, b):
    wa, wb = words(a), words(b)
    if wa == wb:
        return "exact_normalized"
    ga, gb = grams(wa, 8), grams(wb, 8)
    if ga and gb and len(ga & gb) / min(len(ga), len(gb)) >= 0.5:
        return "shared_8gram"
    ta, tb = words(a, True), words(b, True)
    if "numtoken" in ta or "numtoken" in tb:
        if ta == tb:
            return "numeric_template_exact"
        if (grams(ta, 4) & grams(tb, 4) and min(len(ta), len(tb)) / max(len(ta), len(tb)) >= 0.85
                and SequenceMatcher(None, ta, tb, autojunk=False).ratio() >= 0.9):
            return "numeric_template_approximate"
    return None


class DuplicateIndex:
    """Inverted spans avoid quadratic comparisons across the real candidate pool."""

    def __init__(self, rows=()):
        self.rows = []
        self.index = defaultdict(set)
        for row in rows:
            self.add(row["problem"])

    @staticmethod
    def keys(problem):
        w, t = words(problem), words(problem, True)
        return [("exact", w), ("template", t), *[("g8", g) for g in grams(w, 8)],
                *[("t4", g) for g in grams(t, 4)]]

    def add(self, problem):
        for key in self.keys(problem):
            self.index[key].add(len(self.rows))
        self.rows.append(problem)

    def match(self, problem):
        possible = set()
        for key in self.keys(problem):
            possible.update(self.index.get(key, ()))
        for index in sorted(possible):
            reason = duplicate_reason(problem, self.rows[index])
            if reason:
                return reason
        return None


def fetch(path, url):
    if path.exists():
        return
    request = Request(url, headers={"User-Agent": "Jev-LLM-v3-data-preparation"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def asdiv_items(path):
    items = []
    for row in ElementTree.parse(path).getroot().iter("Problem"):
        body, question = row.findtext("Body", ""), row.findtext("Question", "")
        answer = row.findtext("Answer", "")
        match = re.fullmatch(r"\s*([+-]?[\d,]+(?:\.\d+)?|[+-]?\d+/\d+)\s*(?:\(([^()]*)\))?\s*", answer)
        formula = row.findtext("Formula", "")
        items.append({"source": ASDIV_REPO, "source_split": "official-corpus",
                      "source_index": row.attrib["ID"], "original_source_id": row.attrib["ID"],
                      "problem": (body.strip() + " " + question.strip()).strip(),
                      "gold_raw": match.group(1) if match else None,
                      "answer_unit": match.group(2) if match else None,
                      "solution": formula, "steps": len(re.findall(r"[+*/-]", formula.split("=")[0])),
                      "level": row.attrib.get("Grade"), "subject": "asdiv",
                      "original_source": row.attrib.get("Source"), "original_answer": answer,
                      "solution_type": row.findtext("Solution-Type"),
                      "solution_type_attributes": row.find("Solution-Type").attrib,
                      "source_revision": ASDIV_REVISION})
    return items


def publisher_math(node):
    tag = node.tag.removeprefix(MATHML)
    children = list(node)
    if tag == "mfrac" and len(children) == 2:
        return "\\frac{" + publisher_math(children[0]) + "}{" + publisher_math(children[1]) + "}"
    if tag in ("msup", "msub", "msubsup"):
        base = publisher_math(children[0])
        if children[0].tag not in (MATHML + "mi", MATHML + "mn", MATHML + "mo"):
            base = "(" + base + ")"
        if tag == "msubsup":
            return base + "_{" + publisher_math(children[1]) + "}^{" + publisher_math(children[2]) + "}"
        return base + ("^{" if tag == "msup" else "_{") + publisher_math(children[1]) + "}"
    if tag in ("msqrt", "mroot"):
        radicand = publisher_math(children[0]) if tag == "mroot" else "".join(publisher_math(c) for c in children)
        return "\\sqrt" + ("[" + publisher_math(children[1]) + "]" if tag == "mroot" else "") + "{" + radicand + "}"
    if tag == "mspace":
        return " "
    if tag == "mtable":
        return "\\begin{aligned}" + " \\\\ ".join(publisher_math(c) for c in children) + "\\end{aligned}"
    if tag == "mtr":
        return " & ".join(publisher_math(c) for c in children)
    if tag == "munderover":
        return publisher_math(children[0]) + "_{" + publisher_math(children[1]) + "}^{" + publisher_math(children[2]) + "}"
    if tag == "munder":
        lower = publisher_math(children[1])
        if lower == "\u23df":
            return "\\underbrace{" + publisher_math(children[0]) + "}"
        return publisher_math(children[0]) + "_{" + lower + "}"
    if tag == "mover":
        accent = "".join(children[1].itertext()).strip()
        command = {"\u00af": "bar", "\u203e": "overline", "^": "hat", "\u2192": "vec"}.get(accent)
        if not command:
            raise ValueError("Unsupported MathML accent; refuse semantic flattening")
        return "\\" + command + "{" + publisher_math(children[0]) + "}"
    if tag == "mfenced":
        return node.get("open", "(") + node.get("separators", ",").join(publisher_math(c) for c in children) + node.get("close", ")")
    if tag not in ("math", "mrow", "mstyle", "mtd", "mi", "mn", "mo", "mtext"):
        raise ValueError(f"Unsupported MathML structure: {tag}")
    text = (node.text or "") + "".join(publisher_math(c) + ((c.tail or "") if (c.tail or "").strip() else "") for c in children)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {"\u2212": "-", "\u00d7": "\\times ", "\u22c5": "\\cdot ", "\u00b7": "\\cdot ",
                    "\u2248": "\\approx ", "\u2264": "\\le ", "\u2265": "\\ge ", "\u200b": "",
                    "\u2211": "\\sum ", "\u221e": "\\infty ", "\u2061": "", "\u2062": "\\cdot "}
    for character, latex in replacements.items():
        text = text.replace(character, latex)
    return text


def publisher_text(node):
    if node.tag == CNX + "title":
        return ""
    if node.tag == MATHML + "math":
        return "$" + publisher_math(node) + "$"
    text = (node.text or "")
    for child in node:
        text += publisher_text(child) + (child.tail or "")
        if child.tag in (CNX + "para", CNX + "item", CNX + "equation", CNX + "newline"):
            text += "\n"
    return re.sub(r"[ \t]+", " ", text).strip()


def publisher_gold(solution):
    rendered = publisher_text(solution)
    if re.search(r"\\approx|approximately|rounded|nearest|round to", rendered, re.I):
        return None, None
    endpoints = []
    for node in solution.iter():
        if node.tag in (MATHML + "math", MATHML + "mtr"):
            expression = publisher_math(node)
            if "\\begin{aligned}" in expression:
                continue
            rhs = expression.rsplit("=", 1)[-1].strip().rstrip(".")
            # A bare intermediate number is not an author-stated final answer.
            if "=" not in expression and not expression.startswith("\\frac"):
                continue
            value, _ = numeric_gold(rhs)
            if value is not None:
                endpoints.append((previous.fraction_to_str(value), node.get("id") or "mathml-final-rhs"))
        elif node.tag == CNX + "para":
            text = publisher_text(node).strip()
            direct, _ = numeric_gold(text.strip("$ ").rstrip("."))
            if direct is not None:
                endpoints.append((previous.fraction_to_str(direct), node.get("id")))
            conclusion = re.fullmatch(r"(?:There are |The (?:total|answer) is )?"
                r"([\d,]+(?:\.\d+)?|\d+/\d+)\s+(?:total |possible )?"
                r"(?:ways|options|outfits|permutations|arrangements|combinations|outcomes|sundaes|miles|mi|inches)\.?", text, re.I)
            if conclusion:
                value, _ = numeric_gold(conclusion.group(1))
                if value is not None:
                    endpoints.append((previous.fraction_to_str(value), node.get("id")))
    return endpoints[-1] if endpoints else (None, None)


def publisher_items(path, audit=None):
    root = ElementTree.parse(path).getroot()
    module = path.parent.name
    items = []
    holds = Counter()

    def walk(node, contexts=()):
        current = contexts
        for child in node:
            if child.tag == CNX + "para" and re.search(r"for the following exercises|exercises that follow|"
                    r"(?:find|determine|calculate)[^.]*the following", publisher_text(child), re.I):
                current = (child,)
            elif child.tag == CNX + "exercise":
                problem, solution = child.find(CNX + "problem"), child.find(CNX + "solution")
                if problem is None or solution is None:
                    holds["missing_problem_or_author_solution"] += 1
                    continue
                if any(n.tag in (CNX + "image", CNX + "list") for n in problem.iter()):
                    holds["multipart_list_or_diagram_original"] += 1
                    continue  # Multipart/list/diagram originals are not scalar standalones.
                pieces = (*current, problem)
                if any(n.tag == CNX + "link" or n.tag == CNX + "image" for p in pieces for n in p.iter()):
                    holds["external_reference_or_figure_context"] += 1
                    continue
                try:
                    text = "\n".join(publisher_text(p) for p in pieces if publisher_text(p))
                    reference = publisher_text(solution)
                    gold, locator = publisher_gold(solution)
                except (ValueError, IndexError):
                    holds["unsupported_or_malformed_mathml"] += 1
                    continue  # Unsupported math is held, never silently simplified.
                if NON_SCALAR_REQUEST.search(text) or text.count("?") > 1:
                    holds["original_request_not_single_scalar"] += 1
                    continue
                if re.search(r"spinner|round[^.]*nearest|show calculations and round", text, re.I):
                    holds["diagram_or_rounding_instruction"] += 1
                    continue
                if gold is None:
                    holds["no_author_stated_exact_numeric_endpoint"] += 1
                    continue
                eid = child.get("id")
                items.append({"source": OPENSTAX_REPO, "source_split": f"{module}/publisher",
                    "source_index": eid, "original_source_id": f"{module}::{eid}",
                    "original_source": f"https://github.com/{OPENSTAX_REPO}/blob/{OPENSTAX_REVISION}/modules/{module}/index.cnxml",
                    "source_revision": OPENSTAX_REVISION, "subject": "openstax", "level": None,
                    "steps": None, "problem": text, "gold_raw": gold, "solution": reference,
                    "source_problem_xml": ElementTree.tostring(problem, encoding="unicode"),
                    "source_solution_xml": ElementTree.tostring(solution, encoding="unicode"),
                    "source_context_ids": [p.get("id") for p in current], "source_gold_locator": locator,
                    "source_module_uuid": root.find("{http://cnx.rice.edu/cnxml}metadata/{http://cnx.rice.edu/mdml}uuid").text,
                    "rendering_changes": "Original author wording plus original shared instructions/scenario; MathML rendered structurally to LaTeX; no generated numbers or questions."})
            elif child.tag not in (CNX + "solution", CNX + "commentary"):
                walk(child, () if child.tag == CNX + "section" else current)

    walk(root)
    if audit is not None:
        audit.update(module=module, exercise_nodes=sum(n.tag == CNX + "exercise" for n in root.iter()),
                     extracted_scalar_records=len(items), holds=dict(holds))
    return items


def load_sources():
    paths, provenance = {}, {}
    prior_manifest = read_json(ROOT / "data" / "v2" / "dataset_manifest.json")
    verify_hashes(prior_manifest["source_sha256"])
    for key, (repo, revision, names) in previous.SOURCES.items():
        paths[key] = [previous.RAW / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision / n
                      for n in names]
        provenance[key] = {"repo": repo, "revision": revision, "files": [relative(p) for p in paths[key]],
                           "split_policy": "test only" if key != "amc" else
                           "mirror split named train contains held-out AMC exams, not GSM8K train"}
        card = RAW / "source_metadata" / key / revision / "README.md"
        fetch(card, f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md")
        provenance[key].update(license="Apache-2.0" if key == "amc" else "MIT",
                              license_evidence=relative(card),
                              provenance_limitation="Distributor declaration; not a proof of model decontamination or upstream exam rights.")
        if key == "amc":
            provenance[key]["answers"] = "83 modified AMC12 2022-2023 questions with adapted integer answers; not necessarily the original exam answer."
    pool = previous.gsm8k_items(paths["gsm8k"]) + previous.math_items(paths["math"])
    import pandas as pd

    amc = pd.read_parquet(paths["amc"][0])
    competition = previous.competition_items(paths["amc"])
    for item in competition:
        if item["subject"] == "amc12":
            row = amc.iloc[item["source_index"]]
            url = str(row["url"])
            item.update(original_source_id=url, original_source=url,
                        exam_identity=url.split("/")[-1], mirror_row_id=str(row.get("id", "")))
        else:
            item.update(original_source_id=f"{item['source_split']}::Problem_{item['source_index']}",
                        exam_identity=item["source_split"], original_source="AoPS via pinned local math_agent copy")
        pool.append(item)
    for item in pool:
        key = "gsm8k" if item["subject"] == "gsm8k" else "amc" if item["subject"] == "amc12" else "math"
        item["source_revision"] = (previous.SOURCES[key][1] if item["subject"] != "aime"
                                   else "sha256-pinned-local-copy")
        item.setdefault("original_source_id", source_id(item))
    asdiv_base = RAW / "asdiv" / ASDIV_REVISION
    for name in ("dataset/ASDiv.xml", "README.md"):
        fetch(asdiv_base / name, f"https://raw.githubusercontent.com/{ASDIV_REPO}/{ASDIV_REVISION}/{name}")
    if "CC BY-NC 4.0" not in (asdiv_base / "README.md").read_text(encoding="utf-8"):
        raise ValueError("Official ASDiv pinned revision lacks expected license declaration")
    if digest(asdiv_base / "dataset" / "ASDiv.xml") != "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929":
        raise ValueError("Official ASDiv corpus bytes do not match the independently acquired pin")
    pool += asdiv_items(asdiv_base / "dataset" / "ASDiv.xml")
    provenance["asdiv"] = {"repo": ASDIV_REPO, "revision": ASDIV_REVISION,
        "files": [relative(asdiv_base / n) for n in ("dataset/ASDiv.xml", "README.md")],
        "license": "CC-BY-NC-4.0", "license_evidence": relative(asdiv_base / "README.md"),
        "license_limitation": "Noncommercial research only; preserve original Source and attribution.",
        "answers": "Official author-annotated Answer and Formula; still require independent reviews.",
        "citation": "Miao, Liang and Su (ACL 2020), A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers",
        "authoritative_url": f"https://github.com/{ASDIV_REPO}/tree/{ASDIV_REVISION}"}
    provenance["aime"] = {"files": ["data/raw/" + n for n in previous.AIME_FILES],
        "revision": "local SHA256 pins inherited from v1/v2",
        "answers": "Existing math_agent copies of AIME 2024/2025 key; no detailed solution in local source",
        "license": "Not established by local copy; inherited source, not a newly licensed corpus",
        "limitation": "Review must solve independently; provenance does not assert a newly verified official key.",
        "known_answer_authority_issue": "Sibling math_agent metadata reports AIME I 2024 problem 12 uses override 385 versus claimed official 384; do not assert local key is official."}
    hashes = dict(prior_manifest["source_sha256"])
    hashes.update({relative(p): digest(p) for p in asdiv_base.rglob("*") if p.is_file()})
    hashes.update({relative(p): digest(p) for p in (RAW / "source_metadata").rglob("*") if p.is_file()})
    openstax_base = RAW / "openstax" / OPENSTAX_REVISION
    openstax_files = ["README.md", "LICENSE", "collections/college-algebra-2e.collection.xml",
                      "modules/m63490/index.cnxml", *[f"modules/{m}/index.cnxml" for m in OPENSTAX_MODULES]]
    for name in openstax_files:
        fetch(openstax_base / name, f"https://raw.githubusercontent.com/{OPENSTAX_REPO}/{OPENSTAX_REVISION}/{name}")
    license_text = (openstax_base / "LICENSE").read_text(encoding="utf-8")
    collection = (openstax_base / "collections" / "college-algebra-2e.collection.xml").read_text(encoding="utf-8")
    if ("Attribution-NonCommercial-ShareAlike 4.0 International" not in license_text
            or "creativecommons.org/licenses/by-nc-sa/4.0" not in collection):
        raise ValueError("OpenStax pinned license does not establish expected CC-BY-NC-SA-4.0")
    provenance["openstax"] = {"repo": OPENSTAX_REPO, "revision": OPENSTAX_REVISION,
        "files": [relative(openstax_base / n) for n in openstax_files],
        "license": "CC-BY-NC-SA-4.0", "license_evidence": relative(openstax_base / "LICENSE"),
        "license_limitation": "Noncommercial research, attribution and ShareAlike; preserve source XML and identify rendering changes.",
        "attribution": "OpenStax, Rice University; College Algebra 2e; lead author Jay Abramson and contributing authors.",
        "answers": "Publisher exercise solution nodes, never generated answer variants.",
        "authoritative_url": f"https://github.com/{OPENSTAX_REPO}/tree/{OPENSTAX_REVISION}"}
    hashes.update({relative(openstax_base / n): digest(openstax_base / n) for n in openstax_files})
    provenance["openstax"]["extraction_audit"] = {}
    for module in OPENSTAX_MODULES:
        module_audit = {}
        pool += publisher_items(openstax_base / "modules" / module / "index.cnxml", module_audit)
        provenance["openstax"]["extraction_audit"][module] = module_audit
    return pool, provenance, hashes


def tokenizer_counter():
    from jevlab.prompts import load_fast_tokenizer_file, render_generator_prompt

    system = previous.load_prompt(ROOT / "prompts" / "generador.txt")
    metadata, tokenizers = {}, {}
    for key, (repo, revision) in MODELS.items():
        base = (ROOT / "data" / "v2" / "model_meta" / "G4" if key == "G4" else
                previous.RAW / "model_meta" / "B" if key == "B13" else RAW / "model_meta" / key / revision)
        if key == "O9":
            for name in (*META_FILES, "config.json", "README.md", "LICENSE"):
                fetch(base / name, f"https://huggingface.co/{repo}/resolve/{revision}/{name}")
            verify_metadata_bytes(base, O9_FILE_PINS)
        else:
            old = read_json(ROOT / "data" / "v2" / "dataset_manifest.json")["tokenizers"]["G4" if key == "G4" else "B"]
            if old["revision"] != revision:
                raise ValueError(f"Previous {key} revision does not match required pin")
            verify_metadata_bytes(base, {n: ("sha256", old["sha256"][n]) for n in META_FILES})
        metadata[key] = {"repo": repo, "revision": revision, "directory": relative(base),
                         "sha256": {n: digest(base / n) for n in META_FILES}}
        tokenizers[key] = load_fast_tokenizer_file(str(base))

    def count(problem):
        return {key: len(tok.encode(render_generator_prompt(tok, system, problem), add_special_tokens=False))
                for key, tok in tokenizers.items()}

    return count, metadata


def verify_metadata_bytes(base, pins):
    for name, (algorithm, expected) in pins.items():
        content = (base / name).read_bytes()
        actual = (digest_bytes(content) if algorithm == "sha256" else
                  hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest())
        if actual != expected:
            raise ValueError(f"Tokenizer metadata does not match authoritative revision bytes: {base}/{name}")


def numeric_gold(raw):
    """Only explicit numeric gold; unit stripping is recorded, never expression evaluation."""
    if raw is None:
        return None, None
    raw = unicodedata.normalize("NFKC", str(raw)).strip()
    text = raw
    unit = None
    # Dollar math delimiters are not currency. An unpaired/escaped dollar is a unit.
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    if text.startswith("\\$") or text.startswith("$"):
        unit, text = "dollars", re.sub(r"^(?:\\\$|\$)\s*", "", text)
    match = re.fullmatch(r"(.+?)\s*(?:\\text\{\s*([^{}]+)\s*\}|(dollars?|cents?|euros?|USD|EUR|GBP|percent|\\?%))", text, re.I)
    if match:
        text, unit = match.group(1).strip(), (match.group(2) or match.group(3)).strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    return previous.parse_latex_gold(text), unit


def family(item):
    return "openstax" if item["subject"] == "openstax" else "asdiv" if item["subject"] == "asdiv" else previous.family(item)


def provisional(item):
    text, subject = item["problem"], item["subject"]
    value, unit = numeric_gold(item.get("gold_raw"))
    if value is None:
        return None, "gold_not_exact_number"
    if abs(value) > MAX_ANSWER:
        return None, "gold_abs_gt_1e9"
    if FORMAT_EXCLUDE.search(text):
        return None, "image_choices_approximation_or_non_scalar_request"
    if NON_SCALAR_REQUEST.search(text) or text.count("?") > 1:
        return None, "non_scalar_original_request"
    if re.search(r"(?:find|give|determine) (?:the|an?|all) ordered pairs?", text, re.I):
        return None, "requests_pair_answer_not_scalar_count"
    domain, domain_rule = content_domain(item)
    if domain is None:
        return None, domain_rule
    if subject in ("gsm8k", "asdiv"):
        steps = item["steps"]
        if subject == "gsm8k":
            difficulty = "easy" if steps in (2, 3) else "medium" if steps in (4, 5) else "hard" if steps >= 7 else None
        else:
            difficulty = "easy" if 1 <= steps <= 3 else "medium" if 4 <= steps <= 6 else "hard" if steps >= 7 else None
        if difficulty is None:
            return None, "operation_count_outside_fixed_bands"
        if subject == "asdiv" and domain == "arithmetic" and steps < 2:
            return None, "arithmetic_domain_requires_multistep_not_single_operation"
        rule = f"{subject} annotated operations={steps}; provisional, not agent-reviewed"
    elif subject in ("amc12", "aime"):
        difficulty, rule = "hard", f"{subject} exam; provisional, not agent-reviewed"
    elif subject == "openstax":
        scalar_text = text.replace("$", "")
        constrained = re.search(r"must form a string|must end|must begin|"
                                r"at least[^.]*\bor\b|(?:choose|select)[^.]*\d+[^.]*,\s*\d+", scalar_text, re.I)
        sampled_groups = (re.search(r"grabs? \d+|selects? \d+|chooses? \d+", scalar_text, re.I)
                          and re.search(r"probability of (?:getting|choosing|selecting) \d+ \w+", scalar_text, re.I))
        coupled_variation = re.search(r"varies jointly|varies directly[^.]*inversely", scalar_text, re.I)
        equal_distance = re.search(r"(?:commute|return trip)[^.]*(?:longer|less)", scalar_text, re.I)
        two_cohort_updates = (re.search(r"ratio", scalar_text, re.I)
                             and re.search(r"arrive|increase|enter", scalar_text, re.I)
                             and re.search(r"leave|decrease|depart|exit", scalar_text, re.I))
        multistage_ratio = domain == "ratios_percentages" and (coupled_variation or equal_distance or two_cohort_updates)
        difficulty = "medium" if constrained or sampled_groups or multistage_ratio else "easy"
        rule = "Publisher scalar: medium only constrained repeated-letter/grouped selection/overlap/multi-category sampling or coupled variation/equal-distance/two-cohort updates; direct formulas easy; no hard grade inflation"
    else:
        level = item["level"]
        difficulty = "easy" if level in (1, 2) else "medium" if level == 3 else "hard" if level in (4, 5) else None
        if difficulty is None:
            return None, "math_level_missing"
        rule = f"MATH {subject} Level {level}; provisional, not agent-reviewed"
    return {"provisional_domain": domain, "provisional_difficulty": difficulty, "difficulty_rule": rule,
            "classification_rule": domain_rule,
            "gold_numerator": value.numerator, "gold_denominator": value.denominator,
            "gold_answer": previous.fraction_to_str(value), "answer_unit": item.get("answer_unit") or unit}, None


def capacity(rows):
    return {domain: {diff: sum(r["provisional_domain"] == domain and r["provisional_difficulty"] == diff
                              for r in rows) for diff in TEST_QUOTA} for domain in DOMAINS}


def diverse_order(rows):
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for row in sorted(rows, key=source_id):
        buckets[(row["provisional_domain"], row["provisional_difficulty"], row["source_family"])].append(row)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    queues = {key: deque(value) for key, value in buckets.items()}
    ordered = []
    for domain in DOMAINS:
        for diff in TEST_QUOTA:
            keys = sorted(k for k in queues if k[:2] == (domain, diff))
            while any(queues[k] for k in keys):
                for key in keys:
                    if queues[key]:
                        ordered.append(queues[key].popleft())
    return ordered


def discover(pool, used, extra, count_tokens):
    excluded, candidates = Counter(), []
    old_index, extra_index = DuplicateIndex(used), DuplicateIndex(extra)
    used_ids, extra_ids = {r["source_id"] for r in used}, {r["source_id"] for r in extra}
    for item in sorted(pool, key=source_id):
        sid = source_id(item)
        if sid in used_ids or sid in extra_ids:
            excluded["used_source_id" if sid in used_ids else "rejected_archive_source_id"] += 1
            continue
        assigned, reason = provisional(item)
        if reason:
            excluded[reason] += 1
            continue
        reason = old_index.match(item["problem"]) or extra_index.match(item["problem"])
        if reason:
            excluded["previous_or_archive_" + reason] += 1
            continue
        counts = count_tokens(item["problem"])
        if set(counts) != set(MODELS) or any(type(v) is not int or v < 1 for v in counts.values()):
            raise ValueError("Token counter must measure all three pinned rendered prompts")
        if max(counts.values()) > MAX_TOKENS:
            excluded["rendered_prompt_gt_1024"] += 1
            continue
        candidates.append({**item, **assigned, "raw_problem": item["problem"], "language": "en",
                           "source_id": sid, "source_family": family(item), "prompt_tokens": counts})
    filtered = []
    selected_index = DuplicateIndex()
    for row in diverse_order(candidates):
        reason = selected_index.match(row["problem"])
        if reason:
            excluded["candidate_pool_" + reason] += 1
            continue
        selected_index.add(row["problem"])
        row["candidate_sha256"] = candidate_sha(row)
        row["id"] = "jev-v3-c-" + row["candidate_sha256"][:16]
        filtered.append(row)
    if len({r["id"] for r in filtered}) != len(filtered):
        raise ValueError("Candidate ID collision")
    return filtered, {"raw_pool_count": len(pool), "eligible_before_pool_dedup": len(candidates),
                      "candidate_pool_count": len(filtered), "per_stratum": capacity(filtered),
                      "source_family_counts": dict(Counter(r["source_family"] for r in filtered)),
                      "exclusions": dict(excluded)}


def review_priority(rows):
    wanted = {(d, f): TEST_QUOTA[f] + PILOT_QUOTA[f] for d in DOMAINS for f in TEST_QUOTA}
    initial, reserve = [], []
    for row in rows:
        key = (row["provisional_domain"], row["provisional_difficulty"])
        if wanted[key]:
            initial.append(row)
            wanted[key] -= 1
        else:
            reserve.append(row)
    backup, remainder = [], []
    backup_target = {(d, f): 50 if (d, f) in (("ratios_percentages", "hard"), ("counting_probability", "medium"))
                     else 4 for d in DOMAINS for f in TEST_QUOTA}
    for row in reserve:
        key = (row["provisional_domain"], row["provisional_difficulty"])
        if backup_target[key]:
            backup.append(row)
            backup_target[key] -= 1
        else:
            remainder.append(row)
    rng = random.Random(SEED)
    rng.shuffle(initial)
    rng.shuffle(backup)
    return initial + backup + remainder, len(initial), [{"domain": d, "difficulty": f, "missing": n}
                                             for (d, f), n in wanted.items() if n]


def review_policy():
    return {"policy_id": REVIEW_POLICY_ID, "contract": "REVIEW_CONTRACT.md",
        "contract_sha256": digest(OUT / "REVIEW_CONTRACT.md"),
        "eligibility_document_sha256": digest(OUT / "ELIGIBILITY_POLICY.md"),
        "method": "Two actual independent blind solutions and separate source comparisons; intrinsic label agreement suffices despite source hints. Actual third-agent adjudication for reviewer disagreement/retained eligible vetoes; all original verdicts retained.",
        "quota_axis": QUOTA_AXIS,
        "sampling_tier": "ORIGINAL candidate provisional_difficulty from the unchanged published/source-proxy rule; not a guarantee of intrinsic difficulty. Selection uses genuine reviewed domain plus this immutable tier.",
        "scope_integrity": {"rule_id": SCOPE_RULE_ID, "incident_ledger": "reviews/" + SCOPE_LEDGER,
            "threshold": "A documented actual access outside assigned inputs disqualifies all candidates of the affected execution, independent of outcomes, exact material or claimed nonuse.",
            "reason": "review_scope_breach", "original_verdicts_and_registrations_immutable": True,
            "replacement_reviews_cannot_reverse_disqualification": True},
        "required_independent_reviewers": 2, "human_review_required": False,
        "reviewed_field_means_human_review": True, "dev_reuse_not_new_agent_review": True,
        "domain_policy": {"version": "content-first-global-before-sampling-20261003",
            "applies_to": "Every source subject, every difficulty, before randomization; never source-ID, prefix, quota or scarcity exceptions.",
            "precedence": ["actual geometric reasoning excluded", "stochastic events",
                           "substantive number theory unless primary arrangements/strings/subsets", "formal algebra", "combinatorial structure",
                           "central shares/percentages/unit rates/direct or inverse variation", "remaining source-subject hint"],
            "ratio_boundaries": "Rational answer, incidental percent, polynomial root ratio, gcd ratio or probability alone is not proportional domain evidence. Equations may express real-world rate/mixture/share relations.",
            "geometry_boundaries": "No blanket square/point/line/distance substring exclusion. Perfect powers, integer solutions, scoring points and travel-rate distances are not shape measurement.",
            "arithmetic": "Genuine non-symbolic non-proportional multistep arithmetic; single annotated ASDiv operation excluded.",
            "regex_evidence": {"counting": COUNTING_CONTENT.pattern, "number_theory": NT_CONTENT.pattern,
                               "formal_algebra": FORMAL_ALGEBRA.pattern, "quantities": QUANTITIES.pattern}},
        "difficulty_policy": {"gsm8k": "2-3 calculations easy,4-5 medium,>=7 hard; other bands excluded",
            "math": "Original Level1-2 easy,Level3 medium,Level4-5 hard, all provisional and independently assessed",
            "asdiv": "Annotated operations1-3 easy,4-6 medium,>=7 hard; grade never inflates difficulty; arithmetic requires>=2",
            "competition": "Hard source sampling tier only; intrinsic reviewers may call a routine contest question easy without changing the source tier",
            "openstax": "Direct formula applications easy; constrained repeated-letter arrangements/grouped selections/overlap/multi-category sampling or coupled-variation/equal-distance/two-cohort update modeling medium; no textbook hard inflation",
            "disagreement": "Fresh fully approving reviewers agreeing on intrinsic domain/difficulty need no source-hint adjudication. Reviewer label disagreements or eligible old false verdicts require actual adjudication. Invalid mathematics, ambiguity and source-gold mismatches remain excluded."},
        "adjudication": {"required_actual_independent_agents": 1, "original_reviews_immutable": True,
            "final_labels": "Intrinsic mathematical domain/difficulty, independent of quota, source ID or shortage; preserve source provisional labels and candidate/packet hashes.",
            "units": "An original unit_explicit=false is resolvable only for an inherently dimensionless explicitly identified unique query, with both blind mathematics valid and original false/reason retained.",
            "gold": "Answer must equal ORIGINAL source gold; never generate or overwrite gold.",
            "vetoes": "No override of unsolved, ambiguous, incorrect-math, gold_matches=false or solution_consistent=false."},
        "eligibility": {"language": "en", "exact_integer_fraction_finite_decimal": True,
            "max_abs_answer": MAX_ANSWER, "max_rendered_tokens_each_pinned_G4_B13_O9": MAX_TOKENS,
            "money_and_percent_allowed": True, "dollar_math_delimiters_not_currency_exclusions": True,
            "units": "Reviewers must identify explicit unique requested unit. Dollars/cents distinguished; numeric percentages stay percentage points, not silently divided by100. Count/probability answers can be dimensionless.",
            "non_scalar": "Scalar counts of ordered pairs allowed; requests to give an ordered pair, multipart outputs, diagrams or approximate quantities excluded",
            "publisher_extraction": "Original author scalar nodes and original context only; preserve source XML/IDs and structured MathML. Missing, rounded or nonnumeric publisher gold excluded; never generate variants."},
        "seeds": {"construction": SEED, "latency": LATENCY_SEED},
        "quota": {"test": TEST_QUOTA, "pilot": PILOT_QUOTA, "latency": LATENCY_QUOTA}}


def public_manifest(metadata, input_hashes, policy, analysis_hash=None, complete=False, intrinsic_counts=None):
    """Whitelist public provenance. Never expand private preparation/review metadata."""
    if complete and (not isinstance(intrinsic_counts, dict) or set(intrinsic_counts) != {"test", "pilot"}
                     or any(set(intrinsic_counts[s]) != set(TEST_QUOTA)
                            or any(type(n) is not int or n < 0 for n in intrinsic_counts[s].values())
                            or sum(intrinsic_counts[s].values()) != count for s, count in (("test", 500), ("pilot", 50)))):
        raise ValueError("Public publication requires actual selected intrinsic difficulty counts, not source quotas")
    return {"dataset_version": "v3", "sealed": complete,
        "status": "FROZEN" if complete else "UNREVIEWED_PREPARATION_ONLY",
        "construction_seed": SEED, "latency_seed": LATENCY_SEED,
        "actual_n": {"test": 500 if complete else 0, "pilot": 50 if complete else 0,
                     "dev": 20, "latency": 100 if complete else 0},
        "requested_n": {"test": 500, "pilot": 50, "dev": 20, "latency": 100},
        "quota": {"test": TEST_QUOTA, "pilot": PILOT_QUOTA, "latency": LATENCY_QUOTA},
        "quota_axis": QUOTA_AXIS,
        "input_sha256": input_hashes, "builder_sha256": metadata["builder_sha256"],
        "review": {"agent_reviewed_all": complete, "human_reviewed": False, "reviewed": False,
                   "scope": ["test", "pilot"], "dev_agent_reviewed": False,
                    "policy_id": policy["policy_id"], "policy_sha256": policy["contract_sha256"],
                    "intrinsic_difficulty_counts": intrinsic_counts,
                    "source_sampling_tier_is_intrinsic_difficulty": False},
        "analysis_manifest_sha256": analysis_hash,
        "policy_artifact_sha256": {"review_policy.json": metadata.get("review_policy_json_sha256"),
                                   "ELIGIBILITY_POLICY.md": policy.get("eligibility_document_sha256")},
        "source_revisions": {key: {k: source[k] for k in ("repo", "revision", "license") if k in source}
                             for key, source in metadata["sources"].items()},
        "test_reserved_before_pilot": True, "latency_test_subset_only": True,
        "max_prompt_tokens": MAX_TOKENS, "max_abs_answer": MAX_ANSWER}


def verify_preparation_lineage(stage, metadata=None, *, policy_inputs=None):
    """Verify original source-construction pins plus an explicit policy-only amendment.

    Returns the resolved protected inventory (old contract pins resolve to archive).
    Consumers must use this rather than comparing archived contract pins to new docs.
    """
    metadata = metadata or read_json(stage / "preparation_manifest.json")
    hashes = verify_seal(stage)
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()
              and not p.is_relative_to(stage / "reviews") and p != stage / "SHA256SUMS"}
    if actual != set(hashes):
        raise ValueError("Staging inventory has unexpected/unsealed files")
    builder_path = Path(__file__) if policy_inputs is None else policy_inputs / "data/build_dataset_v3.py"
    if (metadata["construction_seed"] != SEED or metadata["latency_seed"] != LATENCY_SEED
            or metadata["builder_sha256"] != digest(builder_path)):
        raise ValueError("Staged builder/seed changed; explicit policy amendment required")
    protected = dict(metadata["previous_artifact_sha256"])
    if policy_inputs is not None:
        for name in ("data/v3/REVIEW_CONTRACT.md", "data/v3/ELIGIBILITY_POLICY.md", "data/build_dataset_v3.py"):
            if name in protected and digest(policy_inputs / name) == protected[name]:
                protected[relative(policy_inputs / name)] = protected.pop(name)
    current, current_stage, seen = metadata, stage, set()
    lineage = current.get("previous_preparation")
    while lineage:
        archive = ROOT / lineage["archive"]
        if (not archive.resolve().is_relative_to((ROOT / "data/v3/policy_amendments").resolve())
                or archive.is_symlink() or archive.resolve() in seen):
            raise ValueError("Invalid preparation lineage archive")
        seen.add(archive.resolve())
        inventory = verify_seal(archive)
        if set(inventory) != {p.relative_to(archive).as_posix() for p in archive.rglob("*")
                              if p.is_file() and p != archive / "SHA256SUMS"}:
            raise ValueError("Unsealed preparation archive member")
        snapshot = read_json(archive / "snapshot.json")
        if digest(archive / "snapshot.json") != lineage["snapshot_sha256"]:
            raise ValueError("Preparation snapshot changed")
        if {k: v for k, v in inventory.items() if k != "snapshot.json"} != snapshot["sha256"]:
            raise ValueError("Snapshot inventory differs from archived bytes")
        old_stage = archive / "data/v3/staging"
        verify_seal(old_stage)
        old = read_json(old_stage / "preparation_manifest.json")
        old_policy = read_json(old_stage / "review_policy.json")
        pins = {"previous_seal_sha256": digest(old_stage / "SHA256SUMS"),
                "builder_sha256": digest(archive / "data/build_dataset_v3.py"),
                "contract_sha256": digest(archive / "data/v3/REVIEW_CONTRACT.md"),
                "eligibility_document_sha256": digest(archive / "data/v3/ELIGIBILITY_POLICY.md")}
        if (any(lineage.get(k) != v for k, v in pins.items()) or old["builder_sha256"] != pins["builder_sha256"]
                or old["review"]["policy_sha256"] != pins["contract_sha256"]
                or old_policy["contract_sha256"] != pins["contract_sha256"]
                or old_policy["eligibility_document_sha256"] != pins["eligibility_document_sha256"]):
            raise ValueError("Original archived builder/contract pins changed")
        current_policy = read_json(current_stage / "review_policy.json")
        if (digest(current_stage / "review_policy.json") != current["review_policy_json_sha256"]
                or current_policy["contract_sha256"] != current["review"]["policy_sha256"]
                or current_policy["policy_id"] != current["review"]["policy_id"]):
            raise ValueError("Historical policy descriptors changed")
        expected = {**old, "builder_sha256": current["builder_sha256"],
                    "review_policy_json_sha256": current["review_policy_json_sha256"],
                    "review": {**old["review"], "policy_id": current["review"]["policy_id"],
                               "policy_sha256": current["review"]["policy_sha256"]},
                    "previous_preparation": lineage}
        if "quota_axis" in current:
            if current["quota_axis"] != current_policy.get("quota_axis"):
                raise ValueError("Sampling quota axis differs from declared policy")
            expected["quota_axis"] = current["quota_axis"]
        if current != expected:
            raise ValueError("Policy amendment changed source-construction metadata")
        mutable = {"preparation_manifest.json", "review_policy.json", "public_manifest_preview.json", "SHA256SUMS",
                   "reviews/review_progress.json", "reviews/reviewers.json", "reviews/adjudications/adjudicators.json",
                   "reviews/adjudication_registry.json", "reviews/" + SCOPE_LEDGER}
        for name, expected_sha in snapshot["sha256"].items():
            if name.startswith("data/v3/staging/"):
                member = name.removeprefix("data/v3/staging/")
                if member not in mutable and digest(current_stage / member) != expected_sha:
                    raise ValueError("Original candidate/packet/review evidence changed: " + member)
        old_registry = read_json(old_stage / "reviews/reviewers.json")["reviewers"]
        registry = read_json(current_stage / "reviews/reviewers.json")["reviewers"]
        if registry[:len(old_registry)] != old_registry:
            raise ValueError("Original actual reviewer registrations changed")
        old_adj = old_stage / "reviews/adjudications/adjudicators.json"
        if old_adj.exists():
            entries = read_json(old_adj)["adjudicators"]
            current_entries = read_json(current_stage / "reviews/adjudications/adjudicators.json")["adjudicators"]
            if current_entries[:len(entries)] != entries:
                raise ValueError("Original actual adjudicator registrations changed")
        old_mirror = old_stage / "reviews/adjudication_registry.json"
        if old_mirror.exists():
            mirror = read_json(old_mirror)
            live = read_json(current_stage / "reviews/adjudication_registry.json")
            if ({k: v for k, v in mirror.items() if k != "adjudicators"} != {k: v for k, v in live.items() if k != "adjudicators"}
                    or live["adjudicators"][:len(mirror["adjudicators"])] != mirror["adjudicators"]):
                raise ValueError("Original coordinator adjudication registry changed")
        old_incidents = old_stage / "reviews" / SCOPE_LEDGER
        if old_incidents.exists() and not (current_stage / "reviews" / SCOPE_LEDGER).read_bytes().startswith(old_incidents.read_bytes()):
            raise ValueError("Scope incident ledger is append-only; original incident bytes changed")
        for name in ("data/v3/REVIEW_CONTRACT.md", "data/v3/ELIGIBILITY_POLICY.md", "data/build_dataset_v3.py"):
            if name in protected and digest(archive / name) == protected[name]:
                protected[relative(archive / name)] = protected.pop(name)
        current, current_stage = old, old_stage
        lineage = current.get("previous_preparation")
    verify_hashes(protected)
    verify_hashes(metadata["source_sha256"])
    verify_hashes({metadata["source_lock_path"]: metadata["source_lock_sha256"]})
    policy = read_json(stage / "review_policy.json")
    if (digest(stage / "review_policy.json") != metadata["review_policy_json_sha256"]
            or policy["contract_sha256"] != metadata["review"]["policy_sha256"]
            or policy["policy_id"] != metadata["review"]["policy_id"]
            or metadata.get("quota_axis") != policy.get("quota_axis")
            or (policy_inputs is None and policy != review_policy())
            or (policy_inputs is not None and (policy["contract_sha256"] != digest(policy_inputs / "data/v3/REVIEW_CONTRACT.md")
                or policy["eligibility_document_sha256"] != digest(policy_inputs / "data/v3/ELIGIBILITY_POLICY.md")))):
        raise ValueError("Declared review policy/contract changed")
    return protected


def replace_generated(path, value):
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o644)
    pending.replace(path)


def amend_review_policy_before_inference(reason, archive, *, technical_scope_repair=False):
    """Policy-only migration; candidates, packets, raw events and old SHA pins stay intact."""
    stage = OUT / "staging"
    if (OUT / "SHA256SUMS").exists() or any(OUT.glob("*_inputs.jsonl")) or any(OUT.glob("*_gold.jsonl")):
        raise ValueError("Policy amendment forbidden after final/partial publication")
    status_path = ROOT / "results/v3/implementation_status.json"
    if not status_path.is_file() or status_path.is_symlink():
        raise ValueError("Policy amendment requires explicit no-inference/no-results status")
    for path in (ROOT / "results/v3").rglob("*"):
        if path.is_file():
            if technical_scope_repair and path.relative_to(ROOT / "results/v3").as_posix() in ("review_manager_state.json", "review_manager_state.md"):
                manager = ROOT / "results/v3/review_manager_state.json"
                state = read_json(manager)
                if (path.is_symlink() or manager.is_symlink() or state.get("freeze_not_attempted") is not True
                        or state.get("operator_a100_hours_spent_this_session") != 0
                        or state.get("policy_id") != REVIEW_POLICY_ID or state.get("policy_sha256") != digest(OUT / "REVIEW_CONTRACT.md")
                        or any(state.get(k, False) is not False for k in ("v3_model_inference_executed", "pilot_executed", "dataset_frozen"))):
                    raise ValueError("Technical repair requires verified no-inference review coordination state")
                continue  # Review coordination records are not inference/results artifacts.
            if path.name != "implementation_status.json":
                raise ValueError("Policy amendment forbidden after inference/results exist")
            status = read_json(path)
            if any(status.get(k) is not False for k in ("v3_model_inference_executed", "pilot_executed", "dataset_frozen")):
                raise ValueError("Policy amendment requires explicit no-inference/no-results status")
    if any((OUT / name).exists() for name in ("inference", "results")):
        raise ValueError("Policy amendment forbidden after inference/results exist")
    if not review_text(reason, 40):
        raise ValueError("Detailed user-authorized amendment reason required")
    archive = (ROOT / archive).resolve()
    if not archive.is_relative_to((OUT / "policy_amendments").resolve()):
        raise ValueError("Amendment requires an archived original preparation")
    archived_hashes = verify_seal(archive)
    snapshot = read_json(archive / "snapshot.json")
    archive_files = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file() and p != archive / "SHA256SUMS"}
    if (archive_files != set(archived_hashes)
            or {k: v for k, v in archived_hashes.items() if k != "snapshot.json"} != snapshot["sha256"]):
        raise ValueError("Invalid original snapshot inventory")
    metadata = read_json(stage / "preparation_manifest.json")
    if (metadata["review"]["policy_id"] == REVIEW_POLICY_ID
            and metadata["review"]["policy_sha256"] == digest(OUT / "REVIEW_CONTRACT.md")
            and metadata["builder_sha256"] == digest(Path(__file__))):
        raise ValueError("Policy amendment already applied; refusing the same migration")
    hashes = verify_seal(stage)
    old_files = {relative(p): digest(p) for p in stage.rglob("*") if p.is_file()}
    if old_files != {k: v for k, v in snapshot["sha256"].items() if k.startswith("data/v3/staging/")}:
        raise ValueError("Staging/reviews changed since original snapshot; preserve a fresh complete snapshot")
    sealed_files = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()
                    and not p.is_relative_to(stage / "reviews") and p != stage / "SHA256SUMS"}
    if sealed_files != set(hashes) or metadata["construction_seed"] != SEED or metadata["latency_seed"] != LATENCY_SEED:
        raise ValueError("Policy amendment cannot change the original sealed inventory or seeds")
    old_stage = archive / "data/v3/staging"
    verify_seal(old_stage)
    if read_json(old_stage / "preparation_manifest.json") != metadata:
        raise ValueError("Archived original preparation manifest differs from current staging")
    old_policy = read_json(old_stage / "review_policy.json")
    policy = review_policy()
    if technical_scope_repair and (old_policy != {k: v for k, v in policy.items() if k != "scope_integrity"}
                                  and old_policy != policy):
        raise ValueError("Technical scope repair cannot alter the contract, quota, source or mathematical review policy")
    lineage = {"archive": relative(archive), "snapshot_sha256": digest(archive / "snapshot.json"),
               "previous_seal_sha256": digest(old_stage / "SHA256SUMS"),
               "builder_sha256": digest(archive / "data/build_dataset_v3.py"),
               "contract_sha256": digest(archive / "data/v3/REVIEW_CONTRACT.md"),
                "eligibility_document_sha256": digest(archive / "data/v3/ELIGIBILITY_POLICY.md"), "reason": reason}
    if technical_scope_repair:
        lineage["technical_repair"] = SCOPE_RULE_ID
    if (metadata["builder_sha256"] != lineage["builder_sha256"]
            or metadata["review"]["policy_sha256"] != lineage["contract_sha256"]
            or old_policy["contract_sha256"] != lineage["contract_sha256"]
            or old_policy["eligibility_document_sha256"] != lineage["eligibility_document_sha256"]):
        raise ValueError("Snapshot does not match original preparation builder/contracts")
    protected = dict(metadata["previous_artifact_sha256"])
    for name in ("data/v3/REVIEW_CONTRACT.md", "data/v3/ELIGIBILITY_POLICY.md", "data/build_dataset_v3.py"):
        if name in protected:
            protected[relative(archive / name)] = protected.pop(name)
    verify_hashes(protected)
    verify_hashes(metadata["source_sha256"])
    verify_hashes({metadata["source_lock_path"]: metadata["source_lock_sha256"]})
    verify_preparation_lineage(stage, metadata, policy_inputs=archive)
    rows = read_jsonl(stage / "candidate_pool.jsonl")
    load_reviews(rows, stage, metadata["packets"])  # Verify ALL original emitted events before touching descriptors.
    descriptors = ("review_policy.json", "preparation_manifest.json", "public_manifest_preview.json")
    try:
        replace_generated(stage / "review_policy.json", policy)
        metadata = {**metadata, "builder_sha256": digest(Path(__file__)), "previous_preparation": lineage,
                    "quota_axis": QUOTA_AXIS,
                    "review_policy_json_sha256": digest(stage / "review_policy.json"),
                    "review": {**metadata["review"], "policy_id": policy["policy_id"], "policy_sha256": policy["contract_sha256"]}}
        replace_generated(stage / "preparation_manifest.json", metadata)
        replace_generated(stage / "public_manifest_preview.json", public_manifest(metadata, {}, policy))
        marker = stage / "SHA256SUMS"
        marker.chmod(0o644)
        marker.unlink()  # Old marker is byte-preserved and pinned in the sealed archive.
        seal(stage, [stage / name for name in hashes])
        verify_preparation_lineage(stage, metadata)
    except BaseException:
        # Restore only this migration's generated descriptors; no raw/candidate rollback.
        for name in (*descriptors, "SHA256SUMS"):
            target = stage / name
            if target.exists():
                target.chmod(0o644)
            shutil.copyfile(old_stage / name, target)
            pending = stage / (name + ".pending")
            if pending.exists():
                pending.chmod(0o644)
                pending.unlink()
        verify_seal(stage)
        raise
    return lineage


def archive_unreviewed_stage(stage, reason):
    if (OUT / "SHA256SUMS").exists() or any(OUT.glob("*_inputs.jsonl")):
        raise ValueError("Cannot repair a final or partially published dataset")
    verify_seal(stage)
    metadata = read_json(stage / "preparation_manifest.json")
    if metadata["construction_seed"] != SEED or metadata["latency_seed"] != LATENCY_SEED:
        raise ValueError("Staging repair cannot change the required seeds")
    if (read_json(stage / "reviews" / "reviewers.json") != {"reviewers": []}
            or any(p.is_file() and p.name != "reviewers.json" for p in (stage / "reviews").rglob("*"))):
        raise ValueError("Repair forbidden after any reviewer registration, record, or evidence")
    inventory = {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()}
    fingerprint = digest_bytes(canonical(inventory).encode("utf-8"))
    archived = RAW / "pre_review_repairs" / fingerprint
    archived.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(archived)
    return {"directory": relative(archived), "sha256": inventory, "reason": reason,
            "actual_review_records": 0, "construction_seed_unchanged": SEED,
            "latency_seed_unchanged": LATENCY_SEED}


def prepare(repair_reason=None):
    stage = OUT / "staging"
    interrupted = None
    repair = archive_unreviewed_stage(stage, repair_reason) if repair_reason else None
    if stage.exists() and not (stage / "preparation_manifest.json").exists() and not (stage / "SHA256SUMS").exists():
        if (stage / "reviews").exists():
            raise ValueError("Interrupted staging has review artifacts; never discard or replace reviewed candidates")
        inventory = {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()}
        fingerprint = digest_bytes(canonical(inventory).encode("utf-8"))
        archived = RAW / "interrupted_preparations" / fingerprint
        archived.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(archived)
        interrupted = {"directory": relative(archived), "sha256": inventory,
                       "reason": "Incomplete unreviewed preparation (no manifest/seal/review directory); retained before regeneration with same required seed."}
    if (OUT / "SHA256SUMS").exists() or stage.exists():
        raise ValueError("V3 staging/freeze already exists; refusing overwrite or alternate-seed retry")
    protected = protected_inventory()
    used, extra, dev, archives = load_exclusions()
    pool, sources, source_hashes = load_sources()
    counter, tokenizers = tokenizer_counter()
    rows, audit = discover(pool, used, extra, counter)
    rows, priority_count, shortfalls = review_priority(rows)
    primary_capacity = capacity(rows[:priority_count])
    available = capacity(rows)
    backup_counts = {d: {f: min(50 if (d, f) in (("ratios_percentages", "hard"), ("counting_probability", "medium"))
                                      else 4, available[d][f] - primary_capacity[d][f])
                        for f in TEST_QUOTA} for d in DOMAINS}
    priority_total = priority_count + sum(sum(values.values()) for values in backup_counts.values())
    policy = review_policy()
    verify_hashes(protected)
    write_jsonl(stage / "candidate_pool.jsonl", rows)
    write_jsonl(stage / "used_exclusion_registry.jsonl", sorted(used, key=lambda r: r["source_id"]))
    write_jsonl(stage / "extra_exclusion_registry.jsonl", sorted(extra, key=lambda r: r["source_id"]))
    write_jsonl(stage / "dev_reuse_inputs.jsonl", dev[0])
    write_jsonl(stage / "dev_reuse_gold.jsonl", dev[1])
    write_json(stage / "review_policy.json", policy)
    packets = []
    for start in range(0, len(rows), 25):
        batch = rows[start:start+25]
        stem = f"batch_{start // 25 + 1:03d}"
        blind = stage / "packets" / f"{stem}.blind.jsonl"
        reference = stage / "packets" / f"{stem}.reference.jsonl"
        write_jsonl(blind, [{k: r[k] for k in ("id", "problem", "language")} for r in batch])
        write_jsonl(reference, batch)
        packets.append({"batch": stem, "blind": relative(blind), "reference": relative(reference),
                        "blind_sha256": digest(blind), "reference_sha256": digest(reference),
                        "candidate_ids": [r["id"] for r in batch],
                        "priority_candidate_count": max(0, min(25, priority_count-start))})
        packets[-1]["backup_priority_count"] = max(0, min(start+25, priority_total) - max(start, priority_count))
    metadata = {"dataset_version": "v3-staging", "construction_seed": SEED, "latency_seed": LATENCY_SEED,
        "quota_axis": QUOTA_AXIS,
        **audit, "priority_review_count": priority_count, "priority_shortfalls": shortfalls,
        "priority_with_backup_count": priority_total, "backup_per_stratum": backup_counts,
        "quota": {"test": TEST_QUOTA, "pilot": PILOT_QUOTA, "latency": LATENCY_QUOTA},
        "sources": sources, "source_sha256": source_hashes, "tokenizers": tokenizers,
        "builder_sha256": digest(Path(__file__)), "previous_artifact_sha256": protected,
        "generator_prompt_sha256": digest(ROOT / "prompts" / "generador.txt"),
        "max_prompt_tokens": MAX_TOKENS, "max_abs_answer": MAX_ANSWER, "enable_thinking": False,
        "eligibility": {"money_percent_allowed": True, "explicit_unambiguous_answer_unit_required": True,
            "literal_dollar_math_delimiters_allowed": True, "answer_types": "integer/exact fraction/finite decimal",
            "labels": "Global content-first policy before sampling, with original subjects preserved; both actual reviewers must agree. No relabeling to fill quotas.",
            "asdiv_difficulty": "1-3 annotated operations easy; 4-6 medium; >=7 hard; not Grade-based padding",
            "math_prealgebra_arithmetic": "Only non-ratio, non-symbolic, non-geometry, non-number-theory, non-counting statements",
            "filters_sha256": digest(Path(__file__))},
        "duplicates": {"exact_normalized_including_short": True, "eight_gram_threshold": 0.5,
            "eight_gram_denominator": "minimum unique span count", "numeric_template_exact": True,
            "numeric_template_approximate": "Shared 4-token skeleton span, length ratio >=0.85, SequenceMatcher >=0.90",
            "scope": "used union, both rejected archives, and candidate pool"},
        "exclusion_registry": {"used_unique_source_ids": len(used), "extra_unique_source_ids": len(extra),
            "extra_not_in_used": len({r['source_id'] for r in extra} - {r['source_id'] for r in used}),
            "archives": archives, "dev_reuse_still_excluded": True},
        "packets": packets, "review": {"agent_reviewed": False, "agent_reviewed_all": False,
            "human_reviewed": False, "reviewed": False, "policy_id": policy["policy_id"],
            "policy_sha256": policy["contract_sha256"]},
        "status": "STAGING_ONLY; final publication requires full quotas and two evidenced actual agents per candidate"}
    metadata["interrupted_preparation_preserved"] = interrupted
    metadata["pre_review_repair_preserved"] = repair
    metadata["review_policy_json_sha256"] = digest(stage / "review_policy.json")
    if repair:
        old = read_jsonl(ROOT / repair["directory"] / "candidate_pool.jsonl")
        before = {r["source_id"]: r for r in old}
        transitions = Counter()
        changes = []
        for row in rows:
            old_row = before.get(row["source_id"])
            if old_row and old_row["provisional_domain"] != row["provisional_domain"]:
                transitions[old_row["provisional_domain"] + "->" + row["provisional_domain"]] += 1
                changes.append({"source_id": row["source_id"], "old_domain": old_row["provisional_domain"],
                                "new_domain": row["provisional_domain"], "rule": row["classification_rule"],
                                "original_subject": row["subject"], "difficulty_unchanged": old_row["provisional_difficulty"] == row["provisional_difficulty"]})
        write_jsonl(stage / "classification_change_audit.jsonl", changes)
        metadata["global_pre_sampling_classification_changes"] = dict(transitions)
    source_lock = {"sources": sources, "sha256": source_hashes, "tokenizers": tokenizers}
    lock_path = RAW / "source_locks" / (digest_bytes(canonical(source_lock).encode("utf-8")) + ".json")
    if lock_path.exists():
        if read_json(lock_path) != source_lock:
            raise ValueError("Existing content-addressed source lock changed")
    else:
        write_json(lock_path, source_lock)
    metadata["source_lock_path"] = relative(lock_path)
    metadata["source_lock_sha256"] = digest(lock_path)
    write_json(stage / "preparation_manifest.json", metadata)
    write_json(stage / "public_manifest_preview.json", public_manifest(metadata, {}, policy))
    write_json(stage / "reviews" / "reviewers.json", {"reviewers": []})
    paths = [p for p in stage.rglob("*") if p.is_file() and not p.is_relative_to(stage / "reviews")]
    seal(stage, paths)
    verify_hashes(protected)
    print(json.dumps({k: metadata[k] for k in ("candidate_pool_count", "per_stratum", "priority_review_count", "priority_with_backup_count", "backup_per_stratum",
                                              "priority_shortfalls", "source_family_counts")}, indent=2))
    return metadata


def review_text(value, minimum):
    return isinstance(value, str) and len(value.strip()) >= minimum and len(set(value.split())) >= 8


def validate_review(candidate, blind, comparison, *, source_hint_agreement_required=False):
    """Validate content and binding, NOT a claim that JSON alone proves authorship."""
    for record in (blind, comparison):
        if record.get("id") != candidate["id"] or record.get("candidate_sha256") != candidate["candidate_sha256"]:
            raise ValueError("Review is not bound to the staged candidate SHA")
    if blind.get("reviewer_id") != comparison.get("reviewer_id"):
        raise ValueError("Blind solution and reference comparison have different reviewers")
    if blind.get("phase") != "blind" or comparison.get("phase") != "reference":
        raise ValueError("Two separate blind/reference records required")
    if not review_text(blind.get("independent_solution"), 80):
        raise ValueError("A detailed independent solution is required, not a verdict flag")
    if not review_text(comparison.get("reference_comparison"), 60):
        raise ValueError("A detailed source/reference comparison is required")
    answer = blind.get("answer", {})
    if (set(answer) != {"numerator", "denominator"}
            or type(answer.get("numerator")) is not int or type(answer.get("denominator")) is not int
            or answer["denominator"] <= 0):
        raise ValueError("Review answer must be an exact rational")
    domain_valid = blind.get("domain") in DOMAINS or (blind.get("domain") == "outside_scope" and blind.get("domain_ok") is False)
    difficulty_valid = blind.get("difficulty") in TEST_QUOTA or (blind.get("difficulty") == "unresolved" and blind.get("difficulty_ok") is False)
    if (not domain_valid or not difficulty_valid
            or not review_text(blind.get("domain_reason"), 40)
            or not review_text(blind.get("difficulty_reason"), 40)):
        raise ValueError("Independent domain/difficulty judgments with reasons required")
    for field in ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit"):
        if type(blind.get(field)) is not bool:
            raise ValueError(f"Missing boolean verdict: {field}")
    for field in ("gold_matches", "solution_consistent", "approve"):
        if type(comparison.get(field)) is not bool:
            raise ValueError(f"Missing reference verdict: {field}")
    if comparison.get("blind_record_sha256") != digest_bytes(canonical(blind).encode("utf-8")):
        raise ValueError("Reference comparison must bind the already completed blind record")
    approved = (all(blind[field] is True for field in ("statement_ok", "unique_answer", "correctness",
                  "domain_ok", "difficulty_ok", "unit_explicit"))
                and all(comparison[field] is True for field in ("gold_matches", "solution_consistent", "approve")))
    agrees = (Fraction(answer["numerator"], answer["denominator"]) ==
              Fraction(candidate["gold_numerator"], candidate["gold_denominator"]))
    if source_hint_agreement_required:
        agrees &= (blind["domain"] == candidate["provisional_domain"]
                   and blind["difficulty"] == candidate["provisional_difficulty"])
    if approved and not agrees:
        raise ValueError("Approved review disagrees with original source gold or its historical source-hint contract")
    return approved and agrees


def review_reasons(candidate, blind, ref):
    reasons = [field for field in ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")
               if blind[field] is False]
    reasons += [field for field in ("gold_matches", "solution_consistent", "approve") if ref[field] is False]
    for field in ("domain", "difficulty"):
        if blind[field] != candidate["provisional_" + field]:
            reasons.append(field + "_mismatch")
    if Fraction(**blind["answer"]) != Fraction(candidate["gold_numerator"], candidate["gold_denominator"]):
        reasons.append("answer_mismatch")
    return reasons


def adjudicable_pair(candidate, blind, ref, *, source_hint_agreement_required=True):
    """Structural gate only; the third actual agent must establish the stated cause."""
    if (not all(blind[k] is True for k in ("statement_ok", "unique_answer", "correctness"))
            or not all(ref[k] is True for k in ("gold_matches", "solution_consistent"))
            or Fraction(**blind["answer"]) != Fraction(candidate["gold_numerator"], candidate["gold_denominator"])
            or blind["domain"] not in DOMAINS or blind["difficulty"] not in TEST_QUOTA):
        return False
    if not source_hint_agreement_required:
        return (all(blind[k] is True for k in ("domain_ok", "difficulty_ok", "unit_explicit"))
                and ref["approve"] is True)
    mismatch = False
    for field in ("domain", "difficulty"):
        different = blind[field] != candidate["provisional_" + field]
        if blind[field + "_ok"] is False and (not source_hint_agreement_required or not different):
            return False
        mismatch |= different
    return ref["approve"] is True or mismatch or blind["unit_explicit"] is False


def original_review_refs(reviews):
    return [{k: r[k] for k in ("reviewer_id", "blind_record_sha256", "reference_record_sha256")} for r in reviews]


def validate_adjudication(candidate, decision, record):
    if (not decision["adjudication_eligible"] or record.get("phase") != "adjudication"
            or record.get("id") != candidate["id"] or record.get("candidate_sha256") != candidate["candidate_sha256"]):
        raise ValueError("Adjudication cannot override invalid math, ambiguity, unsolved or source-gold vetoes")
    if record.get("original_review_refs") != original_review_refs(decision["reviews"]):
        raise ValueError("Adjudication must bind ALL original blind/reference review hashes")
    fields = {"id", "reviewer_id", "phase", "candidate_sha256", "original_review_refs", "source_math_validation", "answer",
              "answer_correct", "gold_matches", "solution_consistent", "statement_ok", "unique_answer", "correctness", "approve",
              "final_domain", "final_difficulty", "domain_reason", "difficulty_reason", "resolution_reason", "unit_resolution",
              "unit_reason", "query_quote", "dimensionless_query", "dimensionless_kind", "original_unit_concerns", "fixture_only"}
    if set(record) - fields:
        raise ValueError("Adjudication has no quota targets, new gold or undeclared fields")
    for field, minimum in (("source_math_validation", 120), ("resolution_reason", 80), ("unit_reason", 60),
                           ("domain_reason", 40), ("difficulty_reason", 40)):
        if not review_text(record.get(field), minimum):
            raise ValueError("Detailed source mathematics, classification and unit adjudication required: " + field)
    if record.get("final_domain") not in DOMAINS or record.get("final_difficulty") not in TEST_QUOTA:
        raise ValueError("Final adjudication labels must describe in-scope intrinsic mathematics")
    answer = record.get("answer", {})
    if (set(answer) != {"numerator", "denominator"} or type(answer.get("numerator")) is not int
            or type(answer.get("denominator")) is not int or answer["denominator"] <= 0):
        raise ValueError("Adjudicated answer must be an exact rational")
    for field in ("answer_correct", "gold_matches", "solution_consistent", "statement_ok", "unique_answer", "correctness", "approve", "dimensionless_query"):
        if type(record.get(field)) is not bool:
            raise ValueError("Missing adjudication verdict: " + field)
    unit_false = [r for r in decision["reviews"] if r["original_verdicts"]["blind"]["unit_explicit"] is False]
    concerns = [{"reviewer_id": r["reviewer_id"], "blind_record_sha256": r["blind_record_sha256"],
                 "unit_explicit": False, "original_unit_reason": r["original_unit_reason"]} for r in unit_false]
    if record.get("original_unit_concerns") != concerns:
        raise ValueError("Adjudication must retain original false unit flags and reasons")
    if record.get("unit_resolution") not in ("unchanged", "inherently_dimensionless"):
        raise ValueError("Unknown adjudication unit resolution")
    if record.get("dimensionless_kind") not in ("not_dimensionless", "pure_number", "count", "probability", "ratio", "index", "coefficient"):
        raise ValueError("Adjudicator must positively identify the dimensionless query kind, not infer it from missing metadata")
    query = record.get("query_quote")
    if not isinstance(query, str) or len(query.strip()) < 5 or query not in candidate["problem"]:
        raise ValueError("Adjudication must quote the explicitly identified unique original query")
    accepts = all(record[k] is True for k in ("answer_correct", "gold_matches", "solution_consistent", "statement_ok",
                                             "unique_answer", "correctness", "approve"))
    if record["approve"] and not accepts:
        raise ValueError("Adjudication approval cannot override a false mathematical verdict")
    if accepts:
        if Fraction(**answer) != Fraction(candidate["gold_numerator"], candidate["gold_denominator"]):
            raise ValueError("Adjudication answer must match ORIGINAL source gold; no new gold")
        requested_unit = re.search(r"how many\s+(?:dollars?|cents?|euros?|percent)\b|"
                                   r"\bpercent(?:age)?\s+(?:increase|decrease)|what\s+(?:is\s+the\s+)?percent(?:age)?\b|"
                                   r"\bin\s+(?:dollars?|cents?|euros?|miles?|kilometers?|met(?:er|re)s?|centimeters?|"
                                   r"feet|inches|seconds?|minutes?|hours?|days?|years?|lit(?:er|re)s?|gallons?|"
                                   r"grams?|kilograms?|pounds?|ounces?|degrees?)\b", query, re.I)
        if unit_false and (record["unit_resolution"] != "inherently_dimensionless" or record["dimensionless_query"] is not True
                           or record["dimensionless_kind"] == "not_dimensionless" or requested_unit
                           or candidate.get("answer_unit") not in (None, "dimensionless")):
            raise ValueError("False unit verdicts are resolvable ONLY for inherently dimensionless queries")
    return accepts


def load_adjudications(rows, stage, decisions, runs, reviewer_ids):
    directory = stage / "reviews"
    registry_path = directory / "adjudications/adjudicators.json"
    if not registry_path.exists():
        if any((directory / "adjudications").glob("*.jsonl")):
            raise ValueError("Unregistered adjudication records")
        return {}
    candidates = {r["id"]: r for r in rows}
    hashes = {relative(registry_path): digest(registry_path)}
    seen_ids, consumed_ids = set(), set()
    for registration in read_json(registry_path).get("adjudicators", []):
        rid, run = registration.get("reviewer_id"), registration.get("agent_run_id")
        if (not isinstance(rid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{3,100}", rid) or rid in reviewer_ids
                or not isinstance(run, str) or len(run.strip()) < 6 or run != run.strip() or run in runs
                or registration.get("kind") != "agent" or registration.get("independent_adjudication") is not True):
            raise ValueError("Third adjudicator requires separate actual reviewer and distinct execution IDs")
        runs.add(run)
        reviewer_ids.add(rid)
        emitted = {}
        for entry in registration.get("evidence", []):
            path = directory / entry["path"]
            if (path.is_symlink() or not path.resolve().is_relative_to((directory / "evidence").resolve())
                    or digest(path) != entry["sha256"]):
                raise ValueError("Adjudicator emitted JSONL evidence missing/changed")
            text = path.read_text(encoding="utf-8")
            if any(marker in text.casefold() for marker in ("synthetic fixture", "dummy review", "fabricated review")):
                raise ValueError("Fixture/dummy evidence cannot authorize real adjudication")
            hashes[relative(path)] = digest(path)
            for record in read_jsonl(path):
                if (not isinstance(record, dict) or record.get("reviewer_id") != rid or record.get("phase") != "adjudication"
                        or record.get("id") not in candidates or record["id"] in seen_ids):
                    raise ValueError("Unknown/duplicate emitted adjudication; never cherry-pick")
                seen_ids.add(record["id"])
                emitted[record["id"]] = record
        output = directory / "adjudications" / (rid + ".jsonl")
        if not emitted or output.is_symlink():
            raise ValueError("Actual emitted adjudication evidence required")
        hashes[relative(output)] = digest(output)
        for record in read_jsonl(output):
            pid = record.get("id")
            if pid not in emitted or pid in consumed_ids or review_core(record) != review_core(emitted[pid]):
                raise ValueError("Adjudication differs from preserved emitted evidence")
            consumed_ids.add(pid)
            decision = decisions.get(pid, {"adjudication_eligible": False})
            accepts = validate_adjudication(candidates[pid], decision, record)
            decision.update(status="resolved", accepted=accepts, accepted_via_adjudication=accepts,
                            decision_type="adjudication",
                            final_domain=record["final_domain"] if accepts else None,
                            final_difficulty=record["final_difficulty"] if accepts else None,
                            adjudication={"reviewer_id": rid, "agent_run_id": run,
                                          "record_sha256": digest_bytes(canonical(record).encode("utf-8")), "record": record})
        if not set(emitted) <= consumed_ids:
            raise ValueError("Omitted emitted adjudication; retain every rejection")
    expected_outputs = {r["reviewer_id"] + ".jsonl" for r in read_json(registry_path).get("adjudicators", [])}
    if {p.name for p in (directory / "adjudications").glob("*.jsonl")} != expected_outputs:
        raise ValueError("Unregistered adjudication records")
    return hashes


def review_policy_bindings(stage):
    """Registration pins, not new raw verdict fields, distinguish historical reviews."""
    bindings, historical_registrations, seen = {}, set(), set()
    while (stage / "review_policy.json").exists():
        if stage.resolve() in seen:
            raise ValueError("Cyclic review policy lineage")
        seen.add(stage.resolve())
        policy = read_json(stage / "review_policy.json")
        historical = policy.get("quota_axis") != QUOTA_AXIS
        bindings[(policy["policy_id"], policy["contract_sha256"])] = historical
        if historical:
            for reviewer in read_json(stage / "reviews/reviewers.json")["reviewers"]:
                if "review_policy" not in reviewer:
                    historical_registrations.add(canonical(reviewer))
        metadata = read_json(stage / "preparation_manifest.json")
        lineage = metadata.get("previous_preparation")
        if not lineage:
            break
        archive = ROOT / lineage["archive"]
        if not archive.resolve().is_relative_to((ROOT / "data/v3/policy_amendments").resolve()):
            raise ValueError("Escaped review policy lineage")
        stage = archive / "data/v3/staging"
    return bindings, historical_registrations


def load_reviews(rows, stage, packets, *, include_adjudications=True):
    directory = stage / "reviews"
    registry_path = directory / "reviewers.json"
    registry = read_json(registry_path).get("reviewers", [])
    if not registry:
        raise ValueError("Missing actual reviewer registry; no final files written")
    candidates = {r["id"]: r for r in rows}
    packet_by_id = {pid: p for p in packets for pid in p["candidate_ids"]}
    reviewers, runs, evidence_hashes = {}, set(), {relative(registry_path): digest(registry_path)}
    policy_bindings, historical_registrations = review_policy_bindings(stage)
    for reviewer in registry:
        rid, run = reviewer.get("reviewer_id"), reviewer.get("agent_run_id")
        if (not isinstance(rid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{3,100}", rid)
                or rid in reviewers or not isinstance(run, str) or len(run.strip()) < 6 or run != run.strip() or run in runs
                or reviewer.get("kind") != "agent" or reviewer.get("independent_blind_review") is not True):
            raise ValueError("Require unique actual agent reviewer IDs and distinct execution IDs")
        evidence = reviewer.get("evidence", [])
        binding = reviewer.get("review_policy")
        if binding is not None and (not isinstance(binding, dict) or set(binding) != {"id", "sha256"}
                                   or (binding["id"], binding["sha256"]) not in policy_bindings):
            raise ValueError("Reviewer registration must bind an actual declared current or archived policy")
        if not evidence:
            raise ValueError("Actual agent execution evidence is required; self-attested flags alone are insufficient")
        emitted_records = set()
        emitted_keys = set()
        emitted_objects = {}
        for entry in evidence:
            path = (directory / entry["path"]).resolve()
            if not path.is_relative_to(directory.resolve()) or path.is_symlink() or digest(path) != entry["sha256"]:
                raise ValueError("Reviewer evidence missing, escaped reviews directory, or changed")
            text = path.read_text(encoding="utf-8")
            if any(marker in text.casefold() for marker in ("synthetic fixture", "dummy review", "fabricated review")):
                raise ValueError("Fixture/dummy evidence cannot authorize real publication")
            # Only hashes attached by the parent after the blind solve may differ
            # from the complete preserved machine-readable agent judgment.
            for output in read_jsonl(path):
                if not isinstance(output, dict):
                    raise ValueError("Agent evidence must contain actual JSONL judgment objects")
                if (output.get("reviewer_id") != rid or output.get("id") not in candidates
                        or output.get("phase") not in ("blind", "reference")):
                    raise ValueError("Evidence contains unknown candidate/reviewer/phase")
                key = (output["id"], output["phase"])
                if key in emitted_keys:
                    raise ValueError("Duplicate emitted judgments; never cherry-pick agent output")
                emitted_keys.add(key)
                emitted_records.add(canonical(review_core(output)))
                emitted_objects[key] = output
            evidence_hashes[relative(path)] = entry["sha256"]
        for (pid, phase), output in emitted_objects.items():
            raw_blind = emitted_objects.get((pid, "blind"))
            if phase == "reference" and raw_blind is not None and output.get("blind_record_sha256") is not None:
                completed = {**raw_blind, "candidate_sha256": candidates[pid]["candidate_sha256"],
                             "packet_sha256": packet_by_id[pid]["blind_sha256"]}
                if output["blind_record_sha256"] not in (digest_bytes(canonical(raw_blind).encode("utf-8")),
                                                         digest_bytes(canonical(completed).encode("utf-8"))):
                    raise ValueError("Emitted reference hash does not bind its preserved blind solution")
        reviewers[rid] = (reviewer, emitted_records)
        runs.add(run)
    records = {}
    consumed = defaultdict(set)
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == SCOPE_LEDGER:
            continue
        evidence_hashes[relative(path)] = digest(path)
        for record in read_jsonl(path):
            rid, pid, phase = record.get("reviewer_id"), record.get("id"), record.get("phase")
            if rid not in reviewers or pid not in candidates or phase not in ("blind", "reference"):
                raise ValueError(f"Unknown reviewer/candidate/phase in {path}")
            key = (pid, rid, phase)
            if key in records:
                raise ValueError("Duplicate review records; refusing to cherry-pick a verdict")
            expected = packet_by_id[pid]["blind_sha256" if phase == "blind" else "reference_sha256"]
            if record.get("packet_sha256") != expected:
                raise ValueError("Review does not bind its blind/reference packet")
            evidence = reviewers[rid][1]
            if canonical(review_core(record)) not in evidence:
                raise ValueError("Complete review judgments differ from preserved actual agent output")
            records[key] = record
            consumed[rid].add(canonical(review_core(record)))
    for rid, (_, emitted) in reviewers.items():
        if consumed[rid] != emitted:
            raise ValueError("Omitted emitted agent judgments; every rejection/incomplete pair must be retained")
    for registration, _ in reviewers.values():
        if registration.get("review_policy") is None and canonical(registration) not in historical_registrations:
            raise ValueError("Fresh reviewer requires explicit policy binding; no unarchived historical exceptions")
    decisions = {}
    for row in rows:
        votes, pairs = [], []
        for rid in sorted(reviewers):
            blind, ref = (records.get((row["id"], rid, p)) for p in ("blind", "reference"))
            if blind is None and ref is None:
                continue
            if blind is None or ref is None:
                raise ValueError("Incomplete blind/reference review pair")
            binding = reviewers[rid][0].get("review_policy")
            historical = True if binding is None else policy_bindings[(binding["id"], binding["sha256"])]
            ok = validate_review(row, blind, ref, source_hint_agreement_required=historical)
            pairs.append((blind, ref, historical))
            votes.append({"reviewer_id": rid, "agent_run_id": reviewers[rid][0]["agent_run_id"], "approved": ok,
                          "blind_record_sha256": digest_bytes(canonical(blind).encode("utf-8")),
                          "reference_record_sha256": digest_bytes(canonical(ref).encode("utf-8")),
                          "reasons": review_reasons(row, blind, ref),
                          "original_verdicts": {"blind": {k: blind[k] for k in ("statement_ok", "unique_answer", "correctness", "domain_ok", "difficulty_ok", "unit_explicit")},
                                                "reference": {k: ref[k] for k in ("gold_matches", "solution_consistent", "approve")}},
                          "original_unit_reason": (blind["independent_solution"] + "\n" + ref["reference_comparison"])
                                                  if blind["unit_explicit"] is False else None})
        if not votes:
            continue
        labels = {(blind["domain"], blind["difficulty"]) for blind, _, _ in pairs}
        ordinary = len(votes) >= 2 and all(v["approved"] for v in votes) and len(labels) == 1
        eligible = (len(votes) >= 2 and not ordinary and
                    all(adjudicable_pair(row, blind, ref, source_hint_agreement_required=historical)
                        for blind, ref, historical in pairs))
        domain, difficulty = pairs[0][0]["domain"], pairs[0][0]["difficulty"]
        contrast = (domain, difficulty) != (row["provisional_domain"], row["provisional_difficulty"])
        decisions[row["id"]] = {"status": "approved" if ordinary else "awaiting_adjudication" if eligible else
                               "rejected" if any(not v["approved"] for v in votes) else "awaiting_reviews",
                               "accepted": ordinary, "adjudication_eligible": eligible, "reviews": votes,
                               "final_domain": domain if ordinary else None,
                               "final_difficulty": difficulty if ordinary else None,
                               "sampling_tier": row["provisional_difficulty"],
                               "decision_type": "source_hint_contrast" if ordinary and contrast else "ordinary" if ordinary else None,
                               "accepted_via_adjudication": False, "adjudication": None}
    if include_adjudications:
        evidence_hashes.update(load_adjudications(rows, stage, decisions, runs, set(reviewers)))
    evidence_hashes.update(apply_review_scope_incidents(stage, decisions))
    approved = {pid for pid, decision in decisions.items() if decision["accepted"]}
    return approved, decisions, evidence_hashes


def apply_review_scope_incidents(stage, decisions, *, incidents=None):
    """Administrative eligibility layer; never mutate original mathematical votes."""
    directory = stage / "reviews"
    ledger = directory / SCOPE_LEDGER
    data = ledger.read_bytes() if ledger.exists() else b""
    current, visited = stage, set()
    while (current / "preparation_manifest.json").exists():
        lineage = read_json(current / "preparation_manifest.json").get("previous_preparation")
        if not lineage:
            break
        current = ROOT / lineage["archive"] / "data/v3/staging"
        if not current.resolve().is_relative_to((ROOT / "data/v3/policy_amendments").resolve()) or current.resolve() in visited:
            raise ValueError("Invalid scope incident ledger lineage")
        visited.add(current.resolve())
        prior = current / "reviews" / SCOPE_LEDGER
        if prior.exists() and (not ledger.exists() or not data.startswith(prior.read_bytes())):
            raise ValueError("Missing anchored or truncated append-only scope incident ledger")
    receipt_paths = sorted((directory / "scope_incident_receipts").glob("*.json"))
    receipts = []
    for path in receipt_paths:
        receipt = read_json(path)
        size = receipt["committed_prefix_size"]
        if (path.is_symlink() or type(size) is not int or size < 1 or len(data) < size
                or digest_bytes(data[:size]) != receipt["committed_prefix_sha256"]):
            raise ValueError("Committed scope incident ledger prefix was removed or changed")
        receipts.append(receipt["incident"])
    if incidents is None and not ledger.exists():
        manifest = stage / "preparation_manifest.json"
        lineage = read_json(manifest).get("previous_preparation") if manifest.exists() else None
        if lineage and (ROOT / lineage["archive"] / "data/v3/staging/reviews" / SCOPE_LEDGER).exists():
            raise ValueError("Missing anchored append-only scope incident ledger")
        return {}
    hashes = {relative(ledger): digest(ledger)} if ledger.exists() else {}
    hashes.update({relative(p): digest(p) for p in receipt_paths})
    incidents = read_jsonl(ledger) if incidents is None else incidents
    if any(receipt not in incidents for receipt in receipts):
        raise ValueError("Committed scope incident was omitted from its append-only ledger")
    registrations = read_json(directory / "reviewers.json")["reviewers"]
    adj_registry = directory / "adjudications/adjudicators.json"
    if adj_registry.exists():
        registrations += read_json(adj_registry)["adjudicators"]
    seen = set()
    for incident in incidents:
        fields = {"schema_version", "incident_id", "rule_id", "reason", "reviewer_id", "agent_run_id", "candidate_ids", "evidence", "assigned_packet"}
        if (not isinstance(incident, dict) or set(incident) != fields or incident["schema_version"] != 1
                or incident["rule_id"] != SCOPE_RULE_ID or incident["reason"] != "review_scope_breach"):
            raise ValueError("Malformed scope incident or undeclared exclusion rule")
        core = {k: v for k, v in incident.items() if k != "incident_id"}
        expected_id = "jev-v3-scope-" + digest_bytes(canonical(core).encode("utf-8"))[:16]
        if incident["incident_id"] != expected_id or expected_id in seen:
            raise ValueError("Duplicate/changed scope incident binding")
        seen.add(expected_id)
        matches = [r for r in registrations if r["reviewer_id"] == incident["reviewer_id"] and r["agent_run_id"] == incident["agent_run_id"]]
        if len(matches) != 1:
            raise ValueError("Scope incident must bind an exact actual registered reviewer/execution")
        registration = matches[0]
        ids = incident["candidate_ids"]
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or not set(ids) <= set(decisions):
            raise ValueError("Unknown/duplicate scope-affected candidate IDs")
        assigned = {e["id"] for entry in registration["evidence"] for e in read_jsonl(directory / entry["path"])}
        if set(ids) != assigned:
            raise ValueError("Scope breach affects ALL assigned execution candidates; never select by verdict")
        for key in ("evidence", "assigned_packet"):
            binding = incident[key]
            if set(binding) != ({"path", "sha256", "self_report_marker"} if key == "evidence" else {"path", "sha256"}):
                raise ValueError("Scope incident requires exact evidence and assigned-packet bindings")
            path = directory / binding["path"]
            base = directory / "evidence" if key == "evidence" else stage
            if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                    or not path.resolve().is_relative_to(base.resolve()) or digest(path) != binding["sha256"]):
                raise ValueError("Scope incident evidence/assignment missing, escaped or changed")
            hashes[relative(path)] = binding["sha256"]
        report = read_json(directory / incident["evidence"]["path"])
        if (report.get("agent_run_id") != incident["agent_run_id"] or report.get("reviewer_id") != incident["reviewer_id"]
                or report.get("strict_assigned_input_boundary_satisfied") is not False
                or report.get("agent_report") != incident["evidence"]["self_report_marker"]
                or not review_text(report.get("agent_report"), 40) or set(report.get("candidate_ids", [])) != assigned
                or not any(e["path"] == report.get("original_evidence_path") and e["sha256"] == report.get("original_evidence_sha256")
                           for e in registration["evidence"])):
            raise ValueError("Scope incident lacks the exact actual self-report and original execution evidence")
        packet_ids = {e["candidate"]["id"] if "candidate" in e else e["id"]
                      for e in read_jsonl(directory / incident["assigned_packet"]["path"])}
        if packet_ids != assigned:
            raise ValueError("Scope incident assigned packet does not match its complete affected execution")
        packet_binding = incident["assigned_packet"]
        if registration.get("independent_adjudication") is True:
            mirror = directory / "adjudication_registry.json"
            dispatches = read_json(mirror).get("adjudicators", []) if mirror.exists() else []
            dispatch = next((r for r in dispatches if r["reviewer_id"] == incident["reviewer_id"]
                             and r["agent_run_id"] == incident["agent_run_id"]), None)
            path = dispatch.get("packet") if dispatch else report.get("assigned_packet_path")
            sha = dispatch.get("packet_sha256") if dispatch else report.get("assigned_packet_sha256")
            if path != packet_binding["path"] or sha != packet_binding["sha256"]:
                raise ValueError("Scope assignment needs independently preserved actual dispatch binding")
        else:
            packets = read_json(stage / "preparation_manifest.json")["packets"]
            actual_path = (directory / packet_binding["path"]).resolve()
            if not any(set(p["candidate_ids"]) == assigned and actual_path == (ROOT / p["blind"]).resolve()
                       and packet_binding["sha256"] == p["blind_sha256"] for p in packets):
                raise ValueError("Scope assignment must bind the original dispatched blind packet")
        for pid in ids:
            decision = decisions[pid]
            if not decision.get("scope_disqualified"):
                decision["pre_scope_status"] = decision["status"]
                decision["pre_scope_accepted"] = decision["accepted"]
            decision.update(status="scope_disqualified", accepted=False, scope_disqualified=True)
            if "review_scope_breach" not in decision.get("exclusion_reasons", []):
                decision.setdefault("exclusion_reasons", []).append("review_scope_breach")
            decision.setdefault("review_scope_incidents", []).append(incident)
    return hashes


def review_core(record):
    return {k: v for k, v in record.items()
            if k not in ("candidate_sha256", "packet_sha256", "blind_record_sha256")}


def reviewed_capacity(rows, approved, decisions, *, axis="intrinsic_difficulty"):
    if axis not in ("intrinsic_difficulty", QUOTA_AXIS):
        raise ValueError("Unknown reviewed capacity axis")
    return {domain: {diff: sum(r["id"] in approved and decisions[r["id"]]["final_domain"] == domain
                              and (r["provisional_difficulty"] if axis == QUOTA_AXIS else decisions[r["id"]]["final_difficulty"]) == diff
                              for r in rows)
                     for diff in TEST_QUOTA} for domain in DOMAINS}


def select_approved(rows, approved, decisions):
    queues = defaultdict(deque)
    for row in rows:
        if row["id"] in approved:
            decision = decisions[row["id"]]
            if decision.get("scope_disqualified") or not decision["accepted"] or decision["final_domain"] not in DOMAINS or decision["final_difficulty"] not in TEST_QUOTA:
                raise ValueError("Selection requires final reviewed labels, never provisional fallbacks")
            if row["provisional_difficulty"] not in TEST_QUOTA:
                raise ValueError("Sampling tier must be the immutable original source proxy")
            queues[(decision["final_domain"], row["provisional_difficulty"])].append(row)
    selected = {"test": [], "pilot": []}
    shortfalls = []
    for split, quota in (("test", TEST_QUOTA), ("pilot", PILOT_QUOTA)):
        for domain in DOMAINS:
            for diff, count in quota.items():
                queue = queues[(domain, diff)]
                take = min(count, len(queue))
                selected[split].extend(queue.popleft() for _ in range(take))
                if take < count:
                    shortfalls.append({"split": split, "domain": domain, "sampling_tier": diff,
                                       "required": count, "approved_available": take})
    if shortfalls:
        raise ValueError("Insufficient twice-reviewed strata; no final files written: " + canonical(shortfalls))
    if any(d.get("status") == "awaiting_adjudication" for d in decisions.values()):
        raise ValueError("Resolve ALL eligible classification/unit adjudications uniformly before test freeze")
    if len({r["source_family"] for r in selected["test"] + selected["pilot"]}) < 2:
        raise ValueError("Publication requires multiple genuine source families")
    return selected


def latency_subset(test):
    rng, selected = random.Random(LATENCY_SEED), []
    for domain in DOMAINS:
        for diff, count in LATENCY_QUOTA.items():
            ids = sorted(r["id"] for r in test if r["domain"] == domain and r["sampling_tier"] == diff)
            selected.extend(rng.sample(ids, count))
    rng.shuffle(selected)
    return selected


def schedule_for(test, pilot):
    rng = random.Random(SEED)
    result = {"planning_seed": SEED}
    for rows, order_key, blocks_key, block_count in ((test, "order", "blocks", 50),
                                                     (pilot, "pilot_order", "pilot_blocks", 5)):
        blocks = [[] for _ in range(block_count)]
        for domain in DOMAINS:
            ids = sorted(r["id"] for r in rows if r["domain"] == domain)
            rng.shuffle(ids)
            for number, pid in enumerate(ids):
                blocks[number % block_count].append(pid)
        for block in blocks:
            rng.shuffle(block)
        rng.shuffle(blocks)
        result[blocks_key] = blocks
        result[order_key] = [pid for block in blocks for pid in block]
    return result


def latency_plan_for(test_gold, latency_ids, schedule):
    by_id = {r["id"]: r for r in test_gold}
    if len(latency_ids) != 100 or len(set(latency_ids)) != 100 or not set(latency_ids) <= set(by_id):
        raise ValueError("Latency plan must use 100 actual unique test IDs")
    order = [pid for pid in schedule["order"] if pid in set(latency_ids)]
    if len(order) != 100 or set(order) != set(latency_ids):
        raise ValueError("Latency metadata must preserve every selected unique test ID exactly once")
    items = [{"item_id": pid, "problem": by_id[pid]["problem"],
              "domain": by_id[pid]["domain"], "difficulty": by_id[pid]["difficulty"],
              "sampling_tier": by_id[pid]["sampling_tier"]} for pid in order]
    if any(r["difficulty"] not in TEST_QUOTA for r in items):
        raise ValueError("Latency metadata requires actual resolved intrinsic difficulty labels")
    for domain in DOMAINS:
        if Counter(r["sampling_tier"] for r in items if r["domain"] == domain) != LATENCY_QUOTA:
            raise ValueError("Latency plan must have source-tier6/8/6 strata in each genuine reviewed domain")
    return {"dataset_version": "v3", "sealed": True, "selection_seed": LATENCY_SEED,
            "source_split": "test", "split": "test", "items": items,
            "blocks": [order[i:i+10] for i in range(0, 100, 10)], "seeds": EXPERIMENT_SEEDS,
            "repetitions": 3, "conditions": ["JFINAL", "B13_GREEDY"], "generation": 0,
            "expected_rows": 1800, "quota_per_domain": LATENCY_QUOTA, "quota_axis": QUOTA_AXIS,
            "intrinsic_difficulty_counts": {f: sum(r["difficulty"] == f for r in items) for f in TEST_QUOTA}}


def freeze():
    if (OUT / "SHA256SUMS").exists() or any((OUT / f"{s}_{k}.jsonl").exists()
                                          for s in ("test", "pilot", "dev", "latency") for k in ("inputs", "gold")):
        raise ValueError("Final publication exists; refusing to overwrite a freeze or partial publication")
    stage = OUT / "staging"
    hashes = verify_seal(stage)
    metadata = read_json(stage / "preparation_manifest.json")
    protected = verify_preparation_lineage(stage, metadata)
    policy = read_json(stage / "review_policy.json")
    counter, tokenizers = tokenizer_counter()
    if tokenizers != metadata["tokenizers"] or digest(ROOT / "prompts" / "generador.txt") != metadata["generator_prompt_sha256"]:
        raise ValueError("Pinned tokenizer/prompt metadata changed")
    rows = read_jsonl(stage / "candidate_pool.jsonl")
    for row in rows:
        if row["candidate_sha256"] != candidate_sha(row) or row["raw_problem"] != row["problem"]:
            raise ValueError("Candidate content hash/raw statement changed")
    packet_rows = []
    for packet in metadata["packets"]:
        blind_path, ref_path = ROOT / packet["blind"], ROOT / packet["reference"]
        if (not blind_path.resolve().is_relative_to((stage / "packets").resolve())
                or not ref_path.resolve().is_relative_to((stage / "packets").resolve())
                or digest(blind_path) != packet["blind_sha256"] or digest(ref_path) != packet["reference_sha256"]):
            raise ValueError("Packet binding differs from actual staged artifact")
        blind, reference = read_jsonl(blind_path), read_jsonl(ref_path)
        if (not 1 <= len(reference) <= 25 or [r["id"] for r in reference] != packet["candidate_ids"]
                or blind != [{k: r[k] for k in ("id", "problem", "language")} for r in reference]):
            raise ValueError("Packet is not a genuine blind/reference candidate pair")
        packet_rows.extend(reference)
    if packet_rows != rows:
        raise ValueError("Packet reference rows differ from the real candidate pool")
    approved, decisions, review_hashes = load_reviews(rows, stage, metadata["packets"])
    selected = select_approved(rows, approved, decisions)
    # A valid-looking seal/JSON must not authorize invented candidates.
    used, extra, original_dev, _ = load_exclusions()
    pool, _, _ = load_sources()
    replayed, _ = discover(pool, used, extra, counter)
    replayed, _, _ = review_priority(replayed)
    if replayed != rows:
        raise ValueError("Staged candidate pool cannot be replayed from the real pinned sources")
    if select_approved(replayed, approved, decisions) != selected:
        raise ValueError("Pinned source replay differs from final reviewed selection")
    inputs, gold = {}, {}
    rng = random.Random(SEED)
    for split in ("test", "pilot"):
        rng.shuffle(selected[split])
        inputs[split], gold[split] = [], []
        for number, row in enumerate(selected[split], 1):
            pid = f"jev-v3-{split[0]}-{number:03d}"
            if counter(row["problem"]) != row["prompt_tokens"]:
                raise ValueError("Rendered prompt counts changed")
            inputs[split].append({"id": pid, "problem": row["problem"], "language": "en"})
            gold[split].append({**row, "candidate_id": row["id"], "id": pid,
                "domain": decisions[row["id"]]["final_domain"], "difficulty": decisions[row["id"]]["final_difficulty"],
                "sampling_tier": row["provisional_difficulty"],
                "template_group": row["source_id"], "reference_solution": row["solution"],
                "agent_reviews": decisions[row["id"]]["reviews"], "review_decision": decisions[row["id"]],
                "accepted_via_adjudication": decisions[row["id"]]["accepted_via_adjudication"], "agent_reviewed": True,
                "human_reviewed": False, "reviewed": False})
    inputs["dev"] = read_jsonl(stage / "dev_reuse_inputs.jsonl")
    dev_gold = read_jsonl(stage / "dev_reuse_gold.jsonl")
    if inputs["dev"] != original_dev[0] or dev_gold != original_dev[1]:
        raise ValueError("Staged dev differs from the original 20 aligned development records")
    gold["dev"] = [dict(g, prompt_tokens=counter(i["problem"]), agent_reviewed=False,
                        human_reviewed=False, reviewed=False, reuse_only=True)
                   for i, g in zip(inputs["dev"], dev_gold)]
    latency_ids = latency_subset(gold["test"])
    by_id = {r["id"]: r for r in inputs["test"]}
    inputs["latency"] = [by_id[pid] for pid in latency_ids]
    schedule = schedule_for(gold["test"], gold["pilot"])
    latency_plan = latency_plan_for(gold["test"], latency_ids, schedule)
    verify_hashes(review_hashes)
    verify_hashes(protected)
    for split, records in inputs.items():
        write_jsonl(OUT / f"{split}_inputs.jsonl", records)
        if split in gold:
            write_jsonl(OUT / f"{split}_gold.jsonl", gold[split])
    write_json(OUT / "latency_ids.json", {"seed": LATENCY_SEED, "ids": latency_ids, "source_split": "test",
                                         "quota_per_domain": LATENCY_QUOTA, "quota_axis": QUOTA_AXIS})
    write_json(OUT / "latency_plan.json", latency_plan)
    write_json(OUT / "schedule.json", schedule)
    write_json(OUT / "review_policy.json", policy)
    write_json(OUT / "review_manifest.json", {**metadata, "dataset_version": "v3", "status": "FROZEN_ANALYSIS_ONLY",
        "actual_n": {s: len(r) for s, r in inputs.items()}, "review_evidence_sha256": review_hashes,
        "review": {"agent_reviewed": True, "agent_reviewed_all": True, "human_reviewed": False, "reviewed": False,
                   "policy_id": policy["policy_id"], "policy_sha256": policy["contract_sha256"],
                   "required_independent_actual_agents": 2, "dev_agent_reviewed": False,
                   "provenance_boundary": "Execution identity/evidence must be supplied by the delegating parent; JSON cannot authenticate an agent by itself."},
        "selected_twice_approved_candidates": 550,
        "selected_independent_review_pairs": sum(len(decisions[r["id"]]["reviews"]) for rr in selected.values() for r in rr),
        "review_decisions": decisions,
        "accepted_reviewed_strata": reviewed_capacity(rows, approved, decisions),
        "accepted_sampling_strata": reviewed_capacity(rows, approved, decisions, axis=QUOTA_AXIS),
        "selected_candidate_assignments": {split: [{"id": g["id"], "candidate_id": g["candidate_id"],
            "candidate_sha256": g["candidate_sha256"], "source_id": g["source_id"], "agent_reviews": g["agent_reviews"]}
            for g in records] for split, records in gold.items() if split != "dev"},
        "test_reserved_before_pilot": True, "no_quota_fallback_or_relabeling": True,
        "latency_test_subset_only": True,
        "strata": {s: reviewed_capacity(r, approved, decisions, axis=QUOTA_AXIS) for s, r in selected.items()},
        "intrinsic_reviewed_strata": {s: reviewed_capacity(r, approved, decisions) for s, r in selected.items()}})
    input_hashes = {f"{split}_inputs.jsonl": digest(OUT / f"{split}_inputs.jsonl") for split in inputs}
    input_hashes.update({name: digest(OUT / name) for name in ("schedule.json", "latency_plan.json")})
    write_json(OUT / "dataset_manifest.json", public_manifest(metadata, input_hashes, policy,
        digest(OUT / "review_manifest.json"), complete=True,
        intrinsic_counts={s: {f: sum(g["difficulty"] == f for g in gold[s]) for f in TEST_QUOTA} for s in ("test", "pilot")}))
    paths = [p for p in OUT.rglob("*") if p.is_file() and p != OUT / "SHA256SUMS"]
    required = {**protected, **metadata["source_sha256"], **review_hashes,
                **{relative(stage / name): expected for name, expected in hashes.items()},
                relative(Path(__file__)): metadata["builder_sha256"],
                relative(OUT / "REVIEW_CONTRACT.md"): policy["contract_sha256"],
                relative(OUT / "ELIGIBILITY_POLICY.md"): policy["eligibility_document_sha256"],
                metadata["source_lock_path"]: metadata["source_lock_sha256"]}
    for meta in metadata["tokenizers"].values():
        required.update({meta["directory"] + "/" + name: expected for name, expected in meta["sha256"].items()})
    seal(OUT, paths, required)
    verify_hashes(protected)
    print(json.dumps({"actual_n": {s: len(r) for s, r in inputs.items()}, "status": "FROZEN"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--amend-review-policy-before-inference", metavar="REASON",
                       help="Explicit user-authorized policy-only amendment, preserving original staging and actual vetoes")
    modes.add_argument("--repair-review-scope-before-inference", metavar="REASON",
                       help="Explicit technical code/policy attestation for existing assigned-input isolation; contract and quotas stay pinned")
    parser.add_argument("--previous-preparation-archive", type=Path,
                        help="Sealed original builder/contracts/staging snapshot made BEFORE editing policy inputs")
    modes.add_argument("--acquire-sources", action="store_true",
                       help="Acquire pinned source metadata/corpora only; never modify reviewed staging")
    modes.add_argument("--audit-capacity", action="store_true",
                       help="Audit universal pre-review rules against real pinned sources without changing staging")
    parser.add_argument("--repair-staging-before-review", metavar="REASON",
                        help="With --prepare only: preserve entirely unreviewed staging before a documented technical fix; seeds stay fixed")
    args = parser.parse_args()
    try:
        if args.repair_staging_before_review and not args.prepare:
            raise ValueError("Staging repair is only allowed with --prepare")
        if args.amend_review_policy_before_inference or args.repair_review_scope_before_inference:
            if not args.previous_preparation_archive:
                raise ValueError("Explicit amendment requires --previous-preparation-archive")
            print(json.dumps(amend_review_policy_before_inference(args.amend_review_policy_before_inference or args.repair_review_scope_before_inference,
                args.previous_preparation_archive, technical_scope_repair=bool(args.repair_review_scope_before_inference)), indent=2))
        elif args.previous_preparation_archive:
            raise ValueError("Preparation archive is only used for explicit policy amendment")
        elif args.acquire_sources:
            pool, sources, _ = load_sources()
            print(json.dumps({"existing_and_asdiv_records": len(pool), "sources": sources}, indent=2))
        elif args.audit_capacity:
            used, extra, _, _ = load_exclusions()
            pool, _, _ = load_sources()
            counter, _ = tokenizer_counter()
            rows, audit = discover(pool, used, extra, counter)
            destination = OUT / "capacity_audits" / digest(Path(__file__))
            write_json(destination / "audit.json", audit)
            write_jsonl(destination / "candidate_pool.jsonl", rows)
            print(json.dumps(audit, indent=2))
        else:
            prepare(args.repair_staging_before_review) if args.prepare else freeze()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
