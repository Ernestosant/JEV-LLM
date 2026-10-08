# Dataset (Technical Summary)

> Historical context: this document describes the legacy 0.8B experiment, not the completed v3 study. See [v3 results](13_results_v3.md) and [limitations and future work](14_limitations_and_future_work.md).

Built by `data/build_dataset.py` (deterministic, seed `20260927`). Reproduce with: `python data/build_dataset.py`.

## Sources (Pinned Revisions)

| Source | Repo / File | Revision | Use |
|---|---|---|---|
| GSM8K | `openai/gsm8k` · `main/test` | `740312add88f781978c0658806c59bc2815b9866` | multistep arithmetic; ratios (keywords) |
| MATH | `EleutherAI/hendrycks_math` · test split for algebra, number_theory, counting_and_probability, prealgebra | `21a5633873b6a120296cce3e2df9d5550074f4a3` | algebra, number theory, counting/probability, ratios |
| AMC12 2022–23 | `AI-MO/aimo-validation-amc` · train (83) | `69d78a4a2c840e82d69af6bc742bda09005f6316` | hard stratum |
| AIME 2024–2025 | local copies of `math_agent` (`data/raw/aime/*.jsonl`) | SHA-256 in `dataset_manifest.json` | hard stratum |

## Design

* 5 domains: `arithmetic`, `algebra`, `ratios_percentages`, `number_theory`, `counting_probability`.
* **Test (100):** 8 easy + 8 medium + 4 hard per domain. **Dev (20):** 2 + 1 + 1 per domain. Disjoint.
* **Difficulty (heuristic, amendment A5):**
  * GSM8K: 2–3 `<<…>>` steps = easy; 4–5 = medium; ≥ 7 = hard (only as hard-arithmetic filler).
  * MATH: level 1–2 = easy; 3 = medium; 4–5 = hard (filler).
  * AMC/AIME: always hard. Domain assigned by keywords + 16 manual corrections based only on the problem statement
    (`DOMAIN_OVERRIDES` in the script); geometry, trigonometry, multiple-choice, and money problems are excluded.
* **Hard-stratum preference:** competition problems first; if insufficient, declared filler (arithmetic → GSM8K ≥ 7
  steps; ratios → MATH prealgebra L4–5). Actual outcome in `dataset_manifest.json → counts`:
  hard arithmetic = 1 AMC + 3 GSM8K; hard ratios = 1 AMC + 3 MATH; the rest 4/4 competition problems.
* **Filters:** exact answer (integer, `\frac{p}{q}`, finite decimal) converted to `Fraction`; |answer| ≤ 999;
  no money (`$`, dollars, cents), no "what percent", no `[asy]`, units/degrees, bases, "nearest", scientific
  notation, intervals, ordered pairs; no near-duplicates (8-grams); rendered prompt ≤ 512 tokens
  with the G and B tokenizers and templates (observed max: 292).
* Shuffled **opaque IDs** (`jev-t-001`, `jev-d-001`): do not reveal source, domain, or difficulty.
* **Answer types:** 111 integers, 9 fractions/decimals.

## Files

| File | Contents | In the Inference Bundle? |
|---|---|---|
| `test_inputs.jsonl`, `dev_inputs.jsonl` | `id`, `problem`, `language` | yes |
| `schedule.json` | 10 domain-balanced blocks of 10 (seed 20260927) and ordering | yes |
| `test_gold.jsonl`, `dev_gold.jsonl` | `gold_numerator/denominator`, `gold_answer`, `reference_solution`, `domain`, `difficulty`, `difficulty_rule`, `template_group`, `source*`, `prompt_tokens`, `reviewed=false` | **no** (analysis only) |
| `blind_review_ids.json` | 20 test IDs (4/domain) for the §11 blind review, selected before any output | no |
| `review_sheet.csv` | sheet for the second reviewer (uniqueness, problem statement, solution, difficulty) | no |
| `dataset_manifest.json` | sources, exclusions by reason, counts, types | yes |
| `SHA256SUMS` | hashes of all the above | — |

## Limitations

Likely contamination (public benchmarks); difficulty not validated by humans; "hard" GSM8K arithmetic
is easier than competition arithmetic; the money/percentage filter biases GSM8K toward problems without currency;
competition-domain classification uses keywords with manual corrections.
