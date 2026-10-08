# Legacy Minimal-Profile Results

These are results from the earlier experiment, not new Qwen4B v3 results.

Runs completed on 2026-09-29/30: 100 frozen problems, seed 17, six
conditions (600 runs). Models/prompts/budgets unchanged, code 0.1.2,
SHARED4-VLLM-GRAPH profile from amendment v3.1. This is not an R4 run with four physical
copies. Human review of the dataset and reasoning is still pending.

## Quality

Offline grading independent of inference: exactly one FINAL line,
strict numeric grammar and exact rational comparison. Truncations, EOS without
FINAL and round limits count as incorrect; none were excluded.

| Condition | Correct/100 | Accuracy CI95 | Valid format |
|---|---:|---|---:|
| G_SINGLE | 7 | [3, 12] % | 18 % |
| B13 | 73 | [64, 81] % | 94 % |
| B13_GREEDY | 81 | [73, 88] % | 91 % |
| J64 | 17 | [10, 24] % | 34 % |
| JSTEP | 22 | [14, 30] % | 34 % |
| JFINAL | 22 | [15, 30] % | 36 % |

Paired bootstrap by problem, stratified by domain, 10,000 resamples,
seed 271828. Accuracy differences vs B13, with adjusted CI99.1667:

- J64: -56 percentage points; interval [-69, -42].
- JSTEP: -51 percentage points; interval [-64, -37].
- JFINAL: -51 percentage points; interval [-64, -38].

These results do not support higher hybrid accuracy vs B13. The
point-estimate improvements vs G_SINGLE do not eliminate that gap. B13_GREEDY is a secondary
control; the best baseline was not selected per problem.

## Latency

Notebook 08: 32 inputs separate from the test (20 dev and 12 synthetic), six conditions,
192 measurements on a single A100, counterbalanced order and swaps outside the timer.
Latencies from 01-06, obtained on different VMs, remain descriptive.

| Condition | Median in 08 (s) | Ratio of medians B13/method |
|---|---:|---:|
| B13 | 2.502 | 1.00 |
| B13_GREEDY | 2.491 | 1.00 |
| G_SINGLE | 0.744 | 3.36 |
| J64 | 0.980 | 2.55 |
| JSTEP | 2.907 | 0.86 |
| JFINAL | 3.507 | 0.71 |

J64 ratio 2.55: CI95 [1.49, 5.70], CI99.1667 [1.26, 7.85]. JSTEP and JFINAL have
intervals that cross 1. These times include invalid terminations: in 08 B13
ended with status final on 32/32 inputs, J64 on 22/32, JSTEP on 19/32 and JFINAL on
19/32. Do not interpret fast failure termination as greater capability or infer
a joint quality/latency advantage on the test using a different distribution of
questions. No hybrid here demonstrates being both more accurate and faster than B13.

## JFINAL Diagnostics

400 saved candidates, 100 decisions:

- J selection: 22 % correct.
- Expected uniform selection: 9.5 %.
- Offline majority of normalized answers: 25 %.
- Oracle@4: 26 % (ceiling for this set, not a realizable system).
- Selector gap: 4 percentage points.
- Correct choice conditional on a correct candidate existing: 84.62 %.

The lack of correct/formatted proposals substantially limits this arm;
even the oracle over its four proposals does not reach B13. Majority and oracle are offline
diagnostics, without latency from an executed consensus system.

## Audit and Limitations

- The seven final ZIPs passed integrity and checks of IDs, seeds,
  configurations, input hashes, models and version. No duplicate checkpoints.
- A second parser implementation reconstructed all 600 correctness outcomes from immutable
  predictions and gold: zero discrepancies with graded_runs.csv.
- JFINAL: four finite probabilities summing to 1 and correct argmax/permutation mapping
  in 100/100 decisions; zero parseable but nonterminal candidates in the diagnostic.
- G_SINGLE had a nonblocking eager_vs_graph warning (greedy agreement 0.427):
  profile equivalence is not claimed. No tuning or rerun for quality occurred.
- Public benchmarks, possible contamination; heuristic difficulty and reviewed=false.
- One seed and a corpus of 100 problems; do not generalize to other triplets or models.
- SHARED4 uses one G copy: actual memory P_G+P_J, not four stored replicas. Different
  training, baseline expanded by third parties, non-thinking profile and vLLM kernels.
- Shutdown/monitoring operator failures were recovered without changing inference;
  a false audit failure in 08 was corrected by comparing IDs as a set, not in shuffled order.
- All VMs and guards were shut down; absence of server assignments was verified.

## Artifacts

- results/first_two_20260929/: 01_G_SINGLE and 02_B13, with the failed HF429 attempt preserved.
- results/remaining_batches_20260929/03_B13_GREEDY/ through 06_JFINAL/: remaining batches.
- results/remaining_batches_20260929/08_LATENCY/: plan, 192 measurements, swaps, logs and notebook.
- results/remaining_batches_20260929/collected/: only seven verified final ZIPs.
- results/remaining_batches_20260929/07_ANALYSIS/artifacts/: analysis ZIP and executed notebook.
- results/remaining_batches_20260929/07_ANALYSIS/extracted/: report.md, CSV tables,
  graded_runs.csv, analysis.json and figures.
- results/remaining_batches_20260929/analysis_independent_audit.json: independent regrading.
