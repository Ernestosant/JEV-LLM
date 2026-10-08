# Results of the v2 Series

Full execution, completed on October 1, 2026 at 05:14 UTC. [v2 protocol](protocols/protocol_jev_qwen4b_v2.md), with an authorized phase-reload amendment, configuration `config/experiment_v2_reload.json`, and code 0.2.1. One physical copy of the official Qwen3.5-4B generator; BF16, CUDA graphs, 40 GB A100. Inference seed 17.

## Main Result

Selection among four complete solutions achieved **92/100**, versus **85/100** for the large model with sampling. Paired difference: **+7 percentage points**, stratified bootstrap 95% CI **[0, 14]**, 10,000 resamples and seed 271828.

The point estimate exceeds the practical threshold of +5 points, but the interval **includes zero**: the preregistered rule for declaring confirmatory superiority is not met. This does not establish equality. The supporting two-sided exact McNemar test gives p=0.092285.

## Strict Accuracy

One hundred new problems per arm. Truncations, EOS without a valid FINAL, and round limits count as incorrect. Format compliance is reported separately, without recovering correct answers through permissive parsing.

| Condition | Correct / 100 | Descriptive 95% accuracy CI | Valid format |
|---|---:|---|---:|
| Qwen3.5-4B alone, one sampled solution | 88 | [82, 94] % | 95 % |
| Large model, one sampled solution (anchor) | 85 | [78, 91] % | 95 % |
| Large model, one greedy solution | 82 | [74, 89] % | 93 % |
| 4B + JevK5, selection in blocks of up to 64 tokens | 85 | [78, 91] % | 95 % |
| 4B + JevK5, selection at the end of each step | 88 | [81, 94] % | 96 % |
| 4B + JevK5, selection among four complete solutions | 92 | [87, 97] % | 96 % |

None of the five secondary contrasts rejects the null hypothesis after Holm adjustment. Final selection improves the point estimate over 4B alone by +4 points, with a descriptive 95% CI of [-1, 10]; its benefit is also not statistically confirmed.

The large-model greedy control is a sensitivity control, not a second primary hypothesis. The difference versus this control is +10 points, descriptive 95% CI [3, 18]; it does not replace the H1 result, and the baseline is not chosen after the fact.

## Selector Diagnostics

Across 400 proposals and 100 final-selection decisions:

- Ceiling with four proposals, oracle@4: 93/100.
- Correct solutions selected: 92/100; captures 92 of the 93 cases with at least one correct proposal, 98.92 %.
- Expected uniform selection: 85.5 %; majority vote over normalized answers: 91 %.
- Gap to the ceiling: one case. Four exact ties resolved according to the fixed rule.

This describes how the recorded proposals were used. It is not an additional confirmatory comparison of the selector against majority vote or uniform selection.

## Latency on the Same A100

Secondary, descriptive study: **20 development problems + 12 synthetic problems**, 32 measurements per condition, 192 in total. This is not the same set of 100 test problems. All execution statuses are included, including six truncations.

| Condition | Median | p95 |
|---|---:|---:|
| Qwen3.5-4B alone | 1.348 s | 6.393 s |
| Large model with sampling | 2.671 s | 11.130 s |
| Large model greedy | 2.489 s | 11.418 s |
| 4B + block selector | 2.024 s | 6.907 s |
| 4B + step selector | 3.530 s | 17.018 s |
| 4B + complete-solution selector | 1.748 s | 9.392 s |

The ratio of medians for the anchor versus final selection is approximately 1.53. This is not a confirmatory speed test or a joint claim of superiority in quality and latency.

Exclusive reloading of B versus G4+J, confirmed shutdown, and re-warming were **outside `T_total`**. There are 17 recorded swaps. Latency is conditional on loaded, warm engines, not the end-to-end cost of model swapping.

## Pilot and Validation

The frozen pilot of 40 problems per arm produced final selection 39/40, anchor 32/40, and 4B alone 37/40; all blocking checks passed, with zero timeouts/infrastructure errors. The go decision was applied before launching the test. Those 120 cases are not included in the confirmatory results.

The independent pilot audit agrees with the 120 published correctness assessments. The final audit agrees with **all 600 strict correctness assessments**, without discrepancies, and verifies 400 proposals, 100 decisions, probabilities calibrated at T=1.316, argmax, permutations, and ties. It also reproduces point differences and McNemar/Holm p-values. The external auditor has not numerically recalculated the bootstrap intervals; this limitation remains.

481 local tests passed, with 13 Matplotlib deprecation warnings. The seven final ZIPs, hashes, counts, IDs, notebooks, and shutdowns are verified. There are 18 canonical JSON copies of notebooks, without modifying the downloaded originals or ZIPs.

## Operations and Limits

- Maximum of two allocations, isolated and durable operators, WS guard every 15 s/backend every 20 s. All owned VMs were shut down, and the server confirms **zero active allocations**.
- The first latency smoke test failed because sleeping B retained 2.83 GiB, preventing G from reserving its 3 GiB KV cache. Its artifacts were preserved; after user authorization, only that stage was repeated with exclusive reloading. Precision, caches, budgets, and weights were not reduced.
- During proxy renewal in the full study, two ReadTimeout events occurred; the guard restored a connected socket without restarting or repeating inference.
- The cost report records 4.811 A100 hours of operational windows in the reload-revision jobs. It excludes earlier smoke tests and the failed latency attempt; it does not represent billed hours or CU. Actual CU or monetary cost is unavailable.
- One seed, 100 public benchmark problems, heuristic difficulty, omitted human review (`reviewed=false`), possible contamination, and a baseline upscaled by third parties. Deployed parameters do not equate to compute.
- The tests in the 0.8B series are different: correct-answer counts are not compared across series as if paired.

## Artifacts

- `results/v2/analysis_reload/extracted/report.md`: preserved original CPU analysis report.
- `results/v2/analysis_reload/extracted/final_table.csv` and `analysis.json`: complete tables, contrasts, and diagnostics.
- `results/v2/analysis_reload/artifacts/analysis_v2_4b_confirmatory.zip`: original analysis, SHA256 `42a4cf2eac5c31a40370ba3f988a436008fc51ea07ef9c12e0ddd3e10feff740`.
- `results/v2/confirmatory_independent_audit.json` and `.md`: external audit PASS.
- `results/v2/pilot_reload/decision.json` and `pilot_independent_audit.json`: go decision and pilot audit.
- `results/v2/collected_finals_reload/`: exactly seven final ZIPs, without duplicate checkpoints.
- `results/v2/coordinator_reload_state.json`, `summary_reload.md`, and `cost_estimate_reload.json`: complete state, shutdowns, and operational information.
- `results/v2/canonical_notebooks.json`: hashes of originals and JSON copies.
