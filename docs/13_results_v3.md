# Third-Round Results (v3)

## Scope

The third round is complete: it is a **small proof of concept**, not a definitive breakthrough or a general demonstration of architectural superiority. Under the fixed protocol, the hybrid achieved higher accuracy than the primary baseline; it did not demonstrate that it was faster or that it outperformed voting over the same four proposals.

This report describes completed results without rewriting the [preregistered protocol](protocols/protocol_jev_qwen4b_v3.md), data, or original results. The preparation states in [12_implementation_v3.md](12_implementation_v3.md) are historical snapshots, not the final state. The [limitations and future work](14_limitations_and_future_work.md) are part of the interpretation, not optional caveats.

## Design and Coverage

| Component | Actual coverage | Unit of analysis |
|---|---|---|
| Confirmatory quality | 6000 physical cases, exactly 500 problem IDs | 500 problems, not 6000 independent observations |
| Separate technical pilot | 50 additional IDs, 600 cases | Not included in confirmatory quality |
| Controlled latency | 1800 actual measurements on 100 test IDs | 100 problems, nine timings per system and problem |

The stochastic arms use seeds 17, 29, and 43; each greedy reference runs once per problem. JFINAL generates four complete solutions through simultaneous requests to **a single physical copy** of Qwen3.5-4B and uses JevK5-9B to select one. These are not four copies of the weights. The 1500 shared pools contain 6000 recorded candidate positions and are complete; they must not be confused with the 6000 cases in the quality matrix.

The primary comparison, fixed before the test, is JFINAL versus B13_GREEDY. B13 denotes the custom DavidAU model of approximately 13B, not an official Qwen of that size. Q9_GREEDY is the official Qwen3.5-9B control; it also does not match the hybrid's deployed parameter count. Full identities and revisions are recorded in the protocol.

BF16, vLLM 0.30.0, disabled thinking, and up to 2048 tokens per solution were used. Paired latency uses the same 40 GB A100 for both systems: three seeds by three timing repetitions, 900 measurements per system. The median of their nine timings per problem is used. Loading, unloading, probes, and warm-up are outside the case timer; this is not a cold-service or full deployment-cost measurement.

Recorded cumulative consumption was **56752.900384 GPU seconds, equivalent to 15.76469455 A100 hours**, within the total ceiling of 25 A100 hours and the maximum of two simultaneous allocations. This is cumulative resource time, not wall-clock duration, a monetary bill, or billed CU.

## Accuracy

| Condition | Physical cases | Problems | Accuracy (%) |
|---|---:|---:|---:|
| G_SINGLE: 4B with sampling | 1500 | 500 | 91.0 |
| G_GREEDY: 4B greedy | 500 | 500 | 92.2 |
| JFINAL: four proposals + selector | 1500 | 500 | 94.7333 |
| B13: custom model with sampling | 1500 | 500 | 84.2 |
| B13_GREEDY: primary baseline | 500 | 500 | 84.0 |
| Q9_GREEDY: official 9B greedy | 500 | 500 | 89.2 |

For stochastic arms, accuracy is first summarized per problem over three seeds. Greedy is not triplicated as if it were three independent samples. Model failures and truncations count as incorrect; they are not excluded to favor the result.

## Co-Primary Effects

Paired problem-cluster bootstrap, stratified by domain, with 10000 resamples and seed 271828. The two confirmatory 97.5% intervals apply Bonferroni adjustment; they are approximate.

| Effect | Estimate | Bonferroni 97.5% CI | Interpretation |
|---|---:|---|---|
| Accuracy JFINAL - B13_GREEDY | +10.7333 pp | [7.5333, 14.1333] pp | Evidence of higher accuracy on this test |
| Paired geometric speedup B13_GREEDY / JFINAL | 1.00192494 | [0.88386781, 1.12642550] | Does not demonstrate greater speed |

A speedup greater than 1 favors JFINAL. The latency interval includes 1: neither latency superiority nor **joint superiority** is demonstrated. A 10% time reduction is not demonstrated either; even the point estimate does not reach the threshold `1 / 0.9`. The quality difference exceeds the practical threshold of 5 pp, including its confirmatory lower bound. Lack of significance in latency does not demonstrate equivalence.

## Shared Controls

| Offline control on the same pool | Accuracy (%) |
|---|---:|
| Fixed original branch 0, before permutation | 90.7333 |
| Vote/plurality over four valid answers (VOTE4SHARED) | 94.6 |
| JFINAL | 94.7333 |

Voting uses valid terminal numerical answers, without gold, and breaks ties by the lowest original branch. JFINAL - shared vote is **+0.1333 pp**, with descriptive 95% CI **[-0.5333, 0.8] pp** and Holm-adjusted p **0.7653**. Therefore, **there is no evidence of a selector advantage over this vote**. Improvement over a single answer does not, by itself, identify a distinct contribution from learned critique/selection.

The controls reuse exactly JFINAL's candidates; they are not independent runs. No latency was assigned to them: these accuracies do not prove time or cost savings. Oracle@4 (96.6%) is an offline diagnostic that uses gold, not a deployable system.

## Size Anomaly and Errors

G_GREEDY achieving 92.2%, B13_GREEDY 84.0%, and Q9_GREEDY 89.2% does not prove that small models are generally better. The DavidAU baseline is a custom merge/upscale construction with 9B model components and distillations; it is not an official same-family control matched by size or training. The comparisons also do not match FLOPs, training, or the manufacturer's maximum capability.

The secondary contrast 4B greedy - official 9B greedy is +3 pp, but its Holm-adjusted p is 0.087991: it is not rejected at the 0.05 level in the fixed secondary family of five contrasts. It must not be recast as a confirmatory finding based on its unadjusted descriptive interval.

| Standalone greedy errors | Incorrect number with valid FINAL | Invalid FINAL without truncation | Truncation | Total incorrect |
|---|---:|---:|---:|---:|
| 4B | 24 | 0 | 15 | 39 |
| DavidAU ~13B | 55 | 5 | 20 | 80 |

The gap is not explained by truncations alone: the baseline also has more incorrect numerical answers. This does not determine the cause; budget, mode, training, and backend require separate controls. Switching to thinking now, increasing tokens, or adjusting sampling would not replace this comparison: it would be a new study.

## Audit and Limits

The independent strict parser reviewed **6000 predictions and found zero disagreements**. This is evidence of agreement in extraction/numerical correctness assessment, not that all gold answers are true or that the backends are equivalent.

The test contains 394 intrinsically easy problems, 96 medium, and 10 difficult; 340 come from MATH. The 150/200/150 balance is by source level, not actual difficulty. The 550 test/pilot problems underwent agent review, not human review. Public benchmarks, training contamination, and semantic duplicates that were not excluded limit generalization.

G validation included only three short prefixes comparing eager versus graph. B13 and Q9 explicitly omitted that check because of memory constraints. Continuity stress passed, but it does not prove native accuracy equivalence. J1 recovery has local byte-exact evidence with `sourceSha` and a manifest, not a persisted remote transport SHA. In LATV3, the association of memory evidence with phase H2 is supported by an in-memory projection; the raw global B3 summary was not modified. These gaps do not become complete validations merely because execution finished.

## Public Summaries

Links intended for the curated selection in `public_results/v3`, not operational artifacts:

- [Final summary](../public_results/v3/summary.md).
- [Final analytical report](../public_results/v3/report.md).
- [Structured co-primary comparison](../public_results/v3/primary_comparison.json).

Publishing these summaries does not imply publication, operational approval, or redistribution permission for the raw bundle, full dataset, or cloud logs. Execution identifiers, personal paths, and secrets are not included. The defensible conclusion remains local: **higher accuracy than the fixed baseline, without a demonstrated latency advantage or selector advantage over shared voting**.
