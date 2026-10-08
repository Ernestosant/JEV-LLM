# JEV-LLM v3 Results

500 test problems; 6000 quality inference cases and 1800 latency measurements.
Seeds are not independent problems. The primary baseline is the approximately 13B greedy model.

## Primary Comparison

- Accuracy difference JFINAL - B13_GREEDY: 10.7333 percentage points.
- 97.5% accuracy confidence interval: [7.533333333333333, 14.133333333333333].
- Paired geometric speedup: 1.00192; values greater than 1 favor the hybrid.
- 97.5% speedup confidence interval: [0.883867813624088, 1.1264255001970094].

Co-primary confidence intervals use domain-stratified problem-cluster bootstrap resampling and Bonferroni adjustment.
Latency uses 100 problems and the median of nine measurements per system and problem.

## Accuracy

| Condition | Cases | Problems | Accuracy % |
|---|---|---|---|
| G_SINGLE | 1500 | 500 | 91 |
| JFINAL | 1500 | 500 | 94.7333 |
| B13 | 1500 | 500 | 84.2 |
| G_GREEDY | 500 | 500 | 92.2 |
| B13_GREEDY | 500 | 500 | 84 |
| Q9_GREEDY | 500 | 500 | 89.2 |
| FIXED_BRANCH0_SHARED | 1500 | 500 | 90.7333 |
| VOTE4SHARED | 1500 | 500 | 94.6 |

## Verified Conclusions

```json
{
  "quality_superiority": true,
  "latency_superiority": false,
  "joint_superiority": false,
  "quality_practical_point_5pp": true,
  "latency_practical_point_10pct_reduction": false,
  "rule": "Quality lower97.5 > 0; latency lower97.5 > 1; joint requires both. Practical thresholds are point estimates."
}
```

Balance is by source sampling tier, not intrinsic difficulty. Review was agentic, not human.
The final two table entries are offline shared-pool controls, not additional physical inference cases.
See [report.md](report.md) and [limitations and future work](../../docs/14_limitations_and_future_work.md) for controls, secondary contrasts, auditing, and limitations. The complete private analysis JSON is not redistributed here.
