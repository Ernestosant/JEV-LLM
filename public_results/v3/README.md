# Aggregate v3 Proof-of-Concept Results

This directory contains a public selection from the completed study: 500 test problems, 6,000 physical quality cases, and 1,800 latency measurements on 100 problems. The separate 50-problem, 600-case pilot is not pooled with the test.

- [Summary](summary.md): principal results and verified conclusions.
- [Analytical report](report.md): methodology, sample composition, and contrasts.
- [Accuracy table](final_table.csv): six physical arms and controls derived from the same pool.
- [Co-primary comparison](primary_comparison.json): quality and latency effects with confidence intervals.
- [Secondary contrasts](secondary_comparisons.csv): the fixed family with Holm adjustment.
- [Aggregate shared controls](shared_controls_summary.csv): voting and fixed-branch results from the same pool.

These are numerical aggregates and report text, not a redistributable dataset. They contain no individual predictions, reference answers, operational data, credentials, environment-variable values, executed notebooks, or weights.

Shared-control tables reuse JFINAL proposals and **add neither physical cases nor independent problems**. Oracle@4 uses reference answers and is not deployable. Times from separate quality runs are descriptive; only the paired LATENCY study supports the speed comparison.

Read [results and scope](../../docs/13_results_v3.md) and [limitations and future work](../../docs/14_limitations_and_future_work.md) before generalizing.
