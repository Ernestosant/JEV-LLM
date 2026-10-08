# Limitations and Future Work

## Interpretation

JEV-LLM v3 is a **small, completed proof of concept**, not a definitive demonstration of a new paradigm. The [v3 results](13_results_v3.md) support higher JFINAL accuracy than the custom greedy baseline under fixed conditions; they do not support latency superiority, joint superiority, or a selector advantage over voting on its own candidates.

These limitations clarify the scope of the results without modifying the [original protocol](protocols/protocol_jev_qwen4b_v3.md), frozen data, predictions, or original analyses. The motivation was adaptive following earlier rounds; the pilot and historical rounds are not included as confirmatory evidence for this test.

## External Validity

- **Limited difficulty:** 394/500 problems are intrinsically easy, 96 medium, and only 10 difficult. The 150 high-source-level problems do not equate to 150 genuinely difficult problems. Latency includes 78 easy, 21 medium, and one difficult problem.
- **Source concentration:** 340/500 problems come from MATH. Five domains with 100 problems each do not guarantee equivalent diversity in styles, provenance, or mathematical capabilities.
- **Public sources:** excluding previously used identities and applying deduplication does not exclude training contamination or all semantic/template duplicates. A test that is new to this series is not necessarily new to the models.
- **References without human review:** test and pilot were reviewed by agents (`agent_reviewed=true`, `human_reviewed=false`). Two blind solutions and documented adjudication reduce some uncertainty, but may share errors. Zero parser disagreements over 6000 predictions does not establish the mathematical truth of gold answers.
- **Narrow task:** English problems with exact numerical answers and strict FINAL formatting. There is no general evidence for open-ended proofs, symbolic answers, other languages, or nonmathematical tasks.

## Models and Budget

The DavidAU ~13B baseline is a custom merge/upscale model with 9B components and distillations, not a larger official model matched to the 4B by family and training. JevK5-9B has different training, and the official Qwen3.5-9B control does not match the hybrid's parameter count. Counting parameters does not equate to compute: four proposals, a selector, batching, and training differ.

The 4B greedy achieved +3 pp over the official 9B greedy, but Holm p=0.087991 does not allow rejection of the secondary contrast at 0.05. This does not establish an inverse scaling law. The ~13B's worse result is also not limited to truncations: its standalone errors were 55 incorrect numbers, five invalid FINALs, and 20 truncations, versus 24, zero, and 15 for the 4B.

All generators had thinking disabled and a limit of 2048 tokens per solution. That equality defines this comparison but does not measure each manufacturer's/model's best capability. Thinking modes, larger budgets, and model-appropriate sampling must be studied anew, with prior rules and validations, not used to change the original experiment's winner post hoc.

## Selector Attribution

Offline voting over the same four candidates achieved 94.6%, versus JFINAL's 94.7333%. Difference +0.1333 pp, 95% CI [-0.5333, 0.8], Holm p=0.7653: **there is no demonstrated selector advantage**. Failing to demonstrate a difference does not prove equivalence either. The shared fixed branch achieved 90.7333%; comparing it with JFINAL shows the aggregate benefit of selecting among candidates, not an identified contribution from learned critique versus majority vote.

These controls are conditional on JFINAL's pool; they have no measured latency and are not independent replications. Oracle@4 uses gold and is not deployable. A causal study of critique needs to separate generation, voting, selection, and revision, with an identical pool and accounting for their costs.

## Backend and Traceability

| Observed gap | Available evidence | What it does not establish |
|---|---|---|
| G: eager versus graph on three short prefixes | Limited check of G | Full-answer equivalence across the entire distribution |
| B13 and Q9: eager/native check omitted because of memory constraints | Explicit omission; continuity stress passed | Accuracy equivalence with independent native execution |
| J1 recovery | Local byte-exact recovery with `sourceSha` and manifest commitments | Cryptographic verification of remote transport: its SHA was not persisted |
| LATV3: phase H2 memory | Actual H2 evidence linked through an in-memory-only projection | That the raw global B3 summary already represents H2 or has been corrected |

Memory/continuity stress verifies that a path can execute, not that it preserves native accuracy. Parser agreement verifies a different layer. They must not be grouped into a single blanket approval.

J1 evidence supports local recovery, but durable remote transport proof is missing; it cannot be reconstructed retrospectively as if it had been recorded. In LATV3, the distinction between phase-specific evidence and the global summary must be maintained without mutating the raw artifact. These are audit gaps that must be disclosed, not hidden or corrected by rewriting history.

## Latency and Statistics

The paired geometric speedup was 1.00192494, 97.5% CI [0.88386781, 1.12642550]. It does not demonstrate latency superiority or a 10% reduction, and it does not demonstrate equivalence. The joint conclusion requires both co-primary objectives to exceed their thresholds; here only quality does so.

The 1800 actual measurements are nine per system and problem, on the same 40 GB A100; the inferential unit is 100 problem clusters. The 6000 quality predictions also do not equate to 6000 independent problems. Bootstrap and intervals are approximate, and the secondary family uses Holm over five fixed contrasts, not contrasts selected according to results.

Latency is conditional on warm engines. Loading, unloading, probes, and warm-up are outside the per-case timer. Timings for arms executed in different runs are descriptive and do not replace the paired study. A single campaign does not characterize all runtime variability. The cumulative 15.76469455 A100 hours (56752.900384 seconds) are neither demonstrated financial savings nor billing; the ceiling was 25 hours with a maximum of two simultaneous allocations.

## Future Priorities

Each extension must be a separate study, with prior preregistration, budget, and exclusion criteria; not a retrospective change to the original protocol.

1. **Independent native/eager validation:** choose and fix a subset before seeing its results; compare complete answers, tokens, and decisions under native, eager, and graph engines, including B13 and Q9. If memory prevents a path, disclose the limit or design another separate test, rather than treating continuity stress as a pass.
2. **Matched official controls:** compare sizes within the same official family and equivalent modes, with full revisions and verified templates. Separate the size effect from merge/upscale, distillation, and training effects; also evaluate manufacturer-recommended modes in a new matrix.
3. **Justified budgets and sampling:** study quality/cost curves with more tokens and thinking where appropriate, and sensible model-specific sampling settings validated on independent development/pilot data. Do not choose arbitrary temperatures or tune them on the final test.
4. **A genuinely difficult and novel holdout:** include enough genuinely difficult problems and new sources, independent human review of references, units, and solutions, and an explicit study of contamination and semantic duplicates. Report provenance uncertainty; do not promise that freshness eliminates all contamination.
5. **Critique versus majority vote:** isolate voting, learned selection, and critique/revision over shared candidates, with preregistered tie-breaking and no gold in decisions. Also measure latency, tokens, memory, and cost for each policy; include comparisons at comparable compute budgets.
6. **Runtime variability:** repeat counterbalanced blocks on the same hardware with documented conditions and separate warm, startup, and reload timings. Keep repetitions within the problem cluster; do not turn timing repetitions into new problems.
7. **Greater power and replication:** size a larger sample according to relevant effects and discordances before inference; preregister paired cluster analysis and multiplicity adjustment. Do not increase N, change seeds, or stop upon obtaining a favorable p-value. Replicate with another independent selection before generalizing.
8. **Durable audit and limited publication:** persist remote and local SHA values in future collections, and explicitly index memory evidence by phase. For this series, preserve and explain J1/LATV3 without mutating originals. Publish only curated summaries and review licenses and privacy before any expansion of materials.

## Publication and Conclusion

See the [final public summary](../public_results/v3/summary.md), the [analytical report](../public_results/v3/report.md), and the [co-primary comparison](../public_results/v3/primary_comparison.json), within the curated selection planned for `public_results/v3`. These links do not imply permission to redistribute the full dataset, cloud logs, or raw bundle. Nor do they represent general operational approval. Public documentation does not need personal paths, endpoints, GPU UUIDs, host identifiers, environment values, or credentials.

The useful outcome of this small proof of concept is a limited accuracy improvement and a concrete list of pending controls. **There is no evidence that the selector outperforms shared voting, that the hybrid is faster, or that small models are universally superior.** Future priorities seek to test those questions separately, not turn the original result into a definitive breakthrough.
