# Third-Round Implementation

> **Closing note:** this page preserves the pre-test preparation snapshot. Round v3 is now complete: 500 problems, 6,000 quality cases, and 1,800 latency measurements, with actual analysis and released resources. See [final results](13_results_v3.md) and [limitations and future work](14_limitations_and_future_work.md). The tasks and operational limits in this historical snapshot are not current publication or execution instructions.

## Historical Preparation Status

The v3 protocol and software are implemented locally; data preparation and review are in progress. **There is no frozen v3 test, no v3 experimental results, and no Colab GPU allocated by this implementation.**

The third round is defined in the [v3 protocol](protocols/protocol_jev_qwen4b_v3.md) and `config/experiment_v3.json`: 500 new test problems, 50 additional pilot problems, 100 test questions reserved for latency, three seeds, and agent-only review. The historical series in the [original protocol](protocols/protocol_jev_qwen_original.md) is not modified.

## Implemented

- Namespace `src/jevlab/v3`, version 0.3.0. The historical runtime retains 0.2.1, and its bundles remain intact.
- Six conditions, including 4B greedy and official Qwen3.5-9B greedy; three distinct seeds for stochastic arms and one reference per question for greedy.
- Durable four-proposal pools before calling J, a seed namespace separate from the policy, and failure/termination traces. Branch 0, plurality, expected uniform selection, and oracle controls reuse proposals without pretending to perform new inference or latency measurements.
- Limits 2048/8192, maximum 1024 input tokens, synthetic stress near that maximum, and selector near 16k. Mandatory validations even if a form specifies light/off; unknown versions fail closed.
- Latency study of 1800 actual requests: JFINAL and large-model greedy, 100 questions, three seeds and three repetitions, exclusive phase reloading, and loading costs outside the timer.
- Problem-cluster analysis, two co-primary 97.5% CIs, descriptive 95% CIs, and five secondary contrasts with centered bootstrap and Holm adjustment. Seeds/repetitions must not be treated as independent questions.
- Separate notebooks in `notebooks/v3/`, a development-only bundle, and gates for data, technical pilot, budget, and ETA before allocating resources.
- Isolated operator with WS15s/backend20s guard, polling25s, recovery180s, limit4h, and confirmed shutdown. Confirmatory execution requires actual authorization tied to the config and pilot; plan checks do not create that authorization.

## Actual Preparation

The initial pool was expanded with natural sources and global semantic policies before reviews began. Quotas were not lowered, and the seed was not changed. There are **3422 provisional candidates**, five source families, and 137 pairs of blind/reference packages. The 550 proposed slots have provisional capacity, with 88 priority reserve candidates; this does not mean 550 approved questions.

260 previously used identities and 113 additional identities from previously rejected constructions are excluded. The registry and exact, approximate, and numerical-variant deduplication are stored in `data/v3/staging/`.

New sources acquired with revision/hash and license: official ASDiv (CC BY-NC 4.0) and official OpenStax (CC BY-NC-SA 4.0). These permissions cover noncommercial use with attribution and, where applicable, ShareAlike; permission for commercial use is not assumed. GSM8K train was neither acquired nor used. Inherited competition keys/provenance still require verification and are not automatically presented as official.

## Review Performed

First four packages: **100 candidates** reviewed through eight actual agent runs, two per package. Each agent solved the problems blind, preserved its output, and then compared the reference without changing the blind answer. 400 events were recorded: 200 blind solutions and 200 comparisons.

Result under the current contract: **12 candidates with dual approval; 88 vetoed**. Among the 82 rejected by both reviewers, there are 70 classification discrepancies and eight gold disagreements; the groups overlap. Another six have a unilateral veto. There are also unresolved cases or cases with unit uncertainty. These data do NOT prove that 88 source answers are wrong: many discrepancies concern domain or provisional difficulty.

All verdicts and their identities/hashes are preserved in `data/v3/staging/reviews/`; the current counter is `review_progress.json`. False values are not changed to true, nor are solutions filled in to satisfy a validator. Publication remains blocked until 550 actual approvals meet the exact quotas. Adjudicating disagreements and/or reviewing reserves remains pending; selectively correcting labels to fill quotas is not appropriate.

`data/v3/PREPARATION_REPORT.md` is the preparation snapshot from before the reviews. The later status is available in the review registry; the sealed preparation is not modified retrospectively.

## Local Commands

```powershell
python -B tools/build_notebooks_v3.py
python -B tools/make_bundle_v3.py --dev
python -B tools/colab/run_v3.py --plan --phase test
python -B -m pytest tests -q -p no:cacheprovider
```

The plan is offline and returns `blocked_dataset_review`. The development bundle contains no gold, test, pilot, or private reviewer evidence. The full-bundle builder and runtime do not fabricate a data seal to bypass the block.

Final local verification for this stage: **1023 tests passed**, zero failures, and 13 existing deprecation warnings in the historical Matplotlib analysis. Notebooks were regenerated, and the dev bundle was verified after pinning the review-contract hash in the configuration. The tests use fake engines/backends; they do not test memory or accuracy on actual CUDA. Structured status: `results/v3/implementation_status.json`.

To register an actually issued review:

```powershell
python -B tools/submit_reviews_v3.py --help
```

The registrar attaches only the hashes allowed by the contract, preserves all verdicts, and checks idempotency. A fictitious agent ID must not be assigned to output generated by another procedure.

## Pending

1. Adjudicate discrepancies with evidence and continue reviewing reserve candidates. If reviewed quotas fall short, plan new sources without altering previous data or results.
2. Complete and seal exactly 500 test problems, 50 pilot problems, and 100 latency IDs, with verified policy and provenance.
3. Build full bundles and revalidate coverage, policy, hashes, and absence of gold.
4. Run smoke tests and the technical pilot in Colab; actual CUDA, memory, and native v3 equivalence have not yet been measured.
5. Present the budget and partitioning/ETA to the user. Do not launch confirmatory execution automatically upon receiving a technical go.
6. Execute, analyze, and audit only after that authorization, maintaining a maximum of two VMs and confirmed shutdown.
