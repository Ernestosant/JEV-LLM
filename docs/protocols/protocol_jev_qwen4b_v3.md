# Protocol v3: Third Round, 500 New Problems

> English translation of the frozen local original `protocolo_jev_qwen4b_v3.md` (not redistributed). Source SHA-256: `4efcec4ca41da394c7fa2da34054d648ad2610ba54994d13e0b479e37a367461`. This translation is not a new amendment; it preserves the original experimental history.

Design date: 2026-10-02. Status: implementation and preparation; no frozen test v3 or v3 results exist. The name identifies the third series, NOT the old [`protocolo_jev_qwen_v3.md`](protocol_jev_qwen_original.md) from the 0.8B series. The two previous rounds remain closed and immutable.

## 1. Objectives and Preregistration

Co-primary comparison: one official Qwen3.5-4B copy produces four complete solutions; JevK5-9B selects one. It is compared against the approximately 13B DavidAU model with greedy decoding, both in accuracy and controlled latency.

The study is adaptively motivated by v2. Its tests are not reused as confirmatory tests. Baselines are not chosen per question, and seeds, sources, or rules are not selected according to results. Five hundred problems improve precision but do not guarantee significance or generalization.

## 2. Models and Profiles

Full revisions and configuration in `config/experiment_v3.json`:

| Role | Model | Revision |
|---|---|---|
| G | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| J | alibiserikbay/JevK5-9B | d6521a18a86999190e9d775c915af3d6d6772fc4 |
| B | DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking | 717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe |
| O | Official post-trained Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |

BF16, text only, no quantization, offload, MTP, or speculation. CUDA graphs, vLLM 0.30.0. JevK5 runtime v0.3.0 pinned to `6c6522fe5462a05fdb82bceeb0e8c624c11f1517`; calibration 1.316. Isolated experimental package `jevlab.v3` version 0.3.0; the historical package retains its version.

A single physical G copy and four simultaneous requests, not four weight copies. Language parameters are counted from headers, excluding vision/MTP and counting tied weights once. The official 9B control does not match the hybrid's deployed parameters. No comparison equalizes FLOPs or training.

## 3. Dataset and Freshness

500 test, 50 additional pilot, 20 original dev reused only for debugging. English language. Test per genuinely reviewed domain: 30 low-level, 40 medium, 30 high ACCORDING TO SOURCE LEVEL, for multistep arithmetic, algebra, ratios/percentages, number theory, and counting/probability. Pilot per domain: 3/4/3 at those levels. Internal easy/medium/hard codes in sampling_tier do not imply the same intrinsic difficulty.

Construction seed 20261002, no alternatives. Exclude the union of 260 used problems, exact and approximate duplicates, and numerical/template variants. Separately record identifiable candidates from the two rejected v2 builds: do not confuse them with problems that underwent inference. Do not alter previous sources or recreate their freeze.

Only naturally existing problems with source identifier, revision/hash, attribution, license, and reference answer. Initial sources: GSM8K TEST, MATH TEST from the four existing domains, inherited AMC/AIME, and official ASDiv. Additional sources may be acquired before review/freezing, with adapters, pin, and license recorded. Do not invent problems or modify their numbers to fill quotas. Do not use GSM8K train as an independent test: the selector declares having used it in training. MATH train is not authorized as automatic filler.

Explicitly correct eligibility policies: mathematical `$...$` does not imply money; monetary/percentage quantities are allowed if an exact number in an unambiguous unit is requested. Gold integer, fraction, or finite decimal, magnitude <=10^9. Do not convert percentages to another scale or modify the requested quantity. Up to 1024 rendered tokens under G/B/O. Exclude necessary images, multiple-choice questions, open-ended proofs, approximations, multiple answers, or unsupported symbolic answers. Output format remains a single strict numerical FINAL line.

Difficulty is provisional until agent-based review of structure, reasoning, and number of required operations, never according to which model answers correctly. Do not change domains/difficulty to satisfy quotas. Maintain classification and rejection audits. If a stratum lacks enough valid candidates, stop freezing and expand sources; do NOT publish fewer than 500 as if they were 500.

Authorized amendment BEFORE inference: the user chooses 'Balance by source level' after checking that only 2 of the first 100 candidates are intrinsically difficult. Retain sampling_tier from the original source/proxy for 30/40/30 sampling; difficulty retains the actual intrinsic judgment, without forced quotas. Publish both distributions. Do not claim 150 genuinely difficult problems. The domain is always the genuinely reviewed one. Two blind solutions that validate mathematics/units and agree on intrinsic labels may receive ordinary approval even if they differ from the source hint; real disagreements between reviewers require adjudication. Do not rewrite previous reviews; both migrations and their evidence are preserved.

## 4. Exclusively Agent-Based Review

The user authorizes only agent-based review, not human review. Two real reviewers in separate contexts first solve the statement without gold, source, provisional label, quotas, or the other's judgment. Preserve the blind output before giving the reference to each same reviewer for a second comparison phase. Attach hashes without rewriting solutions/verdicts.

Review all 550 selected problems: clarity, uniqueness, unit, solution, rational answer, domain, and difficulty. Discrepancies require documented adjudication; do not overwrite gold solely on agent consensus. Unresolved problems are rejected and their verdicts preserved. Do not fabricate run IDs, evidence, or approvals. The verifiable contract is in `data/v3/REVIEW_CONTRACT.md`.

Pre-inference amendment 2026-10-03: SOURCE labels are provisional, not absolute truth. For two mathematically correct blind solutions that disagree only on classification or an inherently dimensionless unit, a real third agent may adjudicate final labels with reasoning and hashes of the original judgments. It cannot change gold, numbers, statements, or repair false/unresolved blind solutions. All original vetoes remain visible, and ordinary acceptance is distinguished from adjudicated acceptance. The rule applies uniformly without giving quotas to the adjudicator, before selection; no reclassification to fill quotas. Previous preparation/policy are archived byte-for-byte before migration; original packages/candidates and reviews are not rewritten.

`agent_reviewed=true`, `human_reviewed=false`, `reviewed=false` in test/pilot gold; reused dev is not presented as newly reviewed. Agent-based reviews can make correlated errors: this limitation does not disappear because there are two agents.

After review, reserve all test strata BEFORE the pilot. Select deterministically and freeze inputs, gold, policy, exclusion register, review, and hashes. Inference receives only blind inputs and a PUBLIC manifest without answers/review evidence. Solutions and references remain outside that bundle.

## 5. Matrix and Seeds

| Condition | Quality seeds | Test runs |
|---|---|---:|
| G_SINGLE: 4B alone with sampling | 17,29,43 | 1500 |
| G_GREEDY: 4B alone greedy | reference 17, effective sampling None | 500 |
| JFINAL: 4B + selector of four complete solutions | 17,29,43 | 1500 |
| B13: ~13B model with sampling | 17,29,43 | 1500 |
| B13_GREEDY: ~13B model greedy, anchor | reference 17, effective sampling None | 500 |
| Q9_GREEDY: official 9B greedy | reference 17, effective sampling None | 500 |

Total 6000 quality inferences. Blocks/steps are outside this series. Greedy runs once per problem and is referenced against three seeds, never counted as three independent samples. Reject repeated seeds.

Fixed proposal namespace `G4_FINAL4_V3`, branches 0..3; pool identity separate from policy/condition. Save durable proposals BEFORE J, even partial proposals or if J fails. Offline controls reuse exactly that pool: original branch 0 before permutation, expected uniform selection, plurality of valid TERMINAL rational answers, oracle@4. Vote tie-break: lowest represented original branch; no valid votes: branch 0 with normal grading. Never use gold for voting or assign generation latency to offline-computed controls.

## 6. Limits and Validations

Temperature 0.7, top_p 0.9, top_k 0 disabled; neutral penalties. Greedy temperature 0, seed None. Thinking explicitly disabled in all generators. Output 2048 per solution, proposals 8192 per pool, context 4096, prompt 1024, J input 16384 without truncation, timeout 600 s. A100 40 GB; caches G3/J4/B3/O3 GiB. These differences from v2 prevent attributing changes between rounds solely to the dataset.

Blocking: weight/tokenizer/template identity; parser; sampling semantics; lengths; no thinking; full snapshot hashes; native selector versus vLLM; four maximum inputs and outputs of 2048 with J16k resident; each standalone profile; absence of B/O with G+J during reloads; confirmed process closure. Unknown version or omitted validation fails closed. Eager/graph and batching diagnostics are published; greedy disagreement <0.75 is expanded to 20 dev problems and reported, not confused with a change in answers.

Do not quantize, offload, or change hardware/caches/budget/models to overcome OOM without an explicit amendment before the test. Prefix cache disabled. Declare effective engine, package, and environment options.

## 7. Controlled Latency

Selection seed 20261003: 100 test questions, 20/genuinely reviewed domain in quotas 6/8/6 BY SOURCE LEVEL; actual intrinsic difficulty reported separately. IDs frozen BEFORE inference. JFINAL and B13_GREEDY: three seeds, three real timing repetitions per seed, 900 requests per system and 1800 in total. Repetitions do not change proposal seeds; record output hashes and possible divergences.

One same A100/UUID for all pairs; counterbalanced blocks. Unloading, closure, loading, probes, and warmup remain outside T_total and are reported separately. Exclusive B versus exclusive G+J; full unloading and verification of process exit, NOT sleep with resident weights. Do not include gold in latency inputs.

For each problem and system, median of its nine times; B/J ratio. Main speedup estimator: exp(mean of log(ratio)) over 100 problems. Also report p50/p95, tokens, truncations, timeout, subset formatting/correctness, and reload windows. Do not select only correct answers or fast pairs. Technical ETA determines feasibility before confirmation; if partitioning across GPUs is needed, record a design and analysis amendment before opening the test, rather than later claiming a single UUID.

## 8. Statistics and Conclusions

Unit: problem. Quality: mean of three stochastic-arm seeds for each of 500 problems, against a single greedy reference. Latency: 100 clusters; never 1800 independent observations. Paired domain-stratified bootstrap, 10000 resamples, seed 271828; keep seeds/repetitions together.

Two co-primary objectives: JFINAL-B13_GREEDY accuracy difference and that pair's speedup. Confirmatory CI 97.5% Bonferroni over 2, descriptive CI95. More accurate if lower bound>0; faster if speedup lower bound>1; joint superiority requires BOTH. Practical interest: quality point estimate>=5pp and time reduction>=10% (speedup>=1/0.9), separately report whether the full CI supports it. Do not interpret lack of significance as equivalence.

FIXED secondary family of five: JFINAL-G_SINGLE, JFINAL-shared vote, G_GREEDY-Q9_GREEDY, JFINAL-Q9_GREEDY, G_SINGLE-B13. Approximate two-sided p-values from centered/clustered bootstrap, Holm correction for 5. Do not apply McNemar to fractional means of 3 seeds or to 1500 rows as independent. Uniform selection and oracle are diagnostics, not deployable arms.

Before test inference, power analysis over relevant difference/discordance scenarios; report uncertainty if 500 is insufficient. Do not increase N or select seeds according to p-values. Strict offline grading of immutable predictions, independent numerical auditor, and independent recalculation of intervals/p-values. All model failures, truncations, and timeouts count as incorrect; platform incidents are retained and only unfinished work is recovered, without deleting failures or retrying for quality. Do not exclude questions after results.

## 9. Technical Pilot and Budget

50 questions across the six profiles: 150 per stochastic profile and 50 per greedy profile, 600 runs. Technical criterion: blocking checks passed, complete coverage, and timeout/infraunion<5% per arm and aggregate; NO criterion of beating the baseline. Publish formatting/tokens/time/memory and test/latency ETA without quality tuning.

Technical go does NOT authorize the confirmatory budget. Require subsequent user authorization tied to the config hash and pilot report. CU/monetary cost are not invented: distinguish observed time, extrapolation, and unavailable billing. Partitions and deadlines are fixed before the test.

Explicit user authorization 2026-10-03: execute through results with a TOTAL CAP of 25 accumulated A100 hours, including smoke, pilot, test, and latency, not 25 additional hours after the pilot. Confirmation tied to the report/config is materialized after verifying technical go and ETA within the remaining balance, under this already granted conditional authorization. A stable per-experiment ledger retains spending and reservations even if a JSON format changes. When the balance is exhausted, preserve artifacts and close owned VMs; do not expand the ceiling without new authorization. This does not mean CU/billed cost is known.

## 10. Colab Operation

Maximum 2 simultaneous assignments, all A100 40 GB; one GPU for latency. Session-isolated CLI, executable wrapper, and COLAB_SESSION_CONFIG; CLI Python for APIs/guard/closure. WebSocket 15 s/backend 20 s guard, monitor 25 s, and checks 2/5/10/30 min. Bounded recovery 180 s; transport timeout does not mean the notebook is dead. Durable operator, WSL host, and coordinator independent of the agent's turn. Do not trust CLI 0.7.4's automatic daemon.

Download checkpoints, verify hashes/coverage/notebooks/ZIPs, release immediately upon completion, and confirm endpoint absence on the server. Never stop others' sessions or repeat live remote work. A real VM loss requires preservation, identification of incomplete work, and a recovery record. It does not modify seeds/profiles. Initial deadline 4 h per VM, revised with pilot ETA before confirmation; no silent extension of charges. At completion, zero owned assignments; external ones are declared and left untouched.

## 11. Implementation and Deliverables

Separate namespaces: `data/v3`, `config/experiment_v3.json`, `src/jevlab/v3`, `notebooks/v3`, `dist/v3`, `results/v3`. Do not overwrite the old v3 or v2 protocol, their data, notebooks, bundles, or results. Pipeline: acquisition/review -> freeze -> tests/bundles -> dev smoke -> technical pilot -> budget authorization -> 6000 quality -> 1800 latency -> CPU analysis/audit -> closure.

Explicit versions, policies, and schemas. Exact verifier of 500 IDs/3 seeds or reference 500, 1800 real measurements, and pools with complete branches or explicitly accounted failures. Public manifest without gold linked by hashes; detailed review evidence separated into analysis. Report with descriptive names and human-review flags false.

## 12. Mandatory Limitations

Adaptive motivation; only one main triplet and one official control; public benchmarks and possible contamination; heterogeneous sources/licenses and difficulty; agent-based review without human review; four proposals consume more compute; different sampling policies; latency conditional on warm engines; batch/precision effects; different datasets and budgets between rounds. Do not claim conclusions for all models or all mathematical problems.

## Change Log

2026-10-02: third-series protocol registered before the test. User fixes 500 test + 50 pilot, official 9B control, agent-based review only, and authorizes starting implementation. Capacity/review status in `data/v3/PREPARATION_REPORT.md`; data not frozen until actual requirements are met.

2026-10-03: user requires executing everything through results and authorizes 25 total A100 hours. Classification/unit adjudication is implemented with evidence preserved before any inference, and cumulative budget and artifact integrity are strengthened. Hypotheses, seeds, weights, quotas, and token budgets do not change.

Pre-inference sampling amendment: the user authorizes balance by source level, retaining 500 test + 50 pilot and 100 per domain. sampling_tier and reviewed intrinsic difficulty are separated; both are published without claiming high level equals actual difficulty. Models, hypotheses, seeds, tokens, and GPU budget do not change.
