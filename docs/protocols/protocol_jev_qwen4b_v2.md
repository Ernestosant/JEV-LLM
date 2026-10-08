# Protocol v2 · One Qwen3.5-4B Copy with Four Samples + JevK5-9B Selector Versus the ~13B Model

> English translation of the frozen local original `protocolo_jev_qwen4b_v2.md` (not redistributed). Source SHA-256: `08f4510d559063767ac147ba713eddee85e4cf61b64e455e66bfc854eb5cdbcd`. This translation is not a new amendment; it preserves the original experimental history.

**Project:** JEV LLM · second experimental series
**Design date:** September 30, 2026
**Status:** series completed on October 1, 2026: pilot go, 600 confirmatory predictions, 192 latency measurements, and audited CPU analysis. Implementation v0.2.1 with an authorized phase-reload amendment; the original smoke OOM is preserved. Test and pilot frozen with seed 20261001. Results in [`docs/11_results_v2.md`](../11_results_v2.md); hypotheses are not modified according to these results.
**Derived from:** [`protocolo_jev_qwen_v3.md`](protocol_jev_qwen_original.md) and [`docs/protocolo_v3.1_enmienda.md`](../protocolo_v3.1_enmienda.md). Everything this document does not modify is inherited from them. The name "v2" identifies this project's **second experimental series**; it is unrelated to the historical v1/v2 protocols superseded by v3.
**Previous results:** the Qwen3.5-0.8B series (v3 + v3.1, seed 17, 600 runs) remains closed as measured. This protocol does not correct or replace it.

## 0. Decision Summary

The 0.8B generator is replaced with **a single physical copy of `Qwen/Qwen3.5-4B`**. In each round, that same copy produces **four independent samples**, each with its own seed, processed in the same batch. No weight replicas are loaded. JevK5-9B chooses one of the four proposals with the same contract, native prompt, and calibration as in v3. The baseline remains the same derived model of approximately 13B.

**Main question:** does the "4B with four samples + selector" system outperform the ~13B model with sampling in accuracy, on a **new test** that no model has seen in this project?

Latency becomes a **secondary** question. The 4B is expected to decode more slowly than the 0.8B, and this series seeks quality, not speed.

### 0.1 Motivation and Its Adaptive Nature

The motivation comes from an **exploratory analysis after** the 0.8B series. That analysis is not part of the preregistered results.

- With the 0.8B, even reading the answer permissively (last number in the output), the best hybrid reached 36 out of 100, versus 73 for the ~13B model with sampling. Formatting does not explain the gap.
- Among four complete solutions, the "if the correct choice were always made" ceiling was 38 out of 100 with permissive reading. The selector reached 36. **The bottleneck was the generator, not the selector.**

Because this decision was made **after opening the v3 test**, that test is excluded from any confirmatory claim in this series. A new test is constructed (§3).

## 1. Fixed Models

| Role | Model (Hugging Face) | Revision | Language-model parameters | Copies on GPU |
|---|---|---|---:|---:|
| Generator G4 | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 4,205,751,296 | **1** |
| Selector J | `alibiserikbay/JevK5-9B` | `d6521a18a86999190e9d775c915af3d6d6772fc4` | 8,953,803,264 | 1 |
| Baseline B | `DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking` | `717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe` | 12,413,584,128 | 1 |

- Counts come from the safetensors headers published at those revisions. They include embeddings and the output head, count tied weights once, and exclude the vision tower and MTP. For G4 this excludes 333,514,240 vision parameters and 120,599,552 MTP parameters.
- G4 is the **post-trained** checkpoint (not `-Base`), with Apache-2.0 license, `Qwen3_5ForConditionalGeneration` architecture, 32 layers (8 full-attention layers), and tied embeddings. Only the text path is loaded, without MTP or speculative decoding.
- **Important:** G4's chat template enables thinking by default when `enable_thinking` is undefined. `enable_thinking=False` must be passed explicitly; preflight checks this and it is blocking.
- The JevK5 runtime (`v0.3.0`, commit `6c6522fe…`), calibration temperature 1.316, and letter-logit reading do not change.

### 1.0 Why Qwen3.5-4B and Not a "Qwen3.8-4B"

As of September 30, 2026, **Qwen has not published an official 4B model in the 3.8 family**. Only Qwen3.8-27B, Qwen3.8-Flash-Next (~180B MoE), and Qwen3.8-2.4T-A95B exist. Therefore, the most recent official 4B is Qwen3.5-4B.

The third-party distillate `empero-ai/Qwen3.8-4B-Distill` (revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`), a Qwen3.5-4B retrained with ~45k reasoning traces from Qwen3.8-2.4T, was evaluated and **rejected**. The reasons:

- **Its own model card reports worse GSM8K performance** than the base (78.5 % versus 85.0 %, in reasoning mode). This is precisely the capability this series needs.
- It is trained to respond **always with a `<think>` block**. That conflicts with the protocol's non-thinking mode and 1,024-token limit.
- Its training data are not public, increasing the risk of contamination with public mathematics benchmarks.

Architecture, tokenizer, and template are identical to those of Qwen3.5-4B, so it could be studied in a separate series without changing the pipeline.

### 1.1 Parameter Budget (Redefined)

With a single copy, the matching criterion becomes **unique deployed parameters**:

`P_desplegado = P_G4 + P_J = 13,159,554,560` versus `P_B = 12,413,584,128`.

Therefore, the baseline is **5.67 % smaller** than the hybrid system: `100·(P_B/P_desplegado − 1) = −5.67 %`. The "four replicas" criterion (`4·P_G4 + P_J ≈ 25.78B`) **no longer holds and is not claimed**.

The comparison is between systems with similar numbers of resident parameters. It does not equalize FLOPs, inference steps, GPU time, or training. The hybrid system **processes more tokens** (four samples plus selector reading), and this is reported.

## 2. Questions and Hypotheses

| Code | Hypothesis | Type |
|---|---|---|
| **H1** | Selection among four complete 4B solutions has **higher accuracy** than the ~13B model with sampling | **Primary (confirmatory)** |
| H2 | Selection by blocks of up to 64 tokens from the 4B has higher accuracy than the ~13B model with sampling | Secondary |
| H3 | Selection by steps from the 4B has higher accuracy than the ~13B model with sampling | Secondary |
| H4 | The selector adds value over the 4B alone (each hybrid versus one 4B solution) | Secondary |
| H5 | Latency on the same GPU (separate study) | Secondary, descriptive |

Why the primary hypothesis is selection over complete solutions:
- It was the variant that made best use of the four-proposal ceiling in the exploratory series.
- Its compute is the most bounded: a single selector reading.
- This choice is declared **before** constructing the new test.

## 3. Data

### 3.1 New Test (Frozen Before Any Inference in This Series)

- **100 problems** with the same design and rules as `data/build_dataset.py`.
- 5 domains × (8 easy, 8 medium, 4 hard).
- Same sources and pinned revisions.
- Same filters: exact answer, |answer| ≤ 999, no money, no percentages as answers, ≤ 512 rendered tokens.
- Same difficulty rules and preference for competition problems in the hard tier, with the same disclosed fillers.
- **Strict exclusion:** none of the 120 problems from the previous series (test and development), and no approximate duplicates of them, under the same 8-gram criterion.
- Construction seed: **20261001**.
- Monetary-filter correction before inference: explicit money questions such as "How much money is used?", omitted by the inherited pattern, are also excluded. The two rejected builds (one due to that filter and another due to an unauthorized seed change) are archived in `results/v2/preparation/`. The final build retains **20261001**; it was not chosen according to model results.
- Pool availability after excluding the 120 used problems, before applying length and duplicate filters:

| Domain | Easy | Medium | Hard (competition / filler) |
|---|---:|---:|---|
| Arithmetic | 442 | 207 | 0 / 8 (GSM8K ≥ 7 steps) |
| Algebra | 41 | 12 | 14 / 27 (MATH L4–5) |
| Ratios and percentages | 24 | 23 | 0 / 4 (2 GSM8K + 2 MATH) |
| Number theory | 27 | 11 | 10 / 37 |
| Counting and probability | 51 | 34 | 6 / 109 |

Three strata are tight: hard ratios (4 available for 4 slots), medium number theory (11 for 8), and medium algebra (12 for 8).

**If any stratum cannot be filled after applying filters, publish the actual reduced N.** Do not relax filters or reassign domains after looking.

- Separate files, as in v3 §3.3: `test_v2_inputs.jsonl` and `test_v2_gold.jsonl`, plus `schedule_v2.json` and `SHA256SUMS`. Inference receives only inputs.
- Human review (second reviewer and blind review of 20 preregistered IDs) must be completed **before** the confirmatory run. If not completed, disclose it as a limitation.
- The user authorized continuing without human review: `reviewed=false` in all records; it is not presented as a reviewed dataset.

### 3.2 Pilot Set

- **40 problems** taken from the remaining pool **after** constructing test v2: 8 per domain, easy and medium.
- They overlap neither test v2 nor previous data.
- They are constructed in the same commit as test v2 and frozen with it.
- Deterministic selection: 4 easy and 4 medium per domain; if medium problems are insufficient, fill with easy problems from the same domain without changing labels. The frozen pilot contains 5 easy and 3 medium in number theory, and 4+4 in the other domains.

### 3.3 Development

The 20 development problems from the previous series are reused. They serve only for software debugging, warmup, and the latency study; not for choosing configurations.

## 4. Experimental Matrix

| Condition (description) | Code | Resident models |
|---|---|---|
| Qwen3.5-4B alone: one sampled solution | `G4_SINGLE` | 1 × G4 |
| ~13B model: one sampled solution (**anchor**) | `B13` | 1 × B |
| ~13B model: one solution without sampling (sensitivity control) | `B13_GREEDY` | 1 × B |
| 4B + selector: choose among four continuations each block of up to 64 tokens | `J64_G4` | 1 × G4 + J |
| 4B + selector: choose among four continuations at the end of each step | `JSTEP_G4` | 1 × G4 + J |
| 4B + selector: choose among four complete solutions | `JFINAL_G4` | 1 × G4 + J |

- **Fixed profile: seed 17 in all arms**, as in the previous series. This gives 600 runs. This profile is not expanded according to results. Any later expansion is declared as a replication or exploratory analysis.
- Offline diagnostics on the four complete `JFINAL_G4` solutions, without additional generation: expected uniform selection, majority vote of normalized answers, four-proposal ceiling (oracle@4), and accuracy conditional on a correct proposal existing.

## 5. Common Configuration (Unchanged)

All of the following remains exactly as in v3 §5 and v3.1:

| Field | Value |
|---|---|
| Weights | BF16, no quantization or offload |
| Sampling | `temperature=0.7`, `top_p=0.9`, `top_k=0` (disabled), neutral penalties |
| Thinking mode | explicitly disabled in G4 and B |
| Maximum output | 1,024 tokens |
| Total hybrid proposal budget | 4,096 sampled tokens |
| Block limit | 64 tokens |
| Step limit | 128 tokens, maximum 64 rounds |
| G4 / B context | 4,096 tokens |
| Maximum selector input | 16,384 tokens, without truncation |
| Timeout per run | 600 s |

**Prompts:** files `prompts/generador.txt`, `prompts/criterio_paso.txt`, and `prompts/criterio_final.txt` are used **unchanged** (hashes in the manifest). The only factor that changes is the generator, to allow attribution of the effect to it.

Any prompt variant, for example explicitly asking "what is the next step?", is **outside this series**. Studying it requires another protocol, with its own development and a separate test.

## 6. Selector and Algorithms

- Selector contract: unchanged (v3 §6 and v3.1 A3). Same JSON with `option_0…option_3`, same criteria, derived-seed permutation, argmax with ties resolved to the first position, and equivalence with the native runtime as a blocking validation.
- Algorithms: unchanged (v3 §7 and v3.1 A4). Same budget rules, termination by FINAL line or EOS, overproduction, step boundaries, and step loop with timeout.

## 7. Implementation Profile: `1COPY-G4-VLLM-GRAPH`

- vLLM 0.30.0 with CUDA graphs, BF16, and the pinned v3.1 stack. One G4 engine with `max_num_seqs=4`, receiving the four requests in the same batch, and one J engine in another process.
- **Planned memory on a 40 GB A100:** G4 weights ≈ 7.8 GiB, J ≈ 16.7 GiB, explicit KV cache (G4 3 GiB, J 4 GiB), plus contexts and graphs. Total ≈ 33 GiB. This is an estimate; preflight stress testing decides.
- **If it does not fit:** stop the run and report it. No quantization, CPU offloading, or model change without a new protocol.
- **GPU:** 40 GB A100 in **all** arms, including the 4B alone, to reduce hardware heterogeneity relative to v3.1.
- `KV_CACHE_GB_G=3.0` is an infrastructure parameter fixed in this document. Everything else is as in v3.1.

## 8. Prior Validations (Preflight)

The v3.1 A6 validations remain mandatory, with these additions:

1. **G4 template without thinking** (blocking): rendering must end in `<think>\n\n</think>\n\n`, and text and hash are saved.
2. **G4 tokenizer:** save its hash and verify that EOS/PAD tokens and the set of newline-containing tokens are built for G4, without reusing the 0.8B cache.
3. **Joint stress** (blocking): four maximum-length 4B sequences plus a ~16k-token selector input, with both engines resident. Report peak memory.
4. **Eager versus CUDA graphs in G4 (A100):** a greedy-agreement threshold ≥ 0.75 is required. If it is not reached, validation does not block, but the comparison must be expanded to 20 development prompts and **published**. Additionally, the pilot repeats this check with the 0.8B on A100 to clarify the 0.427 warning observed on L4 in the previous series.
5. vLLM selector equivalence against the native runtime (blocking), as in v3.1.

### 8.1 Infrastructure Amendment: Phase Reloading for Latency

The smoke with the four G4 arms passed on a 40 GB A100: four 1,024-token outputs and a 16,330-token J input, with an observed NVML peak of 36.12 GiB. However, the latency study failed while keeping B asleep via `sleep(level=1)`: B retained 2.83 GiB of GPU memory and G4 could not reserve the fixed 3 GiB cache. The session remained connected; artifacts were retained and resource release was verified.

**User-authorized change after the OOM and before the pilot:** completely unload a phase's engines from GPU and close them before loading the next phase's engines, on the **same 40 GB A100**. B and G4+J are never resident simultaneously. There is no quantization, offload, or reduction of the G3/J4/B3 GiB caches. When measuring G4 alone, J is closed and reloaded/tested before measuring a hybrid.

Unloading, confirmed process closure, loading, selector probes, and re-warmup remain **outside `T_total`** and are recorded in `swaps.jsonl` and `warmup.jsonl`. The six conditions, counterbalanced blocks, data, seeds, and solving algorithms do not change. Reload cost is published separately; resulting latency is conditional on loaded, warm engines, not an end-to-end model-switching cost.

The amendment is fixed in `config/experiment_v2_reload.json`, code 0.2.1, `notebooks/v2-reload/`, and `dist/v2-reload/`. The configuration, 0.2.0 bundles, and all earlier artifacts remain intact. Only completed and verified development evidence is reused; the failed latency smoke is repeated under a new run name. Pilot and confirmatory runs, if authorized by go, use the new consistent revision.

## 9. Feasibility Pilot (Preregistered)

- **Arms:** `JFINAL_G4`, `B13`, and `G4_SINGLE`, on the 40-problem pilot set with seed 17.
- **Measurements:** valid format, accuracy, oracle@4, fraction of the ceiling captured by the selector, memory, times, timeouts, and OOM.
- **Rule for proceeding to the confirmatory run** (applied as written):

1. All blocking validations pass.
2. Fewer than 5 % of runs have timeout or infrastructure error.
3. Pilot `JFINAL_G4` accuracy is **greater than or equal to** pilot `B13` accuracy (point estimate).

- **If not met:** stop the series and publish the pilot as a negative feasibility result. A run after a no-go is labeled **exploratory**.
- The pilot does **not** authorize changes to prompts, hyperparameters, budgets, or models. Only infrastructure repairs are allowed, entered in the change log.
- Pilot data do not enter confirmatory analysis.
- The error rule applies both to each arm (40 expected cases) and to the aggregate (120). An incomplete run or one with unverifiable artifacts never authorizes continuation.

## 10. Execution Order

As in v3.1: one notebook per condition, **batches of at most two VMs**, a durable operator per VM, and a session guard (WebSocket pings every 15 s, assignment check every 20 s, server-confirmed closure).

1. **Pilot:** batch [`JFINAL_G4`, `B13`], then [`G4_SINGLE`]. Apply the §9 rule.
2. **Confirmatory:** batches [`B13`, `B13_GREEDY`], [`G4_SINGLE`, `J64_G4`], and [`JSTEP_G4`, `JFINAL_G4`].
3. **Latency study** on a single A100: same design as notebook 08 (20 development problems + 12 synthetic problems, counterbalanced blocks, model switching outside the timer), with G4.
4. **Analysis** on CPU, after the latency study.

Latencies from steps 1 and 2 are **descriptive**, because they come from different VMs. Only step 3 provides same-GPU latency comparisons.

## 11. Evaluation

- **Primary:** the strict v3 §11 evaluator, unchanged. Exactly one `FINAL:` line, strict numerical grammar, and exact rational comparison. Timeouts, truncations, EOS without FINAL, and round limits count as incorrect.
- **Secondary and declared exploratory:** permissive reading (last number in output) to separate formatting failures from reasoning failures. **Never used for hypothesis testing.**
- Grading is offline, after saving immutable predictions, with a second independent parser implementation as an audit, as in the previous series.

## 12. Statistics and Conclusion Rules

- **Unit:** the problem. Paired bootstrap by ID, stratified by domain, 10,000 resamples, seed 271828.
- **H1 (primary):** accuracy difference `JFINAL_G4 − B13`, in percentage points, with 95 % CI. Declare **"more accurate"** if the CI excludes 0. The practical-interest threshold is **+5 points**, and whether the point estimate exceeds it is reported.
- **H2–H4:** fixed family of five comparisons: `J64_G4` and `JSTEP_G4` versus `B13`, and each of `J64_G4`, `JSTEP_G4`, and `JFINAL_G4` versus `G4_SINGLE`. Do not select the best hybrid after seeing results. Publish Holm-adjusted exact two-sided McNemar p-values (alpha 0.05), descriptive 95 % bootstrap CIs, and conservative simultaneous Bonferroni 99 % bootstrap CIs (0.05/5). The latter are not called "Holm CIs." The primary hypothesis and its CI95 do not change.
- **Control with the ~13B model without sampling:** sensitivity control. Do not choose the best baseline per problem, but transparently report if the hybrid beats the sampled anchor and not the version without sampling.
- **Power:** with 100 problems and one seed, the 95 % CI for a paired difference has a typical half-width of about 7 points. In the previous series, "~13B without sampling versus with sampling" gave +8 points with CI95 [+1, +15]. **True differences smaller than about 8–10 points may remain unresolved.** This is reported as uncertainty, not equivalence.
- **Comparison with the 0.8B series:** descriptive only. Tests differ, so there is no pairing between series. Differences relative to the anchor within each series may be compared, always labeled as such.

## 13. Metrics and Artifacts

Same as v3 §12 and v3.1 (predictions, per-run metrics, candidates, decisions, rounds, memory, events, failures, preflight, logs, manifest, and zips). Additions:
- Table with **descriptive condition names**, in addition to the code.
- Tokens processed by the hybrid system versus the anchor (compute proxy, labeled as such).
- Fraction of the four-proposal ceiling captured by the selector.

## 14. Estimated Cost (Unmeasured)

- Preparation per notebook on A100: 12–15 min (installation, ~26 GB download in B arms, startup, and preflight). G4 download is ~9.3 GB.
- Hybrid inference with 4B: estimated at between 2 and 3 times that of 0.8B per sample. On the order of 15–30 min per arm with 100 problems.
- **Estimated total:** pilot ≈ 1.5–2 h of A100 (≈ 8–11 CU); confirmatory ≈ 4–6 h (≈ 22–32 CU); latency ≈ 45 min (≈ 4 CU). Actual ETA must be reviewed after the pilot before launching the confirmatory run.

## 15. Mandatory Limitations

- The motivation for changing the generator is adaptive (§0.1). A new test and preregistration mitigate it, but do not eliminate it.
- One seed and 100 problems. One model triplet. No generalization to "all Jev models" or other generators.
- Matching is by unique deployed parameters, not compute: the hybrid system processes more tokens.
- Different training for G4, J, and B. The baseline is a third-party expanded model. Everything runs without thinking, in vLLM with CUDA graphs, and sampling is not batch-invariant.
- Benchmarks are public: possible contamination. Difficulty is heuristic. Human review is pending if not completed beforehand.
- Hard arithmetic and ratios tiers still depend on noncompetition fillers.

## 16. Planned Implementation Changes (Before Running)

1. `config/experiment_v2.json`: G → `Qwen/Qwen3.5-4B@851bf6e8…`, `KV_CACHE_GB_G=3.0`, profile `1COPY-G4-VLLM-GRAPH`, and condition list with `_G4` codes.
2. `data/build_dataset_v2.py`: exclude the 120 used problems and their approximate duplicates, seed 20261001, generate test v2 and pilot, and publish actual N per stratum.
3. `jevlab`: version **0.2.0** (included in `config_hash`), support for `SPLIT="pilot"` and `SPLIT="test_v2"`, and descriptive labels in analysis. No changes to algorithms, selector, or evaluator.
4. Regenerated notebooks for series v2 and separate bundles (inference without gold, analysis with gold).
5. Local tests: data exclusion, quotas, hashes, absence of gold in inputs, and G4 non-thinking template.
6. Smoke with `SPLIT="dev"` and `N_PROBLEMS=1` in each new notebook **before** the pilot. Never on test v2.

Approved operational decisions: reduced smoke for the four G4 arms and minimum latency (2 dev + 4 synthetic), plus the 0.8B eager/graph check on A100. Internal codes `G_SINGLE`, `J64`, `JSTEP`, and `JFINAL` are retained; under `config/experiment_v2.json` they correspond respectively to `G4_SINGLE`, `J64_G4`, `JSTEP_G4`, and `JFINAL_G4`. `DATA_SUBDIR="data/v2"` uses files `{test,pilot,dev}_{inputs,gold}.jsonl`; inputs are packaged without gold or references. The isolated operator allows guard recovery for 180 s without restarting remote work, downloads checkpoints, and confirms closure against the server.

## 17. Protocol Change Log

| Date (UTC) | Change | Reason |
|---|---|---|
| 2026-09-30 | Initial version | Design of the one-copy 4B series, decided after exploratory analysis of the 0.8B series |
| 2026-09-30 | Official Qwen3.5-4B confirmed and rejection of `empero-ai/Qwen3.8-4B-Distill` documented (§1.0) | No official Qwen3.8-4B exists; the distillate performs worse on GSM8K according to its own model card and is trained to always think |
| 2026-09-30 | Pre-inference clarifications: fixed secondary family of five contrasts, Holm for p-values and Bonferroni for simultaneous CIs; pilot errors per arm and aggregate | Avoid post hoc selection and incorrect statistical labeling |
| 2026-09-30 | Final data 100/40, corrected monetary filter, and original seed 20261001 restored with audits of rejected candidates; human review omitted by authorization | Correct preparation before inference without relaxing quotas or filters |
| 2026-09-30 | Isolated implementation 0.2.0, internal aliases, reduced smoke, and bounded guard recovery | Authorized execution plan and protection of Colab sessions |
| 2026-10-01 | Authorized amendment 0.2.1: exclusive B versus G4+J reloading on the same A100, outside the timer; repeat only latency smoke | Actual study OOM with B asleep; retain development arms that did finish |
