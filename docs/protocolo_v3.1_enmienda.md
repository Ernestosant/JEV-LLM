# Amendment v3.1 to the Original Protocol

**Date:** September 27–28, 2026 · **Status:** frozen **before** building the dataset and executing
any test case. Any subsequent change is recorded in `docs/06_desviaciones_y_cambios.md` with a date.

This amendment does **not** modify models, prompts, selector criteria, budgets, seeds, or the statistics
of the [original v3 protocol](protocols/protocol_jev_qwen_original.md). It states, with justification, the implementation differences imposed by the environment
(vLLM on Google Colab Pro) and the design decisions requested by the researcher.

## A1. Primary Profile: `SHARED4-VLLM-GRAPH` (Replaces `R4-BF16-EAGER`)

* **What changes.** The four generators are **a single copy** of `Qwen/Qwen3.5-0.8B` served by a vLLM engine
  receiving **four independent requests per round**, each with its derived seed (§5.2) in the same
  batch (continuous batching). The v3 protocol provided for `SHARED4` as a separate profile (§0, §8.3); here it becomes
  the primary profile.
* **Why.** This generates the 4 candidates in parallel at lower compute cost: decoding is
  memory-bandwidth-bound, and 4 batched sequences cost almost as much as one. R4 in vLLM requires
  4 engines = 4 CUDA processes (§8.1 asks to avoid this unless measured and documented).
* **Consequence for the budget hypothesis.** `P_desplegado_R4 = 4·P_G + P_J` is reported only as a
  **nominal** calculation. Actual VRAM and inventory are `P_G + P_J`; memory is never inflated with fictitious copies.
  Parameters measured from safetensors headers (language path, without vision or MTP):
  `P_G = 752,393,024`, `P_J = 8,953,803,264` (P_B is recorded in the B13 manifest).
* **Optional R4 reference.** Parameter `RUN_R4_MEMORY_CHECK=True` in hybrid notebooks: starts 3 additional G engines
  alongside G+J, reads NVML, and shuts them down (memory only, no runs).

## A2. One Notebook per Condition on Separate VMs; Confirmatory Latency in Notebook 08

* The arms `G_SINGLE`, `B13`, `B13_GREEDY`, `J64`, `JSTEP`, `JFINAL` execute in separate notebooks that
  can run in parallel on different VMs (researcher request). This breaks block counterbalancing
  within a single session (§10.1) and compares latencies across different GPUs/hosts (§9.1).
* **Mitigation.** (i) Each run records UTC wall-clock start/end times and `perf_counter` timings (T_total,
  first accepted output, proposals, selector, prefill, control). (ii) Latencies and speedups from notebooks
  01–06 are labeled **descriptive**. (iii) Notebook **08_estudio_latencia** measures all conditions on **a
  single A100**, in counterbalanced blocks (alternating B before/after; rotated hybrid order), on a
  question set **different from the test** (20 development problems + synthetic prompts with controlled output-length
  bands). Swapping B ↔ G+J uses vLLM *sleep mode* outside the timer. **H-latency conclusions
  come from notebook 08.** (iv) A hardware calibration probe is recorded on each VM only as a
  sensitivity covariate (never for normalization).
* `G_SINGLE` runs on L4 (cheaper); its latency is not included in the six primary comparisons.

## A3. JevK5-9B Selector Served by vLLM

* The §6 contract is retained: native prompt (`jevk5.prompt.messages` + template with `enable_thinking=False`),
  JSON criterion with `option_0..option_3`, reading **letter logits**, published temperature **1.316**, decision
  by argmax (ties → first permuted position), without sampling the decision or requesting reasoning.
* **Implementation.** A single prefill pass in vLLM; the *raw logits* for letters A–D are read
  (`logprobs_mode="raw_logits"` + `logprob_token_ids`), divided by T, and softmax is applied. vLLM technically generates
  one token, which is discarded. No inter-round prefix cache (§6.1).
* **Verified equivalence (blocking preflight).** A subprocess loads the published runtime
  `jevk5.JevK5(snapshot, graphs=False)` (transformers) and compares probabilities and
  argmax on the same input IDs. Tolerance: argmax must agree unless the native top-2 margin is ≤ 2 bf16 ulps. In the smoke test:
  8/8 decisions agree, max |Δp| = 0.022.

## A4. Inference Engine and Termination

* vLLM **0.30.0** with **CUDA graphs** (profile `VLLM-BF16-GRAPH`) in all conditions; `ENFORCE_EAGER` remains a
  parameter for an eager replication. Preflight compares greedy graph versus eager tokens on dev (smoke test: 100 % identical).
* All conditions use the same **step loop** (`add_request`/`step`/`abort_request`): uniform
  termination when the FINAL line is completed, at EOS, or at a limit; measured TTFT; actual timeout of 600 s.
* **Overproduction.** Tokens delivered after the token completing the FINAL line are counted as
  overproduction and removed from the candidate seen by J and from the prefix. Tokens the engine might compute after
  `abort` and never deliver **are not observable** (declared limitation).
* **Caches.** KV/DeltaNet state is not cloned manually: each J64/JSTEP round resends `prompt + accepted IDs`
  (without re-tokenizing), and vLLM performs full prefill (prefix caching disabled by default; experimental for hybrid
  models). These rereads count as `processed_prompt_tokens`. The cloning test in §8.2 is
  replaced by a **token continuity test** (prefixes 1, 63, 64, 65, 127, 128: continuing from the prefix
  must reproduce the greedy continuation of a single request); smoke test: 100 % agreement.
* §8.1 (streams) is replaced by a **batching audit**: 4 sequential versus batched requests on 10
  development prefixes (smoke test: speedup ×2.17).
* **Memory.** vLLM runs in subprocesses, so the notebook's `torch.cuda.max_memory_*` does not apply. Reported:
  weight and KV memory from the vLLM log, KV explicitly sized with `kv_cache_memory_bytes` ("reserved"),
  and NVML sampled at 10 Hz (observed peak, not exact). `empty_cache()` is not called per round.

## A5. Dataset Sampled from Public Benchmarks (Replaces Original Problems)

* 100 test + 20 development problems sampled from **GSM8K test**, **MATH test** (algebra, number_theory,
  counting_and_probability, prealgebra), **AMC12 2022–23** (`AI-MO/aimo-validation-amc`), and **AIME 2024–2025**, with
  SHA-pinned revisions (see `docs/02_dataset.md`). 5 domains × (8 easy, 8 medium, 4 difficult).
* **Heuristic difficulty** (not human review): `<<…>>` steps in GSM8K, level in MATH, competition = difficult.
* **Preregistered exclusions:** non-exact answers (radicals, π, lists...), answers ≥ 1000 (thousands
  separator), problems whose natural answer is money or a percentage, `[asy]`, units/degrees, bases, "nearest".
* **Contamination:** GSM8K/MATH/AMC/AIME are likely to be in the models' pretraining; J declares
  training on GSM8K *train* (only *test* is used). Mandatory limitation.
* **Human review pending:** `reviewed=false`; the researcher completes `data/review_sheet.csv` and the blind
  review of the 20 preregistered IDs in `data/blind_review_ids.json` (§11).

## A6. Integrated Preflight

The §14 validations run inside each notebook (parser, lock/hashes, templates, prompt length,
sampling semantics, native reference + selector equivalence, continuity, batching, maximum-length
stress including an input of ~16k to J, eager versus graph). Blocking checks stop the notebook if they fail.

## A7. Code Versioning

`jevlab.__version__` is part of `config_hash`: a pipeline repair never resumes or mixes with
runs of the previous code (§11: repeat all affected conditions).
