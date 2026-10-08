# Legacy Smoke Test - 28 Sep 2026

**Objective:** run **each complete notebook** on **a single problem** (`N_PROBLEMS=1`, `RUN_TAG="smoke"`,
seed 17) to check installation, preflight, engines, loop, artifacts, analysis and tools.
**These are not experimental results** (n=1; accuracy figures are meaningless), nor new Qwen4B v3 results.

Artifacts: `results/smoke/a100/`, `results/smoke/l4/`, `results/smoke/analysis/` (output of `07_analisis`),
`results/smoke/_superseded_v0.1.0/` (first pass, before the EOS fix).

## Execution

| Notebook | VM | Method | Code | Result | Total notebook duration |
|---|---|---|---|---|---|
| 04_J64 | A100 40GB | cell by cell (debugging), then papermill | 0.1.1 | ok | 4.5 min |
| 05_JSTEP | A100 40GB | papermill (`launch_seq.sh`) | 0.1.1 | ok | 4.7 min |
| 06_JFINAL | A100 40GB | papermill | 0.1.1 | ok | 4.6 min |
| 02_B13 | A100 40GB | papermill | 0.1.1 | ok | 3.3 min |
| 03_B13_GREEDY | A100 40GB | papermill | 0.1.1 | ok | 3.4 min |
| 08_estudio_latencia | A100 40GB | papermill (2 dev + 4 synthetic, blocks of 3, 6 conditions -> 36 runs) | 0.1.1 | ok | ~18 min (B+J+G engines with sleep mode ~10 min) |
| 01_G_SINGLE | L4 | papermill (`launch.sh`) | 0.1.1 | ok | ~2 min (after first installation) |
| 07_analisis | CPU | papermill | 0.1.1 | ok | < 2 min |

Stack installation on a new VM: ~3.5 min. Subsequent notebooks on the same VM reuse packages,
downloaded weights and the vLLM compilation cache.

## Preflight (All Validations Passed)

See the full table in `05_preflight_validacion.md`. Highlights: hashes of all files for the 3 models
match the Hub; vLLM selector vs native JevK5 runtime equivalence 8/8 (|delta p| <= 0.022); token continuity
100 %; graph = eager 100 %; batching x2.17; input of 16,232 tokens to J without error.

## Measured Memory and Parameters

| | Value |
|---|---|
| P_G (linguistic, tied embeddings counted once) | 752,393,024 |
| P_J | 8,953,803,264 |
| Weights loaded by vLLM | G 1.53 GiB; J 16.71 GiB; B 23.25 GiB |
| NVML after engine startup (including reserved KV) | G+J 25.99 GiB; B 27.35 GiB; G alone (L4) 4.36 GiB |
| Peak NVML observed in cases | hybrids 26.4-28.8 GiB; B 27.4-29.6 GiB |
| Engine startup (with cache) | J ~39 s; G ~34 s; B ~43 s (first time without cache: ~3-4 min) |

## Test Case (jev-t-003, GSM8K; gold = 26)

| Condition | Status | Output (final) | T_total |
|---|---|---|---|
| B13 | final | `FINAL: 26` correct | 1.38 s |
| B13_GREEDY | final | `FINAL: 26` correct | 1.14 s |
| G_SINGLE | eos_invalid | `...84 - 10 = 74` (no FINAL) | 0.37 s (L4) |
| J64 | final | `FINAL: 36` incorrect | 0.99 s |
| JSTEP | eos_invalid | `Final answer: 26` (unsupported format) | 1.69 s |
| JFINAL | eos_invalid | J selected (p=0.98) the only candidate with correct reasoning, but none had a FINAL line | 3.62 s |

Latency study (same A100, 6 items): median speedup vs B13: J64 1.78, JFINAL 1.03, JSTEP 0.49,
G_SINGLE 1.78 (very wide CIs with n=6; validates only the mechanism).

## Findings (What Failed, Why, How It Was Resolved)

1. **EOS within `token_ids` (bug, fixed in v0.1.1).** vLLM delivers `<|im_end|>` in the IDs; it appeared in the
   text (`FINAL: 36<|im_end|>` -> invalid format) and in what J saw. It is removed from the text and counted as a
   sampled token. **All** conditions were repeated with 0.1.1 (the previous pass remains in `_superseded_v0.1.0`).
2. **Colab's torchaudio/torchcodec** (cu128) break `import transformers` after installing vLLM (torch cu130): the
   installation cell uninstalls them.
3. **Lost EngineCore logs**: vLLM configures logging on import; it is now configured before any
   `import vllm`, and everything goes to `logs/vllm.log`.
4. **Preflight correctly blocked** a J64 run where the GPU was still occupied by another kernel (the
   native J reference would not fit): the notebook stopped before measuring anything.
5. **Orphaned `VLLM::EngineCore` processes** retain ~26 GB if papermill is killed: `launch_seq.sh` and `kill_gpu.sh`
   release the GPU by PID.
6. **Colab CLI:** fragile connections while the VM compiles (retries added); CLI 0.7.0 marked a live
   session as lost when the token expired (~1 h) and left an orphaned VM accruing charges (released via API); upgraded to 0.7.4.
7. **Model behavior (not a bug; no tuning on the test):** with the protocol prompt, the 0.8B often
   omits the `FINAL:` line, writes LaTeX/blank lines or degenerates into counting (`11\n12\n13...`); B follows it
   well. G_SINGLE and hybrids are expected to accumulate `eos_without_final`/`truncated`. The protocol prohibits changing
   the prompt based on results; this will be reported as a failure category.
8. **Notebook 08 synthetic `xlong` band** (300 lines) exceeded JSTEP's 64-round cap (`max_rounds`).
   Redesigned in v0.1.2 (5/15/30/60 lines of "k squared is k*k"); notebook 08 only, validated with a local test.

## Smoke Cost and Estimated Full Run

* Smoke (including debugging): ~2.9 h of A100 + ~0.5 h of L4 + CPU minutes ~**16-18 CU**.
* Minimal run (100 problems, seed 17 + greedy), estimated from the smoke: inference of 2-15 s per
  problem depending on condition and length -> ~1-1.5 h of A100 for the 5 notebooks in sequence (`launch_seq.sh`,
  ~4 min fixed per notebook) + ~10 min of L4 + ~30-40 min of A100 for notebook 08 -> **~8-12 CU**.
  The recommended profile (3 seeds) is ~20-30 CU. Long outputs (truncated at 1024 tokens) increase the time;
  check the ETA in `progress.json` after the first 5 problems.
