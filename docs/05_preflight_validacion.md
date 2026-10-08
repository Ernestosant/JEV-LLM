# Preflight and Validations (§14 + Amendment A6)

> Historical context: this document describes the legacy 0.8B experiment, not the completed v3 study. See [v3 results](13_results_v3.md) and [limitations and future work](14_limitations_and_future_work.md).

All run within each condition notebook (`PREFLIGHT_MODE`): `full` (all, always), `light` (reuses
expensive checks cached in `/content/jev_llm/preflight_cache/<stack_hash>/` if available on the same VM and stack), `off`
(only inexpensive checks). `ABORT_ON_PREFLIGHT_FAIL=True` stops the notebook on a blocking failure.

| Validation | §14 Step | Blocking | What It Checks | Smoke Test Result (A100) |
|---|---|---|---|---|
| `lock` | 1–2 | yes | commit == pinned SHA; SHA-256 of **every file** (including weights) vs Hub LFS and JevK5 `SHA256SUMS`; architecture; language-model parameters from headers | ok; P_G=0.752B, P_J=8.954B |
| `parser_selftest` | 3 | yes | fractions, decimals, `2/4`, Unicode minus, `1/0`, extra text, partial FINAL, FINAL at EOS | 16/16 |
| `chat_template` | 3 | yes | published template renders `<think>\n\n</think>\n\n` with `enable_thinking=False`; text and hash saved | ok (G and B) |
| `prompt_length` | 3 | yes | rendered prompts for the selected problems ≤ 512 tokens | max 150 (N=1) |
| `sampling_semantics` | 3 | yes | `top_k=0` = disabled in vLLM; seeds < 2⁶³ | ok |
| `selector_native_reference` | 5 | yes | published JevK5 runtime (subprocess) on 8 synthetic decisions (short, mid-sentence, duplicated, ~3k tokens); IDs identical to `prompt_text()` | ok, T=1.316 |
| `selector_equivalence` | 5 | yes | vLLM vs native with the same IDs: argmax (2 ulps bf16 tolerance), |Δp| | 0 discrepancies; max |Δp| 0.022 |
| `continuity` | 4 | no | continuing from 1/63/64/65/127/128-token prefixes reproduces the greedy continuation from a single request | 100 % |
| `batching_audit` | 6 | no | 4 sequential vs batched requests (10 dev prefixes, 64 tokens) | 13.6 s vs 6.3 s → ×2.17 |
| `stress` | 6 | yes | 4 (or 1) maximum-length generations with `ignore_eos`; J input near 16,384 tokens; finite logits; peak NVML | ok; J 16,232 tokens; peak 28.6 GiB |
| `eager_vs_graph` | 8.3 | no | greedy tokens with CUDA graphs vs eager engine (3 prompts × 64) | 100 % |
| `r4_memory` (optional) | 8.3 | no | memory with 4 copies of G + J resident (no runs) | not run (optional) |

Note: continuity, batching, and eager/graph are diagnostic (non-blocking) because bf16 and batch order can
legitimately change tokens; their numbers are published. vLLM sampling is not batch-invariant: the same seed
does not guarantee the same output across GPUs/stacks (§5.2).
