# Protocol Deviations and Change Log

Legacy execution history: the protocol v3 and amendment v3.1 references below belong to the earlier experiment, not new Qwen4B v3 results.

## Declared Deviations (Details in `protocolo_v3.1_enmienda.md`)

| # | Protocol v3 | Implementation | Impact / mitigation |
|---|---|---|---|
| D1 | Main profile R4-BF16-EAGER (4 physical copies of G) | SHARED4-VLLM-GRAPH (1 copy, 4 batched requests) | `4*P_G+P_J` is nominal only; actual VRAM reported; optional R4 memory check |
| D2 | Block counterbalancing in one session; do not mix latencies from different GPUs | one notebook per condition on different VMs | latencies 01-06 are descriptive; confirmatory latency in notebook 08 (one VM, counterbalanced) |
| D3 | `JevK5.decide` (transformers, `graphs=False`) | same logit reading via vLLM | blocking equivalence check against the native runtime in each hybrid notebook |
| D4 | Eager with cache; KV/DeltaNet state cloning | vLLM with CUDA graphs; re-prefill of `prompt+accepted IDs` per round | continuity and eager-vs-graph tests; rereads counted |
| D5 | 100 original problems, difficulty assessed by human review, 2nd reviewer | sampling from GSM8K/MATH/AMC/AIME; heuristic difficulty; `reviewed=false` | contamination declared; review sheet and 20 preregistered blind IDs |
| D6 | PyTorch `max_memory_allocated/reserved` | vLLM log + explicit KV + NVML at 10 Hz | "reserved" vs "required" distinguished in the report |
| D7 | Overproduction only within the terminal token | abort after the FINAL line in the stepwise loop | tokens computed but not delivered after the abort are not observable |
| D8 | - | answers >= 1000, money and percentages excluded from the dataset | avoids model-specific formatting bias |

## Change Log (Chronological Order)

| Date (UTC) | `jevlab` version | Change | Reason | Affected conditions |
|---|---|---|---|---|
| 2026-09-28 01:xx | - | Amendment v3.1 drafted and frozen before the dataset | adversarial review of the plan | - |
| 2026-09-28 02:48 | 0.1.0 | Subprocesses (`hwprobe`, `native_ref`) use the bundle's `PYTHONPATH` | `ModuleNotFoundError: jevlab` on the VM | preflight |
| 2026-09-28 02:52 | 0.1.0 | Installation uninstalls Colab's `torchaudio`/`torchcodec` | torchaudio cu128 vs torch cu130 broke `import transformers` | all |
| 2026-09-28 03:43 | 0.1.0 | vLLM logging configured before any `import vllm`; `VLLM_WORKER_MULTIPROC_METHOD=spawn` | engine startup error with no visible cause (subprocess logs lost) | all |
| 2026-09-28 04:20 | 0.1.0 | `analysis.py` without a `tabulate` dependency | report failure on local CPU | analysis |
| 2026-09-28 04:35 | **0.1.1** | **EOS removed from delivered `token_ids`** (counted as a sampled token via `eos`) | vLLM delivers `<|im_end|>` within `token_ids`: it appeared in the public text (`FINAL: 36<|im_end|>` -> invalid format) and in the candidates seen by J | **all** -> all smoke conditions repeated with 0.1.1 |
| 2026-09-28 04:35 | 0.1.1 | `code_version` included in `config_hash` | a fix never resumes previous runs (Section 11) | all |
| 2026-09-28 04:40 | - | CLI tools: retries (`_cexec.sh`), zombie detection, `kill_gpu.sh`, GPU cleanup between notebooks | fragile CLI connection; orphaned `VLLM::EngineCore` processes retained 26 GB | operations |
| 2026-09-28 05:35 | **0.1.2** | Notebook 08 synthetic prompts: 5/15/30/60 lines of "k squared is k*k" (previously 10/40/120/300 integers) | the 300-line band exceeded JSTEP's 64-round cap (`max_rounds`), censoring its latency | notebook 08 only |
| 2026-09-28 04:1x | - | Colab CLI upgraded from 0.7.0 to 0.7.4 | 0.7.0 marked a live session as lost when the token expired (~1 h) and left an orphaned VM (manually released via the `unassign` API) | operations |
