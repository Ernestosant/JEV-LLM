# JEV-LLM

**A small proof of concept for selecting among LLM-generated solutions.**

JEV-LLM explores whether a Qwen3.5-4B generator, using one physical model copy and four batched requests, can improve its mathematical answers when JevK5-9B selects among four complete solutions using option logits. The experiment uses vLLM, BF16, and CUDA graphs on Google Colab with a 40 GB A100.

This project **does not train a new model, is not a general benchmark, and does not establish that smaller models are better because of their size**. It is a limited experimental prototype whose results and limitations are published for critical assessment.

## Proof-of-concept results

Round v3 completed **500 test problems, 6,000 quality cases, and 1,800 latency measurements**. A separate 50-problem pilot produced 600 technical cases and was not pooled with the test. Cumulative consumption was **15.76 A100 hours**, within the original 25-hour limit, and all resources were released.

| Condition | Quality cases | Strict accuracy |
|---|---:|---:|
| Qwen3.5-4B, one sampled solution | 1,500 | 91.00% |
| Qwen3.5-4B, one greedy solution | 500 | 92.20% |
| 4B + JevK5-9B, final selection (`JFINAL`) | 1,500 | 94.73% |
| Custom approximately 13B baseline, sampled | 1,500 | 84.20% |
| Custom approximately 13B baseline, greedy | 500 | 84.00% |
| Official Qwen3.5-9B, greedy | 500 | 89.20% |

The approximately 13B baseline is `DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking`, a merge and upscale of checkpoints derived from 9B models, not an official 13B Qwen counterpart differing only in size. Exact revisions are recorded in [`config/experiment_v3.json`](config/experiment_v3.json).

### Limited conclusions

- **Quality against the primary approximately 13B greedy baseline:** `JFINAL` achieved +10.73 percentage points; the Bonferroni-adjusted 97.5% co-primary confidence interval was [7.53, 14.13]. Quality superiority is supported for this dataset and protocol.
- **Speed:** the paired geometric speedup was 1.002x, with a 97.5% confidence interval of [0.884, 1.126]. Neither latency superiority nor joint quality-and-speed superiority was established.
- **Selector-specific value:** voting over normalized answers from the same four proposals achieved 94.60%. `JFINAL` differed by only +0.13 points, without significance after Holm correction. The critic did not demonstrate an advantage over this simple control.
- **4B against the official 9B:** the point difference was +3 points, but the secondary contrast did not pass Holm correction (adjusted `p` approximately 0.088). This does not establish general 4B superiority.

Seeds are not treated as independent problems. Confidence intervals use paired problem-cluster bootstrap resampling, stratified by domain, with 10,000 resamples. The latency study uses 100 problems and nine measurements per system and problem on the same GPU; loading, reloads, and warmup are outside the case timer. Truncated and invalid-format outputs count as incorrect.

## Important limitations

- Models differ in training and fine-tuning, not only in parameter counts. The comparison does not isolate a size effect.
- All generators were evaluated without thinking, with a 2,048-token output limit and a strict `FINAL` format. This is a budget-constrained evaluation, not a measurement of each manufacturer's maximum recommended capability.
- The test contains 394 intrinsically easy, 96 medium, and 10 hard problems. Quotas balance **source sampling tier**, not intrinsic difficulty. Review was agentic, not human.
- Training contamination and semantic duplication in the public source corpora were not excluded.
- Eager-versus-CUDA-graph comparison was performed for the generator on three short development prefixes; it was explicitly skipped for the approximately 13B baseline and official 9B because of memory. Stress and continuity tests do not replace a complete accuracy-equivalence check against an independent backend.
- Collection and audit recoveries were documented without repeating completed inference. `JFINAL` recovery preserved original bytes but lacked a persisted remote transport hash; latency auditing used a memory-only projection of the preserved hybrid-phase proof.

These gaps are explained in [limitations and future work](docs/14_limitations_and_future_work.md). Priorities are backend validation, matched official model sizes, separate mode-and-budget studies, and broader datasets with independent reference checks before stronger claims are made.

## Documentation and published data

- [v3 results and methodology](docs/13_results_v3.md).
- [Comparison gaps and future work](docs/14_limitations_and_future_work.md).
- [Documentation index and prototype history](docs/README.md).
- [Public aggregate results](public_results/v3/README.md).
- [English translation of the v3 protocol](docs/protocols/protocol_jev_qwen4b_v3.md).

The public repository includes code, scientific configuration, prompts, tests, **unexecuted** notebooks, and aggregate results. It excludes credentials, environment-variable files, Colab session stores, logs, environment dumps, weights, frozen datasets, reference answers, and private review packets. Complete local artifacts and their delivery archive are not automatically published.

Model and data sources have separate licenses, including noncommercial restrictions and provenance that still needs clarification. Publishing this prototype grants no rights over those materials and does not replace license review.

## Structure

```text
src/jevlab/               Historical runtime and v3 namespace
config/                  Revisions, budgets, and scientific parameters
prompts/                 Generation and selection instructions
notebooks/v3/postpilot/   Final-round source notebooks, without outputs
tools/                   Preparation, auditing, and experimental operation
data/*.py                Preparation code; no private dataset payload
tests/                   Local tests and synthetic fixtures
docs/                    Results, limitations, and historical documentation
docs/protocols/          English translations of the recorded protocols
public_results/v3/       Aggregate tables and contrasts, without operational data
```

## Use and reproducibility

The scientific configuration is [`config/experiment_v3.json`](config/experiment_v3.json), and the v3 runtime is [`src/jevlab/v3/`](src/jevlab/v3/). Notebooks are generated by `tools/build_notebooks_v3.py` and require the stack specified in their installation cells.

The public clone **cannot reproduce the frozen test on its own**: the original execution required sealed data files, reviews, and local authorization evidence that are not redistributed here. Builders and admission gates fail closed when that evidence is absent. Do not fabricate seals, permissions, or artifacts to bypass them.

Original local protocol filenames retained in configuration are scientific identifiers. Their English reading copies are linked above; translation does not recreate the original local evidence or amend the experiment.

CPU tests use synthetic fixtures and require the suite's imported dependencies, including `pytest`, `numpy`, `pandas`, `sympy`, `matplotlib`, `nbformat`, and `papermill`. Some tests require Windows/WSL or local evidence that is not distributed. No fully portable, globally validated test-suite claim is made.

```bash
python -m pytest tests/test_parsing_seeds.py -q
```

Configuring authentication or launching Colab is a separate action that may consume billable resources. No credentials are embedded, and opening a notebook or cloning the repository does not implicitly authorize GPU allocation.
