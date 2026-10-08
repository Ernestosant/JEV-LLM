# JEV-LLM Documentation

This project is a small proof of concept, not a general demonstration of model superiority or a parameter-count effect. The final v3 round uses **one physical Qwen3.5-4B copy**, four batched proposals, and selection by **JevK5-9B**, alongside independent controls.

**Current status: the v3 experiment and analysis are complete.** The study verified 6,000 quality cases on 500 problems and 1,800 latency measurements on 100 problems. Cumulative consumption was 15.76 A100 hours within the 25-hour budget; all resources were released.

## Current Documentation

| Document | Contents |
|---|---|
| [13_results_v3.md](13_results_v3.md) | Final results, methodology, controls, and scope of conclusions |
| [14_limitations_and_future_work.md](14_limitations_and_future_work.md) | Comparability gaps, missing validation, and research priorities |
| [Public results](../public_results/v3/README.md) | Curated aggregate tables and comparisons |
| [v3 protocol](protocols/protocol_jev_qwen4b_v3.md) | English translation of the recorded final-round protocol |
| [12_implementation_v3.md](12_implementation_v3.md) | Historical preparation snapshot with a completion notice |

`JFINAL` improved quality against the custom approximately 13B greedy baseline, but established neither a speed advantage nor joint superiority. It also did not demonstrate a selector advantage over voting on the same proposals. Read the current reports before interpreting earlier-round numbers.

## Historical Documentation

The following documents describe different prototype stages. Their models, sample sizes, commands, and pending tasks **do not represent final v3 status**. They are preserved to expose decisions without retrospectively rewriting history. English protocol translations retain the original specifications rather than silently modernizing them.

| Document | Contents |
|---|---|
| [Original protocol](protocols/protocol_jev_qwen_original.md) | Initial 0.8B-round specification, not the final 4B v3 protocol |
| [protocolo_v3.1_enmienda.md](protocolo_v3.1_enmienda.md) | Historical SHARED4-VLLM-GRAPH amendment and benchmark dataset |
| [01_arquitectura.md](01_arquitectura.md) | Repository structure, `jevlab`, notebook flow, and decisions |
| [02_dataset.md](02_dataset.md) | Source revisions, historical 5x(8/8/4) design, filters, and limitations |
| [03_runbook_colab.md](03_runbook_colab.md) | Historical CLI/UI execution, monitoring, download, and analysis |
| [04_artefactos.md](04_artefactos.md) | Artifact schema and failure diagnosis |
| [05_preflight_validacion.md](05_preflight_validacion.md) | Implemented protocol section 14 checks and smoke outcomes |
| [06_desviaciones_y_cambios.md](06_desviaciones_y_cambios.md) | Declared deviations and change/bug history |
| [07_smoke_test.md](07_smoke_test.md) | One-problem-per-notebook smoke results and findings |
| [08_session_lifecycle.md](08_session_lifecycle.md) | Keep-alive and server-confirmed release |
| [09_results_minimal.md](09_results_minimal.md) | Historical 600-prediction/192-latency matrix and audit |
| [v2 protocol](protocols/protocol_jev_qwen4b_v2.md) | Recorded 4B v2 specification and authorized reload amendment |
| [10_execution_v2.md](10_execution_v2.md) | Historical runtime 0.2.1 coordination, recovery, and incidents |
| [11_results_v2.md](11_results_v2.md) | Completed v2 results, audits, and H1 interpretation |

## Historical Commands

These examples concern earlier rounds. They are not an automatic reproduction procedure for the public clone and do not authorize Colab spending.

```powershell
python -m pytest tests -q
python tools\build_notebooks.py; python tools\make_bundle.py
```

```bash
# WSL; historical examples only
tools/colab/launch_seq.sh jev-a100 A100 "J64 JSTEP JFINAL B13 B13_GREEDY"
tools/colab/launch.sh G_SINGLE jev-l4 L4
tools/colab/launch_seq.sh jev-lat A100 "LATENCY"
tools/colab/status.sh jev-a100 J64 ; tools/colab/pull.sh jev-a100 ; tools/colab/stop.sh jev-a100
```

The earlier workflow then used `notebooks/07_analisis.ipynb`, `dist/jev_llm_analysis_bundle.zip`, and downloaded archives. These private/generated prerequisites are not supplied by the public clone.

## Historical First-Stage Tasks

These notes are not open tasks in the completed final experiment. Current research priorities are in [14_limitations_and_future_work.md](14_limitations_and_future_work.md).

- Complete `data/review_sheet.csv` with a second reviewer and blind review of `data/blind_review_ids.json` for that earlier stage.
- The minimum historical profile used seed 17 plus greedy. Any extension must be a new replication or exploratory study, not favorable seed selection after seeing results.
