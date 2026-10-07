# Artefactos de resultados

Cada notebook de condición escribe en `/content/jev_llm/results/<COND>/<COND>_<RUN_TAG>_<hash8>/` y empaqueta
todo en `/content/jev_llm/results/<run_name>_checkpoint.zip` (cada `CHECKPOINT_EVERY` casos) y `_final.zip`.
Todas las líneas JSONL se escriben con `flush + fsync` (sobreviven a desconexiones).

| Archivo | Una fila por | Campos principales |
|---|---|---|
| `manifest.json` | corrida | `config` (parámetros semánticos, modelos, runtime, hashes de prompts/datos, perfil, `code_version`), `config_hash`, `environment` (GPU nombre/UUID/driver, host, CPU, RAM, disco, versiones), `hw_probe`, `model_lock` (repo, revisión, verificación, parámetros), `tokens` (EOS, conjunto `\n`), `memory_static` (NVML antes/después, procesos, extracto del log de vLLM, kwargs de motores), `selector` (T, IDs de letras, readout), `preflight` (resumen), `stages` (duración/ok/error), `counts` |
| `predictions.jsonl` | caso (problema × semilla) | `resume_key`, `problem_id`, `seed`, `condition`, `profile`, `status` (`final`, `eos_invalid`, `truncated`, `max_rounds`, `timeout`, `selector_context_limit`, `exception`), `public_output` (cortado tras la línea FINAL), `raw_output`, `t_start_utc`, `t_end_utc` — **sin gold** |
| `metrics.jsonl` | caso | todo lo anterior salvo textos + `T_total`, `T_first_accepted`, `T_proposals`, `T_selector`, `T_prefill`, `T_control`, `T_cache_sync` (0, N/A en vLLM), `prompt_tokens`, `candidate_tokens_total`, `accepted_tokens`, `overproduction_tokens`, `public_overproduction_chars`, `rounds`, `selector_calls`, `selector_input_tokens`, `processed_prompt_tokens`, `duplicate_rounds`, `empty_steps`, `forced_boundaries`, `boundary_overshoots`, `discarded_fraction`, `nvml_peak_used_bytes`, `nvml_hz`, `rss_gib`, `tokenizer` |
| `candidates.jsonl` | candidato (ronda × rama) | `round`, `branch`, `seed_branch`, `limit`, `shown_position`, `chosen`, `text` (lo que vio J), `raw_text`, `n_tokens_delivered`, `n_sampled` (incluye EOS), `finish_reason`, `stop_reason`, `eos`, `terminal`, `has_final`, `terminal_reason`, `overproduction_tokens/chars`, `replacement_chars`, `boundary`, `boundary_overshoot`, `forced_boundary`, `empty_step`, `t_add/t_first/t_end` |
| `decisions.jsonl` | llamada a J | `round`, `perm_seed`, `permutation` (pos→rama), `prefix_tokens`, `n_unique_candidates`, `input_tokens`, `option_chars`, `logits` (A–D crudos), `probabilities` (calibradas), `winner_position`, `winner_branch`, `exact_tie`, `status`, `engine_s`, `wall_s`, `tokenize_s` |
| `rounds.jsonl` | ronda híbrida | `limit`, `round_sampled`, `accepted_after`, `candidates_total_after`, `T_proposals`, `T_selector`, `winner_branch`, `winner_terminal` |
| `memory.jsonl` | caso | pico NVML en la ventana del caso, uso actual |
| `failures.jsonl` | incidente | `kind` (`exception`, `infrastructure`, `run_aborted`), error, traceback, memoria GPU en el momento |
| `events.jsonl` | evento | inicio/fin de etapas con duración y traceback si falla |
| `warmup.jsonl` | caso de calentamiento | métricas de dev (no son resultados) |
| `preflight/` | — | `summary.json`, `lock_<ROL>.json` (hash por archivo vs Hub), `rendered_template_<ROL>.txt`, `native_ref_{in,out}.json`, `native_ref_stderr.txt`, `hw_probe.json` |
| `logs/run.log` | — | log legible (DEBUG por ronda, INFO por caso/etapa) |
| `logs/vllm.log` | — | log de todos los procesos EngineCore (carga, KV, grafos, errores) |
| `progress.json` | — | `done/total`, ETA, memoria GPU, nº de fallos (también en `/content/jev_llm/results/progress_<COND>.json`) |
| `pip_freeze.txt` | — | entorno exacto |

**Notebook 08 (`results/LATENCY/...`)**: `latency_metrics.jsonl` (mismo esquema de métricas + `item_id`, `kind`
[`dev` / `synthetic_<banda>`], `block`, `phase`), `latency_plan.json` (ítems y bloques), `swaps.jsonl` (duración de
cada intercambio B↔G+J y memoria), `warmup.jsonl`.

**Notebook 07 (`analysis_<RUN_TAG>/`)**: `report.md`, `final_table.csv` (tabla §16), `graded_runs.csv` (una fila por
caso con categoría de resultado y todas las métricas), `categories.csv`, `accuracy_by_domain_difficulty.csv`,
`analysis.json` (comparaciones bootstrap, diagnóstico JFINAL, estadísticas del selector y de pasos, latencia 08),
`figures/*.png`.

## Categorías de resultado (§11)

`correct`, `wrong_answer`, `invalid_format` (FINAL presente pero no parseable), `no_final`, `multiple_final`,
`truncated` (límite sin terminar, incl. `max_rounds`), `eos_without_final`, `timeout`, `selector_context_limit`,
`exception`. Todas salvo `correct` cuentan como incorrectas; ningún problema se retira del denominador.

## Cómo diagnosticar un fallo

1. `failures.jsonl` y `events.jsonl` → etapa y traceback. 2. `logs/vllm.log` → errores del motor (OOM, KV,
compilación). 3. `preflight/summary.json` → qué validación bloqueó. 4. Para un caso concreto: `candidates.jsonl` +
`decisions.jsonl` filtrando por `problem_id` reconstruyen la trayectoria completa (qué propuso cada rama, qué vio J,
probabilidades, qué se aceptó).
