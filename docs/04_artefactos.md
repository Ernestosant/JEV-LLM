# Result Artifacts

> Historical context: this document describes the legacy 0.8B experiment, not the completed v3 study. See [v3 results](13_results_v3.md) and [limitations and future work](14_limitations_and_future_work.md).

Each condition notebook writes to `/content/jev_llm/results/<COND>/<COND>_<RUN_TAG>_<hash8>/` and packages
everything into `/content/jev_llm/results/<run_name>_checkpoint.zip` (every `CHECKPOINT_EVERY` cases) and `_final.zip`.
All JSONL lines are written with `flush + fsync` (they survive disconnections).

| File | One Row per | Main Fields |
|---|---|---|
| `manifest.json` | run | `config` (semantic parameters, models, runtime, prompt/data hashes, profile, `code_version`), `config_hash`, `environment` (GPU name/UUID/driver, host, CPU, RAM, disk, versions), `hw_probe`, `model_lock` (repo, revision, verification, parameters), `tokens` (EOS, `\n` set), `memory_static` (NVML before/after, processes, vLLM log excerpt, engine kwargs), `selector` (T, letter IDs, readout), `preflight` (summary), `stages` (duration/ok/error), `counts` |
| `predictions.jsonl` | case (problem × seed) | `resume_key`, `problem_id`, `seed`, `condition`, `profile`, `status` (`final`, `eos_invalid`, `truncated`, `max_rounds`, `timeout`, `selector_context_limit`, `exception`), `public_output` (cut after the FINAL line), `raw_output`, `t_start_utc`, `t_end_utc` — **without gold** |
| `metrics.jsonl` | case | all the above except text + `T_total`, `T_first_accepted`, `T_proposals`, `T_selector`, `T_prefill`, `T_control`, `T_cache_sync` (0, N/A in vLLM), `prompt_tokens`, `candidate_tokens_total`, `accepted_tokens`, `overproduction_tokens`, `public_overproduction_chars`, `rounds`, `selector_calls`, `selector_input_tokens`, `processed_prompt_tokens`, `duplicate_rounds`, `empty_steps`, `forced_boundaries`, `boundary_overshoots`, `discarded_fraction`, `nvml_peak_used_bytes`, `nvml_hz`, `rss_gib`, `tokenizer` |
| `candidates.jsonl` | candidate (round × branch) | `round`, `branch`, `seed_branch`, `limit`, `shown_position`, `chosen`, `text` (what J saw), `raw_text`, `n_tokens_delivered`, `n_sampled` (includes EOS), `finish_reason`, `stop_reason`, `eos`, `terminal`, `has_final`, `terminal_reason`, `overproduction_tokens/chars`, `replacement_chars`, `boundary`, `boundary_overshoot`, `forced_boundary`, `empty_step`, `t_add/t_first/t_end` |
| `decisions.jsonl` | J call | `round`, `perm_seed`, `permutation` (position→branch), `prefix_tokens`, `n_unique_candidates`, `input_tokens`, `option_chars`, `logits` (raw A–D), `probabilities` (calibrated), `winner_position`, `winner_branch`, `exact_tie`, `status`, `engine_s`, `wall_s`, `tokenize_s` |
| `rounds.jsonl` | hybrid round | `limit`, `round_sampled`, `accepted_after`, `candidates_total_after`, `T_proposals`, `T_selector`, `winner_branch`, `winner_terminal` |
| `memory.jsonl` | case | peak NVML during the case window, current usage |
| `failures.jsonl` | incident | `kind` (`exception`, `infrastructure`, `run_aborted`), error, traceback, GPU memory at the time |
| `events.jsonl` | event | stage start/end with duration and traceback on failure |
| `warmup.jsonl` | warm-up case | dev metrics (not results) |
| `preflight/` | — | `summary.json`, `lock_<ROL>.json` (per-file hash vs Hub), `rendered_template_<ROL>.txt`, `native_ref_{in,out}.json`, `native_ref_stderr.txt`, `hw_probe.json` |
| `logs/run.log` | — | readable log (DEBUG per round, INFO per case/stage) |
| `logs/vllm.log` | — | log of all EngineCore processes (loading, KV, graphs, errors) |
| `progress.json` | — | `done/total`, ETA, GPU memory, number of failures (also in `/content/jev_llm/results/progress_<COND>.json`) |
| `pip_freeze.txt` | — | exact environment |

**Notebook 08 (`results/LATENCY/...`)**: `latency_metrics.jsonl` (same metrics schema + `item_id`, `kind`
[`dev` / `synthetic_<banda>`], `block`, `phase`), `latency_plan.json` (items and blocks), `swaps.jsonl` (duration of
each B↔G+J swap and memory), `warmup.jsonl`.

**Notebook 07 (`analysis_<RUN_TAG>/`)**: `report.md`, `final_table.csv` (§16 table), `graded_runs.csv` (one row per
case with outcome category and all metrics), `categories.csv`, `accuracy_by_domain_difficulty.csv`,
`analysis.json` (bootstrap comparisons, JFINAL diagnostic, selector and step statistics, latency 08),
`figures/*.png`.

## Outcome Categories (§11)

`correct`, `wrong_answer`, `invalid_format` (FINAL present but not parseable), `no_final`, `multiple_final`,
`truncated` (limit reached without finishing, incl. `max_rounds`), `eos_without_final`, `timeout`, `selector_context_limit`,
`exception`. All except `correct` count as incorrect; no problem is removed from the denominator.

## Diagnosing a Failure

1. `failures.jsonl` and `events.jsonl` → stage and traceback. 2. `logs/vllm.log` → engine errors (OOM, KV,
compilation). 3. `preflight/summary.json` → which validation blocked execution. 4. For a specific case: `candidates.jsonl` +
`decisions.jsonl`, filtered by `problem_id`, reconstruct the full trajectory (what each branch proposed, what J saw,
probabilities, what was accepted).
