# Runbook: Running the Experiment in Google Colab

> Historical context: this document describes the legacy 0.8B experiment, not the completed v3 study. See [v3 results](13_results_v3.md) and [limitations and future work](14_limitations_and_future_work.md).

**Update 2026-09-29:** `launch.sh` and `launch_seq.sh` start an independent guard:
WebSocket pings every 15 s, allocation health checks every 20 s, and shutdown with server
confirmation. The installed CLI 0.7.4 does NOT include a separate daemon. See [session lifecycle](08_session_lifecycle.md).
For isolated CLI state, export `COLAB_SESSION_CONFIG` with the same path used by
your `colab --config` wrapper; exporting a bash function is insufficient because `timeout colab`
calls the executable on PATH. `stop.sh` cancels the guard and verifies remote release.

## 0. Local Preparation (Windows)

```powershell
python data\build_dataset.py        # only if you change the dataset (already frozen)
python -m pytest tests -q            # 50 CPU tests
python tools\build_notebooks.py      # regenerates notebooks/*.ipynb
python tools\make_bundle.py          # dist\jev_llm_bundle.zip (without gold) and dist\jev_llm_analysis_bundle.zip
```

## 1. GPU per Notebook

| Notebook | Condition | GPU | Reason |
|---|---|---|---|
| `01_G_SINGLE` | G_SINGLE | **L4** (~1.7 CU/h) | 0.8B fits easily; latency marked as non-comparable |
| `02_B13`, `03_B13_GREEDY` | B13, B13_GREEDY | **A100 40GB** (~5.4 CU/h) | B ≈ 25 GB BF16 |
| `04_J64`, `05_JSTEP`, `06_JFINAL` | hybrids | **A100 40GB** | G+J ≈ 18 GiB + KV + J's 16k context; L4 (22.5 GiB) is insufficient |
| `07_analisis` | — | CPU | no GPU |
| `08_estudio_latencia` | all | **A100 40GB** | confirmatory latency on a single VM |

T4 is ruled out (no BF16). A100 80GB / G4 / H100 are unnecessary and cost more CU.
Rates: 2026 user reports; verify in Colab → *Runtime → View resources*.

## 2A. CLI Option (WSL, Recommended for Long Runs)

All commands are launched from WSL at the project root (for example, `/mnt/c/projects/JEV-LLM`).

```bash
# one notebook per VM (parallel) -------------------------------------------------
tools/colab/launch.sh J64 jev-j64 A100            # main run (form defaults)
tools/colab/launch.sh B13 jev-b13 A100
tools/colab/launch.sh G_SINGLE jev-g L4
# ... (JSTEP, JFINAL, B13_GREEDY)

# or several notebooks sequentially on ONE VM (cheaper: reuses weights and compilation)
tools/colab/launch_seq.sh jev-a100 A100 "J64 JSTEP JFINAL B13 B13_GREEDY"
tools/colab/launch_seq.sh jev-a100 A100 "LATENCY"   # latency study (08)

# papermill parameters: -p NAME VALUE (same names as the form)
tools/colab/launch.sh J64 jev-j64 A100 -p N_PROBLEMS 10 -p RUN_TAG prueba10 -p SEEDS 17,29,43

# monitoring ----------------------------------------------------------------------
tools/colab/status.sh jev-j64 J64 30       # live process, progress.json, last 30 lines, GPU, zips
tools/colab/seq_status.sh jev-a100         # sequence status (rc and duration per condition)

# download (repeatable: downloads checkpoints as they appear) -----------------------
tools/colab/pull.sh jev-j64                # -> results/colab/jev-j64/
# stop -----------------------------------------------------------------------------
tools/colab/kill_gpu.sh jev-j64            # kills papermill and processes using the GPU
tools/colab/stop.sh jev-j64                # releases the VM (download first! /content is deleted)
```

`launch*.sh` create the session if it does not exist, upload `dist/jev_llm_bundle.zip` and the notebooks to `/content`,
unpack them into `/content/jev_llm`, and start **papermill in the background** on the VM (it survives closing the
terminal). They are idempotent: if the run is still alive, they respond `ALREADY RUNNING`.

### Known CLI Issues (Observed in the Smoke Test)
* **Fragile connection** to the VM while vLLM compiles/loads (ReadTimeout, "Connection was lost"): scripts use
  `tools/colab/_cexec.sh` with 5 retries. The run on the VM is unaffected.
* **Runtime token ~1 h:** CLI 0.7.0 marked a live session as "lost" (401) and deleted its local record,
  leaving an orphaned VM accruing charges. Upgrade to ≥ 0.7.4 (`uv tool install -U google-colab-cli`). If it happens again,
  `colab sessions` lists it as `[?]`; release it through the Colab UI (*Manage sessions*).
* **HF_TOKEN secret:** outside the UI, `huggingface_hub` tries to read the Colab secret and waits for a timeout
  once (harmless: the models are public).
* `colab exec` has a default timeout of 30 s: always use `--timeout`.

## 2B. Colab UI Option

1. Upload `dist/jev_llm_bundle.zip` to `/content` (files panel), or wait for cell 3, which requests it.
2. Open the notebook (File → Upload notebook) and select the GPU from the table.
3. Adjust the form (cell 2) and run all cells. For a test: `RUN_TAG="smoke"`, `N_PROBLEMS=1`.
4. (Optional) `PERSIST_TO_DRIVE=True` mounts Drive and copies each zip to `DRIVE_DIR`.
5. If Colab disconnects: rerun from cell 3; completed cases are skipped (resumption).

## 3. Analysis

1. Download all `*_final.zip` files (and the `LATENCY` zip) with `pull.sh` or through the UI.
2. In `07_analisis` (CPU): upload `dist/jev_llm_analysis_bundle.zip` and the zips to `/content/jev_llm/incoming/`,
   set `RUN_TAG`, and run. Output: `/content/jev_llm/analysis_<RUN_TAG>.zip` with `report.md`, `final_table.csv`,
   `graded_runs.csv`, `categories.csv`, `accuracy_by_domain_difficulty.csv`, `analysis.json`, `figures/`.
3. Local alternative (replace the existing directory placeholder with the extracted-zips directory): `python -c "import sys; sys.path.insert(0,'src'); from jevlab.analysis import run_analysis; run_analysis('<extracted-zips-directory>', 'data/test_gold.jsonl', 'results/analysis', 'main')"`.

## 4. Observed Costs (Smoke Test, N=1)

See [smoke test](07_smoke_test.md). Fixed times per notebook on A100: installation ~3.5 min (first time on the VM),
weight download+verification ~2.5 min (G+J) / ~4 min (B), engine startup ~3–4 min (less with the compilation
cache), full preflight ~3–4 min. Recommendation: `launch_seq.sh` on a single A100 for the 5 A100 notebooks
and L4 for G_SINGLE; `PREFLIGHT_MODE="light"` from the second notebook onward on the same VM.
