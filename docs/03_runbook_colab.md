# Runbook: ejecutar el experimento en Google Colab

**Actualizacion 2026-09-29:** `launch.sh` y `launch_seq.sh` arrancan un guard independiente:
pings WebSocket cada 15 s, salud de la asignacion cada 20 s y cierre con confirmacion del
servidor. El CLI 0.7.4 instalado NO trae un daemon separado. Ver `08_session_lifecycle.md`.
Para estado aislado de CLI, exporta `COLAB_SESSION_CONFIG` con la misma ruta usada por
tu wrapper `colab --config`; no basta exportar una funcion bash porque `timeout colab`
llama al ejecutable del PATH. `stop.sh` cancela el guard y verifica la liberacion remota.

## 0. Preparación local (Windows)

```powershell
python data\build_dataset.py        # sólo si cambias el dataset (ya está congelado)
python -m pytest tests -q            # 50 tests en CPU
python tools\build_notebooks.py      # regenera notebooks/*.ipynb
python tools\make_bundle.py          # dist\jev_llm_bundle.zip (sin gold) y dist\jev_llm_analysis_bundle.zip
```

## 1. GPU por notebook

| Notebook | Condición | GPU | Motivo |
|---|---|---|---|
| `01_G_SINGLE` | G_SINGLE | **L4** (~1.7 CU/h) | 0.8B cabe de sobra; latencia marcada como no comparable |
| `02_B13`, `03_B13_GREEDY` | B13, B13_GREEDY | **A100 40GB** (~5.4 CU/h) | B ≈ 25 GB BF16 |
| `04_J64`, `05_JSTEP`, `06_JFINAL` | híbridos | **A100 40GB** | G+J ≈ 18 GiB + KV + 16k de J; L4 (22.5 GiB) no alcanza |
| `07_analisis` | — | CPU | sin GPU |
| `08_estudio_latencia` | todas | **A100 40GB** | latencia confirmatoria en una sola VM |

T4 queda descartada (sin BF16). A100 80GB / G4 / H100 no son necesarias y cuestan más CU.
Tarifas: reportes de usuarios 2026; verifica en Colab → *Entorno de ejecución → Ver recursos*.

## 2A. Opción CLI (WSL, recomendada para corridas largas)

Todas las órdenes se lanzan desde WSL en la raíz del proyecto (por ejemplo, `/mnt/c/projects/JEV-LLM`).

```bash
# un notebook por VM (paralelo) -------------------------------------------------
tools/colab/launch.sh J64 jev-j64 A100            # corrida principal (defaults del formulario)
tools/colab/launch.sh B13 jev-b13 A100
tools/colab/launch.sh G_SINGLE jev-g L4
# ... (JSTEP, JFINAL, B13_GREEDY)

# o varios notebooks en secuencia en UNA VM (más barato: reutiliza pesos y compilación)
tools/colab/launch_seq.sh jev-a100 A100 "J64 JSTEP JFINAL B13 B13_GREEDY"
tools/colab/launch_seq.sh jev-a100 A100 "LATENCY"   # estudio de latencia (08)

# parámetros de papermill: -p NOMBRE VALOR (mismos nombres que el formulario)
tools/colab/launch.sh J64 jev-j64 A100 -p N_PROBLEMS 10 -p RUN_TAG prueba10 -p SEEDS 17,29,43

# monitoreo ----------------------------------------------------------------------
tools/colab/status.sh jev-j64 J64 30       # proceso vivo, progress.json, últimas 30 líneas, GPU, zips
tools/colab/seq_status.sh jev-a100         # estado de una secuencia (rc y duración por condición)

# descarga (repetible: baja checkpoints a medida que salen) -----------------------
tools/colab/pull.sh jev-j64                # -> results/colab/jev-j64/
# parar -----------------------------------------------------------------------------
tools/colab/kill_gpu.sh jev-j64            # mata papermill y procesos que ocupan la GPU
tools/colab/stop.sh jev-j64                # libera la VM (¡descarga antes! /content se borra)
```

`launch*.sh` crean la sesión si no existe, suben `dist/jev_llm_bundle.zip` y los notebooks a `/content`, los
descomprimen en `/content/jev_llm` y arrancan **papermill en segundo plano** en la VM (sobrevive a que cierres la
terminal). Son idempotentes: si la corrida sigue viva, responden `ALREADY RUNNING`.

### Problemas conocidos del CLI (observados en el smoke)
* **Conexión frágil** con la VM mientras vLLM compila/carga (ReadTimeout, "Connection was lost"): los scripts usan
  `tools/colab/_cexec.sh` con 5 reintentos. La corrida en la VM no se ve afectada.
* **Token del runtime ~1 h:** el CLI 0.7.0 marcó una sesión viva como "perdida" (401) y borró su registro local,
  dejando la VM huérfana facturando. Actualiza a ≥ 0.7.4 (`uv tool install -U google-colab-cli`). Si vuelve a pasar,
  `colab sessions` la lista como `[?]`; libérala desde la UI de Colab (*Gestionar sesiones*).
* **Secreto HF_TOKEN:** fuera de la UI, `huggingface_hub` intenta leer el secreto de Colab y espera un timeout una
  vez (inofensivo: los modelos son públicos).
* `colab exec` tiene timeout por defecto de 30 s: usa siempre `--timeout`.

## 2B. Opción UI de Colab

1. Sube `dist/jev_llm_bundle.zip` a `/content` (panel de archivos) o déjalo para la celda 3, que lo pide.
2. Abre el notebook (Archivo → Subir cuaderno) y elige la GPU de la tabla.
3. Ajusta el formulario (celda 2) y ejecuta todo. Para una prueba: `RUN_TAG="smoke"`, `N_PROBLEMS=1`.
4. (Opcional) `PERSIST_TO_DRIVE=True` monta Drive y copia cada zip a `DRIVE_DIR`.
5. Si Colab se desconecta: vuelve a ejecutar desde la celda 3; los casos terminados se saltan (reanudación).

## 3. Análisis

1. Descarga todos los `*_final.zip` (y el de `LATENCY`) con `pull.sh` o desde la UI.
2. En `07_analisis` (CPU): sube `dist/jev_llm_analysis_bundle.zip` y los zips a `/content/jev_llm/incoming/`,
   fija `RUN_TAG` y ejecuta. Salida: `/content/jev_llm/analysis_<RUN_TAG>.zip` con `report.md`, `final_table.csv`,
   `graded_runs.csv`, `categories.csv`, `accuracy_by_domain_difficulty.csv`, `analysis.json`, `figures/`.
3. Alternativa local: `python -c "import sys; sys.path.insert(0,'src'); from jevlab.analysis import run_analysis; run_analysis('<dir con zips extraídos>', 'data/test_gold.jsonl', 'results/analysis', 'main')"`.

## 4. Costos observados (smoke, N=1)

Ver `docs/07_smoke_test.md`. Tiempos fijos por notebook en A100: instalación ~3.5 min (primera vez en la VM),
descarga+verificación de pesos ~2.5 min (G+J) / ~4 min (B), arranque de motores ~3–4 min (menos con caché de
compilación), preflight completo ~3–4 min. Recomendación: `launch_seq.sh` en una sola A100 para los 5 notebooks de
A100 y L4 para G_SINGLE; `PREFLIGHT_MODE="light"` a partir del segundo notebook en la misma VM.
