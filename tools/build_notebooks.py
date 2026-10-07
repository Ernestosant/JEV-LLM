"""Generate notebooks 01-08 (Colab forms + papermill 'parameters' tag).

python tools/build_notebooks.py   -> notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
VLLM_PIN = "0.30.0"
JEVK5_COMMIT = "6c6522fe5462a05fdb82bceeb0e8c624c11f1517"

COND_INFO = {
    "G_SINGLE": ("01_G_SINGLE", "L4", "Control del generador: **una** solución de Qwen3.5-0.8B, sin selector.",
                 "G (1 copia, 1 secuencia)"),
    "B13": ("02_B13", "A100", "Baseline principal: **una** solución del 13B con muestreo (T=0.7, top_p=0.9).",
            "B (13B)"),
    "B13_GREEDY": ("03_B13_GREEDY", "A100", "Control de sensibilidad: el 13B con decodificación **greedy** "
                   "(una sola pasada determinista por problema).", "B (13B)"),
    "J64": ("04_J64", "A100", "Híbrido: 4 bloques de hasta **64 tokens** por ronda; JevK5-9B elige uno; se "
            "continúa desde el prefijo elegido.", "G (1 copia, 4 secuencias en batch) + J (9B)"),
    "JSTEP": ("05_JSTEP", "A100", "Híbrido: 4 **pasos** (hasta el primer salto de línea, máx. 128 tokens) por "
              "ronda; máx. 64 rondas.", "G (1 copia, 4 secuencias en batch) + J (9B)"),
    "JFINAL": ("06_JFINAL", "A100", "Híbrido: 4 **soluciones completas**; una sola decisión de JevK5-9B.",
               "G (1 copia, 4 secuencias en batch) + J (9B)"),
}


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str, title: str | None = None, tags=None):
    src = (f"# @title {title}\n" if title else "") + s.strip("\n")
    c = nbf.v4.new_code_cell(src)
    if tags:
        c.metadata["tags"] = tags
    return c


INSTALL = f'''
# Instala el stack fijado ANTES de importar torch en este kernel.
# vLLM {VLLM_PIN} trae su propio torch/transformers; jevk5 se instala sin dependencias
# para no alterar ese stack. Si ya está instalado (re-ejecución), no hace nada.
import importlib.metadata as md, subprocess, sys, time, json
def _v(p):
    try: return md.version(p)
    except md.PackageNotFoundError: return None
t0 = time.time()
need = []
if _v("vllm") != "{VLLM_PIN}": need.append("vllm=={VLLM_PIN}")
for p in ("nvidia-ml-py", "papermill", "pandas", "matplotlib", "tabulate"):
    if _v(p) is None: need.append(p)
if need:
    print("instalando:", need, flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode: raise SystemExit("pip install falló (ver arriba)")
# Colab trae torchaudio/torchcodec compilados para otro CUDA; transformers los importa si
# existen y falla al mezclar versiones. No se usan en este experimento: se desinstalan.
for p in ("torchaudio", "torchcodec"):
    if _v(p) is not None:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", p])
        print("desinstalado (incompatible con el torch de vLLM):", p)
def _jevk5_commit():
    try:
        return json.loads(md.distribution("jevk5").read_text("direct_url.json") or "{{}}").get("vcs_info", {{}}).get("commit_id")
    except (md.PackageNotFoundError, ValueError):
        return None
if _v("jevk5") is None or _jevk5_commit() != "{JEVK5_COMMIT}":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                         "--force-reinstall", "git+https://github.com/allebee/jevk5@{JEVK5_COMMIT}"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    if r.returncode: raise SystemExit("instalación de jevk5 falló")
print({{p: _v(p) for p in ("vllm", "torch", "transformers", "jevk5", "nvidia-ml-py")}})
print(f"instalación lista en {{time.time()-t0:.0f}} s")
'''

BUNDLE = '''
# Localiza el código/datos del experimento en /content/jev_llm (subidos con el CLI o a mano).
# Si no existen: sube el archivo jev_llm_bundle.zip a /content (o se te pedirá aquí).
import os, sys, json, hashlib, zipfile, pathlib
ROOT = pathlib.Path(globals().get("ROOT", "/content/jev_llm"))
ZIP = pathlib.Path("/content/jev_llm_bundle.zip")
if not (ROOT / "src" / "jevlab").exists():
    if not ZIP.exists():
        try:
            from google.colab import files  # solo en la UI de Colab
            print("Sube jev_llm_bundle.zip ...")
            up = files.upload()
            ZIP.write_bytes(next(iter(up.values())))
        except Exception as e:
            raise SystemExit(f"No encuentro {ZIP} ni puedo pedirlo ({e}). Súbelo con el CLI.")
    zipfile.ZipFile(ZIP).extractall(ROOT)
man = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text())
bad = [f for f, h in man["files"].items()
       if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
print(f"bundle {man['bundle_version']} creado {man['created_utc']}: {len(man['files'])} archivos, "
      f"{'OK' if not bad else 'HASH MISMATCH: ' + str(bad)}")
if bad: raise SystemExit("bundle alterado")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
'''

DRIVE = '''
# (Opcional) Persistencia en Google Drive: requiere autorización interactiva en la UI.
# Con el CLI/papermill déjalo en False y descarga los zips con tools/colab/pull.sh.
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
'''


def params_cell(cond: str, gpu: str, series: str = "v1") -> str:
    if series == "v2-reload":
        return params_cell(cond, gpu, "v2").replace("config/experiment_v2.json", "config/experiment_v2_reload.json")
    return f'''
# ======================= PARÁMETROS (formulario de Colab) =======================
# Valores por defecto = {'protocolo 4B v2' if series == 'v2' else 'protocolo v3 + enmienda v3.1'}. Cambiar un parámetro semántico
# cambia el config_hash y crea una carpeta de resultados distinta (no se mezclan).
CONDITION = "{cond}"  # fijo en este notebook
CONFIG_FILE = "config/experiment{'_v2' if series == 'v2' else ''}.json"  # @param {{type:"string"}}
DATA_SUBDIR = "{'data/v2' if series == 'v2' else 'data'}"  # @param {{type:"string"}}
ROOT = "/content/jev_llm{'_v2' if series == 'v2' else ''}"
RUN_TAG = "main"  # @param {{type:"string"}}
# ^ usa "smoke" para pruebas: separa carpetas y reanudación de la corrida real
N_PROBLEMS = 100  # @param {{type:"integer"}}
# ^ cuántos problemas del test (en el orden contrabalanceado congelado) se procesan
SEEDS = "17"  # @param ["17", "17,29,43"] {{allow-input: true}}
SPLIT = "test"  # @param ["test", "pilot", "dev"]
TEMPERATURE = 0.7  # @param {{type:"number"}}
TOP_P = 0.9  # @param {{type:"number"}}
TOP_K = 0  # @param {{type:"integer"}}
REPETITION_PENALTY = 1.0  # @param {{type:"number"}}
N_CANDIDATES = 4  # @param {{type:"integer"}}
MAX_OUTPUT_TOKENS = 1024  # @param {{type:"integer"}}
CANDIDATE_BUDGET = 4096  # @param {{type:"integer"}}
J64_BLOCK = 64  # @param {{type:"integer"}}
JSTEP_STEP = 128  # @param {{type:"integer"}}
JSTEP_MAX_ROUNDS = 64  # @param {{type:"integer"}}
GB_CONTEXT = 4096  # @param {{type:"integer"}}
J_MAX_INPUT = 16384  # @param {{type:"integer"}}
TIMEOUT_S = 600  # @param {{type:"integer"}}
ENFORCE_EAGER = False  # @param {{type:"boolean"}}
KV_CACHE_GB_G = {3.0 if series == 'v2' else 2.0}  # @param {{type:"number"}}
KV_CACHE_GB_J = 4.0  # @param {{type:"number"}}
KV_CACHE_GB_B = 3.0  # @param {{type:"number"}}
J_READOUT = "logprob_token_ids"  # @param ["logprob_token_ids", "full_vocab"]
PREFLIGHT_MODE = "full"  # @param ["full", "light", "off"]
ABORT_ON_PREFLIGHT_FAIL = True  # @param {{type:"boolean"}}
RUN_R4_MEMORY_CHECK = False  # @param {{type:"boolean"}}
WARMUP_N = 3  # @param {{type:"integer"}}
CHECKPOINT_EVERY = 5  # @param {{type:"integer"}}
HASH_WEIGHTS = True  # @param {{type:"boolean"}}
REQUIRED_GPU = "auto"  # @param ["auto", "A100", "L4", "H100", "any"]
# ^ auto = {gpu} para esta condición
PERSIST_TO_DRIVE = False  # @param {{type:"boolean"}}
LOG_LEVEL = "INFO"  # @param ["INFO", "DEBUG"]
'''


PARAM_NAMES = ["CONDITION", "CONFIG_FILE", "DATA_SUBDIR", "ROOT", "RUN_TAG", "N_PROBLEMS", "SEEDS", "SPLIT", "TEMPERATURE", "TOP_P", "TOP_K",
               "REPETITION_PENALTY", "N_CANDIDATES", "MAX_OUTPUT_TOKENS", "CANDIDATE_BUDGET", "J64_BLOCK",
               "JSTEP_STEP", "JSTEP_MAX_ROUNDS", "GB_CONTEXT", "J_MAX_INPUT", "TIMEOUT_S", "ENFORCE_EAGER",
               "KV_CACHE_GB_G", "KV_CACHE_GB_J", "KV_CACHE_GB_B", "J_READOUT", "PREFLIGHT_MODE",
               "ABORT_ON_PREFLIGHT_FAIL", "RUN_R4_MEMORY_CHECK", "WARMUP_N", "CHECKPOINT_EVERY",
               "HASH_WEIGHTS", "REQUIRED_GPU", "PERSIST_TO_DRIVE", "LOG_LEVEL"]


def condition_notebook(cond: str, series: str = "v1") -> nbf.NotebookNode:
    if series == "v2-reload":
        nb = condition_notebook(cond, "v2")
        _reload_notebook(nb)
        return nb
    fname, gpu, desc, resident = COND_INFO[cond]
    if series == "v2":
        cfg = json.loads((ROOT / "config" / "experiment_v2.json").read_text(encoding="utf-8"))
        gpu, desc = "A100", cfg["condition_labels"][cond]
        resident = "B (~13B)" if cond.startswith("B13") else "G4 (1 physical copy, max 4 batched requests)" + (" + J (9B)" if cond.startswith("J") else "")
    nb = nbf.v4.new_notebook()
    params_dict = "dict(" + ", ".join(f"{p}={p}" for p in PARAM_NAMES) + ")"
    nb.cells = [
        md(f"""
# JEV-LLM · {fname} · condición `{cond}`

{desc}

* **Modelos residentes:** {resident} · **GPU requerida:** {gpu} (Colab Pro) · perfil `SHARED4-VLLM-GRAPH` (enmienda v3.1)
* Protocolo: `protocolo_jev_qwen_v3.md` + `docs/protocolo_v3.1_enmienda.md`.
* Las predicciones **no** contienen gold: la corrección se hace offline en `07_analisis`.

### Cómo ejecutarlo
1. **UI de Colab:** Entorno de ejecución → Cambiar tipo → GPU **{gpu}**. Ejecuta las celdas en orden
   (o *Ejecutar todo*). Si `/content/jev_llm` no existe, la celda 3 te pedirá `jev_llm_bundle.zip`.
2. **CLI (recomendado para corridas largas):** desde WSL, `tools/colab/launch.sh {cond} <sesion> {gpu} [-p N_PROBLEMS 1 -p RUN_TAG smoke]`
   lanza este notebook en segundo plano en la VM con papermill; `tools/colab/status.sh` lo monitorea y
   `tools/colab/pull.sh` descarga los zips de resultados a medida que se generan.

### Qué produce (en `/content/jev_llm/results/{cond}/<run_name>/`)
`manifest.json` (config, hashes, entorno, GPU, lock de modelos, memoria), `predictions.jsonl`, `metrics.jsonl`,
`candidates.jsonl`, `decisions.jsonl`, `rounds.jsonl`, `memory.jsonl`, `failures.jsonl`, `events.jsonl`,
`warmup.jsonl`, `preflight/`, `logs/run.log`, `logs/vllm.log`, `progress.json`, `pip_freeze.txt`, y zips
`/content/jev_llm/results/<run_name>_checkpoint.zip` (cada `CHECKPOINT_EVERY` casos) y `_final.zip`.
"""),
        md("## 1. Instalación del stack fijado\nNo importes nada antes de esta celda. Tarda unos minutos la primera vez."),
        code(INSTALL, "1 · Instalar vLLM + jevk5 (fijados)"),
        md("## 2. Parámetros\nFormulario editable. `N_PROBLEMS=1` + `RUN_TAG=\"smoke\"` para una prueba de humo."),
        code(params_cell(cond, gpu, series), "2 · Parámetros", tags=["parameters"]),
        md("## 3. Código y datos del experimento"),
        code(BUNDLE, "3 · Cargar bundle (código + datos sin gold)"),
        code(DRIVE, "3b · (Opcional) montar Google Drive"),
        md("## 4. Experimento\nCrea la carpeta de resultados (nombre = condición + RUN_TAG + config_hash) y el logger."),
        code(f"from jevlab.runner import Experiment\nexp = Experiment(**{params_dict})\n"
             "print('resultados en:', exp.dir)", "4 · Crear experimento"),
        md("## 5. Entorno y hardware\nGPU (modelo, UUID, driver), host, versiones, `pip freeze`, sonda de calibración "
           "(solo para análisis de sensibilidad entre VMs) y muestreo NVML en segundo plano."),
        code("exp.capture_environment()", "5 · Capturar entorno"),
        md("## 6. Modelos fijados por SHA\nDescarga los snapshots exactos, verifica hashes de todos los archivos "
           "contra el Hub (y `SHA256SUMS` de JevK5), cuenta parámetros lingüísticos desde cabeceras safetensors."),
        code("exp.prepare_models()\nimport json; print(json.dumps(exp.lock, indent=1, default=str)[:3000])",
             "6 · Snapshots + verificación"),
        md("## 7. Preflight previo a motores (§14)\nParser, lock, plantillas sin pensamiento, longitud de prompts, "
           "semántica de muestreo" + (" y referencia nativa JevK5 (subproceso)." if cond in ("J64", "JSTEP", "JFINAL") else ".")),
        code("exp.preflight_before_engines()", "7 · Preflight (pre)"),
        md("## 8. Motores vLLM\nCada motor corre en su propio proceso; KV cache con tamaño explícito."),
        code("exp.start_engines()", "8 · Arrancar motores"),
        md("## 9. Preflight con motores (§14)\n" + ("Equivalencia del selector vLLM vs nativo, " if cond in ("J64", "JSTEP", "JFINAL") else "")
           + "continuidad de tokens, " + ("auditoría de batching, " if cond in ("J64", "JSTEP", "JFINAL") else "")
           + "estrés de longitud máxima" + (" (incl. entrada de 16k en J)" if cond in ("J64", "JSTEP", "JFINAL") else "") + "."),
        code("exp.preflight_after_engines()\n"
             "import pandas as pd\n"
             "pd.DataFrame([{'check': k, 'ok': v.get('ok'), 'blocking': v.get('blocking'), 'summary': v.get('summary')}\n"
             "              for k, v in exp.preflight_results.items()])", "9 · Preflight (post)"),
        md("## 10. Calentamiento\nProblemas de desarrollo (no test), fuera del cronómetro; se guardan en `warmup.jsonl`."),
        code("exp.warmup()", "10 · Warm-up"),
        md("## 11. Bucle principal (reanudable)\nCada caso registra hora de pared UTC de inicio/fin y `perf_counter` "
           "(T_total, primera salida aceptada, propuestas, selector, prefill, control), tokens, decisiones, memoria "
           "NVML. Si Colab se desconecta, vuelve a ejecutar desde la celda 3: los casos terminados se saltan."),
        code("exp.run()", "11 · Ejecutar"),
        md("## 12. Cierre y descarga"),
        code("zpath = exp.finalize()\n"
             "import collections, json\n"
             "from jevlab.common import read_jsonl\n"
             "preds = read_jsonl(exp.f['predictions'])\n"
             "print('estados:', collections.Counter(p['status'] for p in preds))\n"
             "print('zip final:', zpath)\n"
             "try:\n    from google.colab import files  # descarga directa en la UI\n"
             "    # files.download(zpath)\nexcept Exception:\n    pass", "12 · Finalizar"),
        code("exp.shutdown()", "13 · Liberar GPU"),
        md("### Monitoreo rápido (opcional)\nEjecuta esta celda en cualquier momento para ver progreso y últimas líneas del log."),
        code("import json, pathlib\n"
             f"p = pathlib.Path('/content/jev_llm/results/progress_{cond}.json')\n"
             "print(p.read_text() if p.exists() else 'sin progreso aún')\n"
             "log = exp.dir / 'logs' / 'run.log'\n"
             "print(''.join(log.read_text().splitlines(True)[-25:]) if log.exists() else '')", "Monitor"),
    ]
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": gpu, "machine_shape": "hm"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if series == "v2":
        _v2_paths(nb)
        nb.cells[0].source = (f"# JEV-LLM v2: {desc}\n\nInternal code: `{cond}`. "
                              "Profile: `1COPY-G4-VLLM-GRAPH`; BF16; A100 40 GB in every arm.\n\n"
                              "Protocol: `protocolo_jev_qwen4b_v2.md`. One physical G4 generator, four independently seeded "
                              "requests in hybrid arms. Inference bundle contains inputs only, never gold.\n\n"
                              "Smoke: `SPLIT=dev`, `N_PROBLEMS=1`, `RUN_TAG=smoke`. Pilot: `SPLIT=pilot`, "
                              "`N_PROBLEMS=40`, `RUN_TAG=pilot`; only G_SINGLE, JFINAL and B13. Apply the preregistered "
                              "pilot go/no-go before confirmatory test. No automatic pilot approval.\n\n"
                              "Per-VM latency is descriptive. Do not use the v1 launch scripts: run this v2 notebook "
                              "directly with papermill or the Colab UI. No VM is allocated by the builder.")
    return nb


def _v2_paths(nb):
    for cell in nb.cells:
        cell.source = cell.source.replace("/content/jev_llm/", "/content/jev_llm_v2/")
        cell.source = cell.source.replace('Path("/content/jev_llm")', 'Path("/content/jev_llm_v2")')
        cell.source = cell.source.replace('globals().get("ROOT", "/content/jev_llm")', 'globals().get("ROOT", "/content/jev_llm_v2")')
        cell.source = cell.source.replace('jev_llm_analysis_bundle.zip', 'jev_llm_v2_analysis_bundle.zip')
        cell.source = cell.source.replace('jev_llm_bundle.zip', 'jev_llm_v2_bundle.zip')
        cell.source = cell.source.replace('SHARED4-VLLM-GRAPH', '1COPY-G4-VLLM-GRAPH')
        cell.source = cell.source.replace('"/content/drive/MyDrive/jev_llm_results"', '"/content/drive/MyDrive/jev_llm_v2_results"')
        cell.source = cell.source.replace('if not (ROOT / "src" / "jevlab").exists():',
            'expected_gold = "analysis_bundle" in ZIP.name\n'
            'manifest_path = ROOT / "BUNDLE_MANIFEST.json"\n'
            'existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}\n'
            'if existing.get("series") != "v2" or existing.get("contains_gold") != expected_gold:')
        if 'Wrong bundle series or gold visibility' in cell.source:
            continue
        cell.source = cell.source.replace('bad = [f for f, h in man["files"].items()',
            'if man.get("series") != "v2" or man.get("contains_gold") != expected_gold:\n'
            '    raise SystemExit("Wrong bundle series or gold visibility")\n'
            'if not expected_gold and any((ROOT / "data").rglob("*_gold.jsonl")):\n'
            '    raise SystemExit("Inference root contains gold; use a clean /content/jev_llm_v2 root")\n'
            'bad = [f for f, h in man["files"].items()')


def analysis_notebook(series: str = "v1") -> nbf.NotebookNode:
    if series == "v2-reload":
        nb = analysis_notebook("v2")
        _reload_notebook(nb)
        return nb
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# JEV-LLM · 07_analisis (CPU)

Une los resultados de todas las condiciones, corrige **offline** con el gold (aritmética racional exacta, §11)
y produce la tabla final (§16), comparaciones pareadas con bootstrap estratificado por dominio (10,000
remuestreos, semilla 271828, IC 95% y 99.1667%), diagnóstico JFINAL (oracle@4, mayoría, brecha del selector),
estadísticas del selector y de pasos, categorías de fallo, figuras y `report.md`.

**Entradas:** zips `*_final.zip` / `*_checkpoint.zip` de los notebooks 01-06 (y 08) en `/content/jev_llm/incoming/`
(o ya descomprimidos en `/content/jev_llm/results/`). Necesita el **bundle de análisis** (`jev_llm_analysis_bundle.zip`, con gold).
Corre en CPU (no necesita GPU).
"""),
        code(INSTALL.replace(f'if _v("vllm") != "{VLLM_PIN}": need.append("vllm=={VLLM_PIN}")', "")
              .replace(f'if _v("jevk5") is None or _jevk5_commit() != "{JEVK5_COMMIT}":', "if False:"), "1 · Dependencias ligeras"),
        code('''
RUN_TAG = "main"  # @param {type:"string"}
CONFIG_FILE = "config/experiment.json"  # @param {type:"string"}
DATA_SUBDIR = "data"  # @param {type:"string"}
ROOT = "/content/jev_llm"  # @param {type:"string"}
SPLIT = "test"  # @param ["test", "pilot", "dev"]
# ^ analiza solo corridas con este RUN_TAG ("smoke" para la prueba de humo; "" = todas)
N_BOOT = 10000  # @param {type:"integer"}
RETOKENIZE_WITH_G = True  # @param {type:"boolean"}
PERSIST_TO_DRIVE = False  # @param {type:"boolean"}
DRIVE_DIR = "/content/drive/MyDrive/jev_llm_results"  # @param {type:"string"}
''', "2 · Parámetros", tags=["parameters"]),
        code(BUNDLE.replace("jev_llm_bundle.zip", "jev_llm_analysis_bundle.zip"), "3 · Cargar bundle de análisis (con gold)"),
        code(DRIVE + '''
import glob, zipfile, shutil, pathlib
inc = ROOT / "incoming"; inc.mkdir(exist_ok=True)
srcs = glob.glob(str(inc / "*.zip")) + glob.glob(str(ROOT / "results" / "*_final.zip"))
if PERSIST_TO_DRIVE: srcs += glob.glob(DRIVE_DIR + "/*.zip")
res = ROOT / "results"; res.mkdir(exist_ok=True)
for z in sorted(srcs):
    zipfile.ZipFile(z).extractall(res)
    print("extraído", z)
''', "4 · Reunir resultados"),
        code('''
from jevlab.analysis import run_analysis
import json
cfg = json.loads((ROOT / CONFIG_FILE).read_text())
tok = None
if RETOKENIZE_WITH_G:
    try:
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tj = hf_hub_download(cfg["models"]["G"]["repo"], "tokenizer.json", revision=cfg["models"]["G"]["revision"])
        tok = PreTrainedTokenizerFast(tokenizer_file=tj)
    except Exception as e:
        print("retokenización desactivada:", e)
OUT = str(ROOT / ("analysis_" + (RUN_TAG or "all")))
A = run_analysis(str(ROOT / "results"), str(ROOT / DATA_SUBDIR / f"{SPLIT}_gold.jsonl"), OUT,
                 RUN_TAG or None, N_BOOT, tok)
A["table"]
''', "5 · Análisis"),
        code('''
from IPython.display import Markdown, Image, display
import glob
display(Markdown(A["report"]))
for f in sorted(glob.glob(OUT + "/figures/*.png")):
    display(Image(f))
''', "6 · Informe y figuras"),
        code('''
import shutil
z = shutil.make_archive(OUT, "zip", OUT)
print("análisis empaquetado:", z)
if PERSIST_TO_DRIVE:
    shutil.copy2(z, DRIVE_DIR)
''', "7 · Empaquetar"),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if series == "v2":
        _v2_paths(nb)
        for cell in nb.cells:
            cell.source = cell.source.replace('CONFIG_FILE = "config/experiment.json"', 'CONFIG_FILE = "config/experiment_v2.json"')
            cell.source = cell.source.replace('DATA_SUBDIR = "data"', 'DATA_SUBDIR = "data/v2"')
            cell.source = cell.source.replace('ROOT = "/content/jev_llm"', 'ROOT = "/content/jev_llm_v2"')
        nb.cells[0].source = ("# JEV-LLM v2: Offline analysis (CPU)\n\nLoad `jev_llm_v2_analysis_bundle.zip` "
                              "and immutable v2 prediction zips into `/content/jev_llm_v2/incoming/`. "
                              "Gold is read from `data/v2/{SPLIT}_gold.jsonl`, never from v1. "
                              "The generator tokenizer is selected from the pinned v2 config.\n\n"
                              "Protocol: `protocolo_jev_qwen4b_v2.md`. H1 is JFINAL minus B13 accuracy; "
                              "latency is secondary/descriptive. The analysis implementation is maintained separately; "
                              "verify its v2 statistical support before publishing confirmatory claims.")
    return nb


def latency_notebook(series: str = "v1") -> nbf.NotebookNode:
    if series == "v2-reload":
        nb = latency_notebook("v2")
        _reload_notebook(nb)
        for cell in nb.cells:
            if "parameters" in cell.metadata.get("tags", []):
                cell.source += '\nLATENCY_SWAP_MODE = "reload"  # fixed: approved same-GPU unload/reload amendment\n'
            if "exp = LatencyStudy" in cell.source:
                cell.source = cell.source.replace("dict(CONDITION=CONDITION,", "dict(LATENCY_SWAP_MODE=LATENCY_SWAP_MODE, CONDITION=CONDITION,")
            cell.source = cell.source.replace("Sleep level 1 restores unchanged weights; level 2 is not supported.",
                "Unload ALL engines, confirm process exit, then load only B or G4+J on the SAME A100 40 GB. "
                "No sleep or CPU offload. J closes for G_SINGLE and is rebuilt/calibrated/probed before hybrids. "
                "All lifecycle work is outside T_total and recorded in swaps.jsonl. Default plan: 192 measured cases.")
            cell.source = cell.source.replace("7 · Motores (B dormido, G+J activos)", "7 · Load G4+J only (reload)")
        return nb
    nb = condition_notebook("J64", series)
    params_extra = '''
# ======================= PARÁMETROS (estudio de latencia, misma VM) =======================
CONDITION = "LATENCY"
CONFIG_FILE = "config/experiment.json"  # @param {type:"string"}
DATA_SUBDIR = "data"  # @param {type:"string"}
ROOT = "/content/jev_llm"
RUN_TAG = "main"  # @param {type:"string"}
LAT_CONDITIONS = "G_SINGLE,J64,JSTEP,JFINAL,B13,B13_GREEDY"  # @param {type:"string"}
LAT_N_DEV = 20  # @param {type:"integer"}
# ^ problemas de desarrollo incluidos (0-20); nunca se usa el test
LAT_SYNTH_PER_BAND = 3  # @param {type:"integer"}
# ^ prompts sintéticos por banda de longitud de salida (short/medium/long/xlong)
LAT_BLOCK_SIZE = 8  # @param {type:"integer"}
SEEDS = "17"  # @param {type:"string"}
SLEEP_LEVEL = 1  # @param [1, 2]
SLEEP_J_FOR_G_SINGLE = True  # @param {type:"boolean"}
TEMPERATURE = 0.7  # @param {type:"number"}
TOP_P = 0.9  # @param {type:"number"}
TOP_K = 0  # @param {type:"integer"}
REPETITION_PENALTY = 1.0
N_CANDIDATES = 4
MAX_OUTPUT_TOKENS = 1024  # @param {type:"integer"}
CANDIDATE_BUDGET = 4096
J64_BLOCK = 64
JSTEP_STEP = 128
JSTEP_MAX_ROUNDS = 64
GB_CONTEXT = 4096
J_MAX_INPUT = 16384
TIMEOUT_S = 600  # @param {type:"integer"}
ENFORCE_EAGER = False  # @param {type:"boolean"}
KV_CACHE_GB_G = 2.0  # @param {type:"number"}
KV_CACHE_GB_J = 4.0  # @param {type:"number"}
KV_CACHE_GB_B = 3.0  # @param {type:"number"}
J_READOUT = "logprob_token_ids"
WARMUP_N = 2  # @param {type:"integer"}
HASH_WEIGHTS = True  # @param {type:"boolean"}
REQUIRED_GPU = "A100"  # @param ["A100", "H100", "any"]
PERSIST_TO_DRIVE = False  # @param {type:"boolean"}
LOG_LEVEL = "INFO"  # @param ["INFO", "DEBUG"]
'''
    if series == "v2":
        params_extra = params_extra.replace('config/experiment.json', 'config/experiment_v2.json')
        params_extra = params_extra.replace('DATA_SUBDIR = "data"', 'DATA_SUBDIR = "data/v2"')
        params_extra = params_extra.replace('ROOT = "/content/jev_llm"', 'ROOT = "/content/jev_llm_v2"')
        params_extra = params_extra.replace('KV_CACHE_GB_G = 2.0', 'KV_CACHE_GB_G = 3.0')
        params_extra = params_extra.replace('SLEEP_LEVEL = 1  # @param [1, 2]', 'SLEEP_LEVEL = 1  # fixed: level 2 discards model weights')
    names = ["CONDITION", "CONFIG_FILE", "DATA_SUBDIR", "ROOT", "RUN_TAG", "LAT_CONDITIONS", "LAT_N_DEV", "LAT_SYNTH_PER_BAND", "LAT_BLOCK_SIZE",
             "SEEDS", "SLEEP_LEVEL", "SLEEP_J_FOR_G_SINGLE", "TEMPERATURE", "TOP_P", "TOP_K",
             "REPETITION_PENALTY", "N_CANDIDATES", "MAX_OUTPUT_TOKENS", "CANDIDATE_BUDGET", "J64_BLOCK",
             "JSTEP_STEP", "JSTEP_MAX_ROUNDS", "GB_CONTEXT", "J_MAX_INPUT", "TIMEOUT_S", "ENFORCE_EAGER",
             "KV_CACHE_GB_G", "KV_CACHE_GB_J", "KV_CACHE_GB_B", "J_READOUT", "WARMUP_N", "HASH_WEIGHTS",
             "REQUIRED_GPU", "PERSIST_TO_DRIVE", "LOG_LEVEL"]
    pd_ = "dict(" + ", ".join(f"{p}={p}" for p in names) + ")"
    nb.cells = [
        md("""
# JEV-LLM · 08_estudio_latencia (una sola A100)

Comparación **confirmatoria de latencia** (enmienda v3.1 §A2): todas las condiciones en la **misma VM/GPU**,
en bloques contrabalanceados (B antes/después de los híbridos alternando; orden rotado de G_SINGLE/J64/JSTEP/JFINAL),
con un conjunto de preguntas **distinto del test**: los 20 problemas de desarrollo + prompts sintéticos con bandas
controladas de longitud de salida (n = 5/15/30/60 líneas "k squared is k*k" + "FINAL: n"; todas caben en el
tope de 64 rondas de JSTEP).
El cambio entre B y G+J usa el *sleep mode* de vLLM **fuera del cronómetro**. Cada ejecución guarda hora de pared UTC
de inicio/fin y los tiempos `perf_counter` con el mismo esquema que los notebooks 01-06.

Salida: `/content/jev_llm/results/LATENCY/<run_name>/latency_metrics.jsonl` (+ `swaps.jsonl`, `latency_plan.json`, logs, zips).
El notebook 07 lo detecta y calcula speedups con bootstrap.
"""),
        nb.cells[1], nb.cells[2],
        md("## 2. Parámetros"),
        code(params_extra, "2 · Parámetros", tags=["parameters"]),
        nb.cells[5], nb.cells[6], nb.cells[7],
        code(f"from jevlab.latency_study import LatencyStudy\nexp = LatencyStudy(**{pd_})\nprint(exp.dir)",
             "4 · Crear estudio"),
        code("exp.capture_environment()", "5 · Entorno"),
        code("exp.prepare_models()", "6 · Snapshots G, J, B + verificación"),
        code("exp.start_engines()", "7 · Motores (B dormido, G+J activos)"),
        code("exp.run()", "8 · Bloques contrabalanceados"),
        code("print(exp.finalize())\nexp.shutdown()", "9 · Finalizar"),
    ]
    if series == "v2":
        _v2_paths(nb)
        nb.cells[0].source = ("# JEV-LLM v2: Same-A100 latency study (secondary/descriptive)\n\n"
                              "Default: 20 development problems + 12 synthetics, no test or gold. "
                              "Smoke: `LAT_N_DEV=2`, `LAT_SYNTH_PER_BAND=1` gives 2 dev + 4 synthetics; "
                              "`N_PROBLEMS` does not control this study. Use `RUN_TAG=smoke`.\n\n"
                              "BF16; one G4 engine, four requests; swaps outside the timer; A100 40 GB. "
                              "Blocking tokenizer/native selector/stress checks run before measurement. "
                              "Sleep level 1 restores unchanged weights; level 2 is not supported.")
        index = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.endswith('exp.start_engines()'))
        nb.cells.insert(index, code('exp.preflight_before_engines()', '6b · Preflight before engines'))
        nb.cells.insert(index + 2, code('exp.preflight_after_engines()', '7b · Preflight with engines'))
    return nb


def _reload_notebook(nb):
    for cell in nb.cells:
        cell.source = cell.source.replace("config/experiment_v2.json", "config/experiment_v2_reload.json")
        cell.source = cell.source.replace('get("series") != "v2"', 'get("series") != "v2-reload"')
    nb.cells[0].source += ("\n\nApproved reload amendment (jevlab 0.2.1): `config/experiment_v2_reload.json`. "
                           "Old config, frozen 0.2.0 bundles and previous smoke outputs are unchanged. "
                           "Use the bundle from `dist/v2-reload/` (same basename); do not use the frozen v2 bundle. "
                           "Runtime ROOT remains `/content/jev_llm_v2`, data remains `data/v2`, and the amended hash prevents old-run resume.")


def main():
    parser = argparse.ArgumentParser(description="Build isolated notebooks; existing series is never rebuilt by --series v2")
    parser.add_argument("--series", choices=("v1", "v2", "v2-reload"), default="v1")
    args = parser.parse_args()
    out = NB / args.series if args.series != "v1" else NB
    out.mkdir(parents=True, exist_ok=True)
    for cond, (fname, *_rest) in COND_INFO.items():
        nbf.write(condition_notebook(cond, args.series), out / f"{fname}.ipynb")
    nbf.write(analysis_notebook(args.series), out / "07_analisis.ipynb")
    nbf.write(latency_notebook(args.series), out / "08_estudio_latencia.ipynb")
    for f in sorted(out.glob("*.ipynb")):
        nbf.validate(nbf.read(f, as_version=4))
        print("ok", f.name)


if __name__ == "__main__":
    main()
