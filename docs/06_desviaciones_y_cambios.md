# Desviaciones del protocolo y registro de cambios

## Desviaciones declaradas (detalle en `protocolo_v3.1_enmienda.md`)

| # | Protocolo v3 | Implementación | Impacto / mitigación |
|---|---|---|---|
| D1 | Perfil principal R4-BF16-EAGER (4 copias físicas de G) | SHARED4-VLLM-GRAPH (1 copia, 4 peticiones en batch) | `4·P_G+P_J` sólo nominal; VRAM real reportada; chequeo R4 de memoria opcional |
| D2 | Contrabalanceo por bloques en una sesión; no mezclar latencias de GPUs distintas | un notebook por condición en VMs distintas | latencias 01–06 descriptivas; latencia confirmatoria en notebook 08 (una VM, contrabalanceado) |
| D3 | `JevK5.decide` (transformers, `graphs=False`) | misma lectura de logits vía vLLM | equivalencia bloqueante contra el runtime nativo en cada notebook híbrido |
| D4 | Eager con caché; clonado de estado KV/DeltaNet | vLLM con CUDA graphs; re-prefill de `prompt+IDs aceptados` por ronda | test de continuidad y eager-vs-graph; relecturas contabilizadas |
| D5 | 100 problemas originales, dificultad por revisión humana, 2º revisor | muestreo de GSM8K/MATH/AMC/AIME; dificultad heurística; `reviewed=false` | contaminación declarada; hoja de revisión y 20 IDs ciegos preregistrados |
| D6 | `max_memory_allocated/reserved` de PyTorch | log de vLLM + KV explícito + NVML 10 Hz | "reservada" vs "requerida" diferenciadas en el informe |
| D7 | Sobreproducción sólo dentro del token terminal | abort tras la línea FINAL en el bucle por pasos | tokens computados y no entregados tras el abort no observables |
| D8 | — | respuestas ≥ 1000, dinero y porcentaje excluidas del dataset | evita sesgo de formato distinto por modelo |

## Registro de cambios (orden cronológico)

| Fecha (UTC) | Versión `jevlab` | Cambio | Motivo | Condiciones afectadas |
|---|---|---|---|---|
| 2026-09-28 01:xx | — | Enmienda v3.1 redactada y congelada antes del dataset | revisión adversarial del plan | — |
| 2026-09-28 02:48 | 0.1.0 | Subprocesos (`hwprobe`, `native_ref`) con `PYTHONPATH` del bundle | `ModuleNotFoundError: jevlab` en la VM | preflight |
| 2026-09-28 02:52 | 0.1.0 | Instalación desinstala `torchaudio`/`torchcodec` de Colab | torchaudio cu128 vs torch cu130 rompía `import transformers` | todas |
| 2026-09-28 03:43 | 0.1.0 | Configuración de logging de vLLM antes de cualquier `import vllm`; `VLLM_WORKER_MULTIPROC_METHOD=spawn` | error de arranque del motor sin causa visible (logs del subproceso perdidos) | todas |
| 2026-09-28 04:20 | 0.1.0 | `analysis.py` sin dependencia de `tabulate` | fallo del informe en CPU local | análisis |
| 2026-09-28 04:35 | **0.1.1** | **EOS eliminado de `token_ids` entregados** (se cuenta como token muestreado vía `eos`) | vLLM entrega `<|im_end|>` dentro de `token_ids`: aparecía en el texto público (`FINAL: 36<|im_end|>` → formato inválido) y en los candidatos que veía J | **todas** → se repitieron todas las condiciones del smoke con 0.1.1 |
| 2026-09-28 04:35 | 0.1.1 | `code_version` dentro del `config_hash` | una reparación nunca reanuda corridas previas (§11) | todas |
| 2026-09-28 04:40 | — | Herramientas CLI: reintentos (`_cexec.sh`), detección de zombies, `kill_gpu.sh`, limpieza de GPU entre notebooks | conexión frágil del CLI; procesos `VLLM::EngineCore` huérfanos retenían 26 GB | operación |
| 2026-09-28 05:35 | **0.1.2** | Prompts sintéticos del notebook 08: 5/15/30/60 líneas "k squared is k*k" (antes 10/40/120/300 enteros) | la banda de 300 líneas excedía el tope de 64 rondas de JSTEP (`max_rounds`), censurando su latencia | sólo notebook 08 |
| 2026-09-28 04:1x | — | CLI de Colab actualizado 0.7.0 → 0.7.4 | 0.7.0 marcó como perdida una sesión viva al expirar el token (~1 h) y dejó una VM huérfana (liberada manualmente vía API `unassign`) | operación |
