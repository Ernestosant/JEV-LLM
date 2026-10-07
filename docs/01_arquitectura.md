# Arquitectura

```
Jev-LLM/
├── protocolo_jev_qwen_v3.md        protocolo original (no se edita)
├── docs/                           documentación (esta carpeta)
├── config/experiment.json          modelos+SHA, muestreo, límites, semillas, estadística
├── prompts/                        generador.txt, criterio_paso.txt, criterio_final.txt (texto del protocolo)
├── data/                           build_dataset.py + dataset congelado + raw/ (fuentes descargadas)
├── src/jevlab/                     paquete del experimento (se sube a Colab dentro del bundle)
├── notebooks/                      01–08 generados por tools/build_notebooks.py
├── tests/                          pytest en CPU (parser, semillas, dataset, finalización de candidatos)
├── tools/                          build_notebooks.py, make_bundle.py, colab/*.sh (CLI vía WSL)
├── dist/                           jev_llm_bundle.zip (sin gold) y jev_llm_analysis_bundle.zip (con gold)
└── results/                        artefactos descargados (smoke/, colab/<sesión>/)
```

## Paquete `jevlab` (mapa a los contratos del §15)

| Módulo | Responsabilidad | Contrato §15 |
|---|---|---|
| `seeds.py` | semillas SHA-256 LE mod 2⁶³−1; permutaciones `|option_order` | §5.2 |
| `parsing.py` | gramática estricta de respuesta, `\boxed{}` de MATH, terminalidad FINAL/EOS, extracción offline | §7.4, §11 |
| `prompts.py` | plantillas de G/B (`enable_thinking=False`), prompt nativo de JevK5 | §5.1, §6 |
| `models.py` | snapshots por SHA, verificación de hashes, inventario de parámetros, conjuntos de tokens (EOS, `\n`) | `load_models` |
| `engines.py` | motor vLLM (subproceso) + bucle por pasos con abort en FINAL, deadline, TTFT; logging de vLLM a archivo | `generate_candidates` |
| `selector.py` | lectura de logits de letras vía vLLM, calibración T, argmax, permutación | `select_candidate` |
| `algorithms.py` | G_SINGLE/B13/B13_GREEDY, J64, JSTEP, JFINAL; presupuestos §7.4; flags de pasos; sobreproducción | `commit_candidate`, `run_case` |
| `runner.py` | orquestación por notebook: entorno, lock, preflight, motores, warm-up, bucle reanudable, artefactos, zips | `run_case` + persistencia |
| `preflight.py` | validaciones §14 (bloqueantes y diagnósticas), caché por stack | §14 |
| `hw.py`, `hwprobe.py` | GPU/host/paquetes, `pip freeze`, muestreo NVML, sonda de calibración (subproceso) | §9.1, §10.3 |
| `native_ref.py` | runtime JevK5 publicado en subproceso (referencia de equivalencia) | §14.5 |
| `evaluate.py` | corrección offline con gold (racionales exactos) | `evaluate_offline` |
| `analysis.py` | tablas, bootstrap pareado estratificado, diagnóstico JFINAL, figuras, informe | `analyze_paired` |
| `latency_study.py` | notebook 08: todas las condiciones en una VM con bloques contrabalanceados y sleep mode | A2 |

## Flujo de un notebook de condición

1. Instalar stack fijado (vLLM 0.30.0 → torch 2.13/cu130, transformers 5.16.1; `jevk5` v0.3.0 sin dependencias;
   se desinstalan torchaudio/torchcodec de Colab, incompatibles).
2. Parámetros (formulario) → `Experiment(**params)` → carpeta `results/<COND>/<COND>_<RUN_TAG>_<hash8>/`.
3. Entorno (GPU, host, paquetes, sonda) · 4. Snapshots + verificación de hashes · 5. Preflight previo (incluye
   referencia nativa J en subproceso) · 6. Motores vLLM (J primero, luego G o B) · 7. Preflight con motores ·
   8. Warm-up con dev · 9. Bucle principal reanudable · 10. Manifiesto final + zips.

## Decisiones clave

* **Una copia de G, 4 peticiones en batch** (SHARED4): mínimo costo para generar 4 candidatos en paralelo.
* **Motores en subprocesos** (`enable_multiprocessing=True`): permite G+J en la misma GPU y que el notebook nunca
  cree contexto CUDA. `kv_cache_memory_bytes` explícito + `gpu_memory_utilization=0.05` (sólo para pasar el chequeo
  de memoria libre del segundo motor).
* **IDs, no texto**: la continuación siempre reenvía IDs aceptados; nunca se re-tokeniza el prefijo.
* **Predicciones sin gold**: la corrección ocurre en `07_analisis` con el bundle de análisis.
* **Reanudación**: clave `sha256(config_hash|problem_id|seed|condition|profile)`; una predicción se escribe al final
  del caso (fsync), así una desconexión nunca deja un caso a medias marcado como hecho.
