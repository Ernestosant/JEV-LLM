# JEV-LLM: documentación

Este proyecto es una pequeña prueba de concepto, no una demostración general de superioridad de un modelo o del efecto del tamaño. La ronda final v3 usa **una copia física de Qwen3.5-4B**, cuatro propuestas en batch y selección por **JevK5-9B**, además de controles independientes.

**Estado actual: experimento v3 y análisis completados.** Se verificaron 6.000 casos de calidad sobre 500 problemas y 1.800 mediciones de latencia sobre 100 problemas. Consumo acumulado: 15,76 horas A100, dentro de 25 horas; todos los recursos liberados.

## Documentación vigente

| Documento | Contenido |
|---|---|
| [13_results_v3.md](13_results_v3.md) | Resultados finales, metodología, controles y alcance de las conclusiones |
| [14_limitations_and_future_work.md](14_limitations_and_future_work.md) | Gaps de comparabilidad, validación pendiente y prioridades de investigación |
| [../public_results/v3/README.md](../public_results/v3/README.md) | Selección pública de tablas y comparaciones agregadas |
| [../protocolo_jev_qwen4b_v3.md](../protocolo_jev_qwen4b_v3.md) | Protocolo congelado de la ronda final |
| [12_implementation_v3.md](12_implementation_v3.md) | Fotografía histórica de preparación, conservada con una nota de cierre |

`JFINAL` mejoró la calidad frente al baseline personalizado ~13B greedy, pero no confirmó ventaja de velocidad ni superioridad conjunta. Tampoco se confirmó ventaja del selector frente al voto sobre las mismas propuestas. Consultar los informes vigentes antes de interpretar números de rondas anteriores.

## Documentación histórica

Los documentos siguientes describen distintas etapas del prototipo. Sus modelos, tamaños de muestra, comandos y tareas pendientes **no representan el estado final v3**. Se conservan para hacer visibles las decisiones y no reescribir retroactivamente la historia.

| Documento | Contenido |
|---|---|
| [protocolo_v3.1_enmienda.md](protocolo_v3.1_enmienda.md) | Enmienda formal: SHARED4-VLLM-GRAPH, notebooks por condición, selector en vLLM, dataset de benchmarks |
| [01_arquitectura.md](01_arquitectura.md) | Estructura del repo, paquete `jevlab`, flujo de un notebook, decisiones |
| [02_dataset.md](02_dataset.md) | Fuentes (SHA), diseño 5×(8/8/4), filtros, archivos, limitaciones |
| [03_runbook_colab.md](03_runbook_colab.md) | Cómo ejecutar (CLI vía WSL o UI), GPUs por notebook, monitoreo, descarga, análisis |
| [04_artefactos.md](04_artefactos.md) | Esquema de todos los archivos de resultados y cómo diagnosticar fallos |
| [05_preflight_validacion.md](05_preflight_validacion.md) | Validaciones del §14 integradas y sus resultados en el smoke |
| [06_desviaciones_y_cambios.md](06_desviaciones_y_cambios.md) | Desviaciones declaradas y registro de cambios/bugs |
| [08_session_lifecycle.md](08_session_lifecycle.md) | Keep-alive rapido y cierre confirmado contra el servidor |
| [09_results_minimal.md](09_results_minimal.md) | Matriz completa: 600 predicciones, 192 mediciones de latencia y auditoria independiente |
| [../protocolo_jev_qwen4b_v2.md](../protocolo_jev_qwen4b_v2.md) | **Serie v2 completada:** una copia de Qwen3.5-4B, test nuevo, piloto y enmienda autorizada de recarga por fase |
| [10_execution_v2.md](10_execution_v2.md) | Serie histórica 0.2.1: continuidad Colab, coordinador, recuperación e incidentes |
| [11_results_v2.md](11_results_v2.md) | Resultados v2 completos: 600 predicciones, 192 latencias, auditoria independiente y conclusion de H1 |
| [../protocolo_jev_qwen4b_v3.md](../protocolo_jev_qwen4b_v3.md) | Tercera ronda: 500 test, 50 piloto, control oficial 9B y comparacion principal contra el modelo grande greedy |
| [12_implementation_v3.md](12_implementation_v3.md) | Preparación histórica v3 anterior a la ejecución final |
| [07_smoke_test.md](07_smoke_test.md) | Prueba de humo (1 problema por notebook): resultados, tiempos, costos, hallazgos |

## Comandos históricos

Los ejemplos de esta sección corresponden a las primeras rondas. No son un procedimiento automático de reproducción del repositorio público ni autorizan consumo de Colab.

```powershell
python -m pytest tests -q
python tools\build_notebooks.py; python tools\make_bundle.py
```
```bash
# WSL
tools/colab/launch_seq.sh jev-a100 A100 "J64 JSTEP JFINAL B13 B13_GREEDY"   # 5 condiciones en una A100
tools/colab/launch.sh G_SINGLE jev-l4 L4                                     # control en L4
tools/colab/launch_seq.sh jev-lat A100 "LATENCY"                             # latencia confirmatoria
tools/colab/status.sh jev-a100 J64 ; tools/colab/pull.sh jev-a100 ; tools/colab/stop.sh jev-a100
```
Luego `notebooks/07_analisis.ipynb` con `dist/jev_llm_analysis_bundle.zip` y los zips descargados.

## Pendientes históricos de la primera etapa

Estas anotaciones no indican tareas abiertas del experimento final. Los trabajos futuros actuales están en [14_limitations_and_future_work.md](14_limitations_and_future_work.md).
* Completar `data/review_sheet.csv` (2º revisor) y la revisión ciega de `data/blind_review_ids.json`.
* Perfil minimo ya ejecutado: semilla 17 + greedy. Una ampliacion posterior debe declararse como nueva replica o analisis exploratorio, no seleccionar semillas por resultados favorables.
