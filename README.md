# JEV-LLM

**Una pequeña prueba de concepto de selección entre soluciones generadas por un LLM.**

JEV-LLM explora si un generador Qwen3.5-4B, con una sola copia física y cuatro solicitudes en batch, puede mejorar sus respuestas matemáticas cuando JevK5-9B selecciona entre las cuatro soluciones completas mediante logits de opciones. El experimento utiliza vLLM, BF16 y CUDA graphs sobre Google Colab con una A100 de 40 GB.

Este proyecto **no entrena un nuevo modelo, no es un benchmark general y no demuestra que los modelos pequeños sean mejores por su tamaño**. Es un prototipo experimental acotado, con resultados y limitaciones publicados para facilitar su evaluación crítica.

## Resultados de la prueba de concepto

La ronda v3 terminó con **500 problemas de test, 6.000 casos de calidad y 1.800 mediciones de latencia**. Un piloto separado de 50 problemas produjo 600 casos técnicos; no se mezcló con el test. El consumo acumulado fue de **15,76 horas A100**, dentro del límite original de 25 horas, y todos los recursos fueron liberados.

| Condición | Casos de calidad | Exactitud estricta |
|---|---:|---:|
| Qwen3.5-4B, una solución con muestreo | 1.500 | 91,00 % |
| Qwen3.5-4B, una solución greedy | 500 | 92,20 % |
| 4B + JevK5-9B, selección final (`JFINAL`) | 1.500 | 94,73 % |
| Baseline personalizado de aproximadamente 13B, muestreo | 1.500 | 84,20 % |
| Baseline personalizado de aproximadamente 13B, greedy | 500 | 84,00 % |
| Qwen3.5-9B oficial, greedy | 500 | 89,20 % |

El baseline ~13B es `DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking`, una mezcla y expansión de checkpoints derivados de 9B, no un Qwen oficial de 13B comparable únicamente por tamaño. Las revisiones exactas están en [`config/experiment_v3.json`](config/experiment_v3.json).

### Conclusiones acotadas

- **Calidad frente al baseline principal ~13B greedy:** `JFINAL` obtuvo +10,73 puntos porcentuales; IC coprincipal del 97,5 %, ajustado por Bonferroni: [7,53; 14,13]. La superioridad en calidad se sostiene para este conjunto y protocolo.
- **Velocidad:** el speedup geométrico pareado fue 1,002x, con IC del 97,5 % [0,884; 1,126]. No se confirmó superioridad de latencia ni la hipótesis conjunta de mayor calidad y velocidad.
- **Valor específico del selector:** el voto por respuestas normalizadas sobre las mismas cuatro propuestas obtuvo 94,60 %. La diferencia de `JFINAL` fue solamente +0,13 puntos y no fue significativa tras Holm. No se demostró ventaja del crítico frente a ese control simple.
- **4B frente al 9B oficial:** la diferencia puntual fue +3 puntos; el contraste secundario no superó Holm (`p` ajustado aproximadamente 0,088). No demuestra superioridad general del 4B.

Las semillas no se tratan como problemas independientes. Los intervalos utilizan bootstrap pareado por problema, estratificado por dominio, con 10.000 remuestreos. La latencia utiliza 100 problemas, nueve mediciones por sistema y problema en la misma GPU; carga, recargas y warmup quedan fuera del cronómetro de caso. Los truncamientos y formatos inválidos cuentan como incorrectos.

## Limitaciones importantes

- Los modelos difieren en entrenamiento y ajustes, no solamente en parámetros. La comparación no aísla el efecto del tamaño.
- Todos se evaluaron sin thinking, con 2.048 tokens de salida y un formato `FINAL` estricto. Es una evaluación bajo presupuesto y no una medición de su máxima capacidad según las recomendaciones del fabricante.
- El test tiene 394 problemas intrínsecamente fáciles, 96 medios y 10 difíciles. Las cuotas equilibran **nivel de fuente**, no dificultad intrínseca. La revisión fue agéntica, no humana.
- No se descartó contaminación de entrenamiento ni duplicación semántica con los corpus públicos de origen.
- La comparación eager frente a CUDA graphs se ejecutó para el generador en tres prefijos cortos de desarrollo; para el baseline ~13B y el 9B oficial fue omitida por memoria. Las pruebas de estrés y continuidad no sustituyen una equivalencia completa de precisión contra otro backend.
- Hubo recuperaciones de recolección y auditoría documentadas, sin repetir inferencias completadas. La recuperación de `JFINAL` conservó bytes originales pero no un hash de transporte remoto persistido; la auditoría de latencia empleó una proyección exclusivamente en memoria de la prueba preservada de la fase híbrida.

Estos gaps se explican en [limitaciones y trabajo futuro](docs/14_limitations_and_future_work.md). La prioridad es validar los backends, comparar tamaños oficiales de la misma familia, estudiar modos y presupuestos separadamente y ampliar el conjunto con referencias independientes antes de formular afirmaciones fuertes.

## Documentación y datos publicados

- [Resultados v3 y metodología](docs/13_results_v3.md).
- [Gaps de comparación y trabajo futuro](docs/14_limitations_and_future_work.md).
- [Índice de documentación e historia del prototipo](docs/README.md).
- [Resultados agregados públicos](public_results/v3/README.md).
- [Protocolo de la ronda v3](protocolo_jev_qwen4b_v3.md).

El repositorio público incluye código, configuración científica, prompts, pruebas, notebooks **sin ejecutar** y resultados agregados. No incluye credenciales, archivos de variables de entorno, sesiones de Colab, logs, dumps de entorno, pesos, datasets congelados, respuestas de referencia ni paquetes privados de revisión. Los artefactos completos locales y su ZIP de entrega no se publican automáticamente.

Los modelos y las fuentes de datos tienen licencias independientes, algunas no comerciales o de procedencia pendiente de aclaración. Publicar este prototipo no concede derechos sobre esos materiales ni sustituye su revisión de licencias.

## Estructura

```text
src/jevlab/               Runtime histórico y namespace v3
config/                  Revisiones, presupuestos y parámetros científicos
prompts/                 Instrucciones de generación y selección
notebooks/v3/postpilot/   Notebooks fuente de la ronda final, sin salidas
tools/                   Preparación, auditoría y operación experimental
data/*.py                Código de preparación; no contiene el dataset privado
tests/                   Pruebas locales y fixtures sintéticos
docs/                    Resultados, limitaciones e historia
public_results/v3/       Tablas y comparaciones agregadas, sin datos operativos
```

## Uso y reproducibilidad

La configuración científica es [`config/experiment_v3.json`](config/experiment_v3.json); el runtime v3 es [`src/jevlab/v3/`](src/jevlab/v3/). Los notebooks son generados por `tools/build_notebooks_v3.py` y requieren el stack indicado en sus celdas de instalación.

El clon público **no reproduce por sí solo el test congelado**: la ejecución original necesitó los archivos de datos sellados, sus revisiones y autorizaciones locales, que no se redistribuyen aquí. Los constructores y gates fallan cerrado si falta esa evidencia. No deben inventarse sellos, permisos ni artefactos para sortearlos.

Las pruebas CPU utilizan fixtures sintéticos. Necesitan las dependencias importadas por la suite, entre ellas `pytest`, `numpy`, `pandas`, `sympy`, `matplotlib`, `nbformat` y `papermill`. Algunas pruebas requieren Windows/WSL o archivos locales de evidencia que no se distribuyen; no se declara una suite global portable y totalmente validada.

```bash
python -m pytest tests/test_parsing_seeds.py -q
```

Configurar autenticación o lanzar Colab es una acción separada que puede consumir recursos facturados. No hay credenciales incorporadas ni permiso implícito para asignar una GPU al abrir un notebook o clonar el repositorio.
