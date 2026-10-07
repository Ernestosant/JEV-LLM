# Resultados Del Perfil Minimo

Ejecuciones completadas 2026-09-29/30: 100 problemas congelados, semilla 17, seis
condiciones (600 ejecuciones). Modelos/prompts/presupuestos sin cambios, codigo 0.1.2,
perfil SHARED4-VLLM-GRAPH de la enmienda v3.1. No es una ejecucion R4 de cuatro copias
fisicas. Revision humana del dataset y del razonamiento todavia pendiente.

## Calidad

Correccion offline independiente de la inferencia: exactamente una linea FINAL,
gramatica numerica estricta y comparacion racional exacta. Truncamientos, EOS sin
FINAL y limites de rondas cuentan como incorrectos; ninguno fue excluido.

| Condicion | Aciertos/100 | IC95 de exactitud | Formato valido |
|---|---:|---|---:|
| G_SINGLE | 7 | [3, 12] % | 18 % |
| B13 | 73 | [64, 81] % | 94 % |
| B13_GREEDY | 81 | [73, 88] % | 91 % |
| J64 | 17 | [10, 24] % | 34 % |
| JSTEP | 22 | [14, 30] % | 34 % |
| JFINAL | 22 | [15, 30] % | 36 % |

Bootstrap pareado por problema, estratificado por dominio, 10,000 remuestreos,
semilla 271828. Diferencias de exactitud frente a B13, con IC99.1667 ajustado:

- J64: -56 puntos porcentuales; intervalo [-69, -42].
- JSTEP: -51 puntos porcentuales; intervalo [-64, -37].
- JFINAL: -51 puntos porcentuales; intervalo [-64, -38].

Estos resultados no respaldan mayor exactitud de los hibridos frente a B13. Las
mejoras puntuales frente a G_SINGLE no eliminan esa brecha. B13_GREEDY es un control
secundario; no se escogio el mejor baseline por problema.

## Latencia

Notebook 08: 32 entradas distintas del test (20 dev y 12 sinteticas), seis condiciones,
192 mediciones en una sola A100, orden contrabalanceado e intercambios fuera del timer.
Las latencias de 01-06, obtenidas en VMs distintas, permanecen descriptivas.

| Condicion | Mediana en 08 (s) | Ratio de medianas B13/metodo |
|---|---:|---:|
| B13 | 2.502 | 1.00 |
| B13_GREEDY | 2.491 | 1.00 |
| G_SINGLE | 0.744 | 3.36 |
| J64 | 0.980 | 2.55 |
| JSTEP | 2.907 | 0.86 |
| JFINAL | 3.507 | 0.71 |

J64 ratio 2.55: IC95 [1.49, 5.70], IC99.1667 [1.26, 7.85]. JSTEP y JFINAL tienen
intervalos que cruzan 1. Estos tiempos incluyen terminaciones invalidas: en 08 B13
termino con estado final en 32/32 entradas, J64 en 22/32, JSTEP en 19/32 y JFINAL en
19/32. No interpretar la terminacion rapida de fallos como mayor capacidad ni inferir
una ventaja conjunta de calidad/latencia sobre el test usando otra distribucion de
preguntas. Ningun hibrido demuestra aqui ser mas preciso y mas rapido que B13.

## Diagnostico JFINAL

400 candidatos guardados, 100 decisiones:

- Seleccion de J: 22 % de aciertos.
- Seleccion uniforme esperada: 9.5 %.
- Mayoria offline de respuestas normalizadas: 25 %.
- Oracle@4: 26 % (techo de este conjunto, no un sistema realizable).
- Brecha del selector: 4 puntos porcentuales.
- Eleccion correcta condicionada a existir un candidato correcto: 84.62 %.

La ausencia de propuestas correctas/formateadas limita considerablemente este brazo;
ni el oracle de sus cuatro propuestas alcanza B13. Mayoria y oracle son diagnosticos
offline, sin latencia de un sistema de consenso ejecutado.

## Auditoria Y Limitaciones

- Los siete ZIP finales pasaron integridad y las comprobaciones de IDs, semillas,
  configuraciones, hashes de entradas, modelos y version. Sin duplicados de checkpoints.
- Una segunda implementacion del parser recompuso los 600 aciertos desde predicciones
  inmutables y gold: cero discrepancias con graded_runs.csv.
- JFINAL: cuatro probabilidades finitas que suman 1 y mapeo argmax/permutacion correcto
  en 100/100 decisiones; cero candidatos parseables pero no terminales en el diagnostico.
- G_SINGLE tuvo una advertencia no bloqueante eager_vs_graph (coincidencia greedy 0.427):
  no se declara equivalencia de perfiles. No hubo ajuste o reejecucion por calidad.
- Los benchmarks son publicos, posible contaminacion; dificultad heuristica y reviewed=false.
- Una sola semilla y corpus de 100 problemas; no generalizar a otras tripletas ni modelos.
- SHARED4 usa una copia G: memoria real P_G+P_J, no cuatro replicas almacenadas. Entrenamiento
  diferente, baseline ampliado por terceros, perfil sin pensamiento y kernels vLLM.
- Fallos de operadores de cierre/monitoreo se recuperaron sin cambiar inferencias;
  falso fallo de auditoria de 08 corregido comparando IDs como conjunto, no orden barajado.
- Todas las VMs y guards fueron cerrados; se verifico ausencia de asignaciones en el servidor.

## Artefactos

- results/first_two_20260929/: 01_G_SINGLE y 02_B13, con intento fallido HF429 conservado.
- results/remaining_batches_20260929/03_B13_GREEDY/ a 06_JFINAL/: lotes restantes.
- results/remaining_batches_20260929/08_LATENCY/: plan, 192 mediciones, swaps, logs y notebook.
- results/remaining_batches_20260929/collected/: solamente siete ZIP finales verificados.
- results/remaining_batches_20260929/07_ANALYSIS/artifacts/: ZIP de analisis y notebook ejecutado.
- results/remaining_batches_20260929/07_ANALYSIS/extracted/: report.md, tablas CSV,
  graded_runs.csv, analysis.json y figuras.
- results/remaining_batches_20260929/analysis_independent_audit.json: recorreccion independiente.
