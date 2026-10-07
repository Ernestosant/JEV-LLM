# Resultados de la serie v2

Ejecucion completa, terminada el 1 de octubre de 2026 a las 05:14 UTC. Protocolo `protocolo_jev_qwen4b_v2.md`, con enmienda autorizada de recarga por fase, configuracion `config/experiment_v2_reload.json` y codigo 0.2.1. Una copia fisica del generador oficial Qwen3.5-4B; BF16, CUDA graphs, A100 de 40 GB. Semilla de inferencia 17.

## Resultado principal

La seleccion entre cuatro soluciones completas obtuvo **92/100**, frente a **85/100** del modelo grande con muestreo. Diferencia pareada: **+7 puntos porcentuales**, IC95 bootstrap estratificado **[0, 14]**, 10,000 remuestreos y semilla 271828.

La estimacion puntual supera el umbral practico de +5 puntos, pero el intervalo **incluye cero**: no se cumple la regla preregistrada para declarar superioridad confirmatoria. No equivale a demostrar igualdad. El contraste exacto bilateral de McNemar, de apoyo, da p=0.092285.

## Exactitud estricta

Cien problemas nuevos por brazo. Truncamientos, EOS sin FINAL valido y limites de rondas cuentan como incorrectos. El formato se informa aparte, sin recuperar aciertos mediante lectura permisiva.

| Condicion | Aciertos / 100 | IC95 descriptivo de exactitud | Formato valido |
|---|---:|---|---:|
| Qwen3.5-4B solo, una solucion con muestreo | 88 | [82, 94] % | 95 % |
| Modelo grande, una solucion con muestreo (ancla) | 85 | [78, 91] % | 95 % |
| Modelo grande, una solucion greedy | 82 | [74, 89] % | 93 % |
| 4B + JevK5, seleccion por bloques de hasta 64 tokens | 85 | [78, 91] % | 95 % |
| 4B + JevK5, seleccion al finalizar cada paso | 88 | [81, 94] % | 96 % |
| 4B + JevK5, seleccion entre cuatro soluciones completas | 92 | [87, 97] % | 96 % |

Ninguno de los cinco contrastes secundarios rechaza la hipotesis nula tras Holm. La seleccion final mejora la estimacion puntual del 4B solo en +4 puntos, con IC95 descriptivo [-1, 10]; su beneficio tampoco queda confirmado estadisticamente.

El control greedy del modelo grande es de sensibilidad, no una segunda hipotesis principal. La diferencia frente a este control es +10 puntos, IC95 descriptivo [3, 18]; no reemplaza el resultado de H1 ni se escoge el baseline a posteriori.

## Diagnostico del selector

Sobre 400 propuestas y 100 decisiones de seleccion final:

- Techo con cuatro propuestas, oracle@4: 93/100.
- Soluciones correctas elegidas: 92/100; captura 92 de los 93 casos con al menos una propuesta correcta, 98.92 %.
- Seleccion uniforme esperada: 85.5 %; voto por mayoria de respuestas normalizadas: 91 %.
- Brecha hasta el techo: un caso. Cuatro empates exactos resueltos conforme a la regla fijada.

Esto describe el aprovechamiento de las propuestas registradas. No es una comparacion confirmatoria adicional de selector frente a mayoria o uniforme.

## Latencia en una misma A100

Estudio secundario y descriptivo: **20 problemas de desarrollo + 12 sinteticos**, 32 mediciones por condicion, 192 en total. No es el mismo conjunto de 100 problemas del test. Se incluyen todos los estados de ejecucion, incluidos seis truncamientos.

| Condicion | Mediana | p95 |
|---|---:|---:|
| Qwen3.5-4B solo | 1.348 s | 6.393 s |
| Modelo grande con muestreo | 2.671 s | 11.130 s |
| Modelo grande greedy | 2.489 s | 11.418 s |
| 4B + selector por bloques | 2.024 s | 6.907 s |
| 4B + selector por pasos | 3.530 s | 17.018 s |
| 4B + selector de soluciones completas | 1.748 s | 9.392 s |

El cociente de medianas del ancla frente a la seleccion final es aproximadamente 1.53. No constituye una prueba confirmatoria de velocidad ni una afirmacion conjunta de superioridad en calidad y latencia.

La recarga exclusiva de B frente a G4+J, su cierre confirmado y el recalentamiento quedaron **fuera de `T_total`**. Hay 17 swaps registrados. La latencia es condicional a motores cargados y calientes, no al costo extremo a extremo del intercambio de modelos.

## Piloto y validacion

El piloto congelado de 40 problemas por brazo produjo seleccion final 39/40, ancla 32/40 y 4B solo 37/40; todas las bloqueantes pasaron y hubo cero timeouts/errores de infraestructura. Se aplico go antes de lanzar el test. Esos 120 casos no entran en los resultados confirmatorios.

La auditoria independiente del piloto coincide con los 120 aciertos publicados. La auditoria final coincide con **las 600 correcciones estrictas**, sin discrepancias, y verifica 400 propuestas, 100 decisiones, probabilidades calibradas a T=1.316, argmax, permutaciones y empates. Tambien reproduce diferencias puntuales y p-valores de McNemar/Holm. Los intervalos bootstrap no se han recalculado numericamente por el auditor externo; se conserva esta limitacion.

Pasaron 481 tests locales, con 13 avisos de deprecacion de Matplotlib. Los siete ZIP finales, hashes, conteos, IDs, notebooks y cierres estan verificados. Hay 18 copias canonicas JSON de notebooks, sin modificar los originales descargados ni los ZIP.

## Operacion y limites

- Maximo de dos asignaciones, operadores aislados y duraderos, guard WS cada 15 s/backend cada 20 s. Todas las VMs propias se cerraron y el servidor confirma **cero asignaciones activas**.
- El primer smoke de latencia fallo por OOM de B dormido. Sus artefactos se conservaron; tras autorizacion del usuario se repitio solo esa etapa con recarga exclusiva. No se redujeron precision, caches, presupuestos ni pesos.
- En la renovacion del proxy del estudio completo hubo dos ReadTimeout; el guard recupero socket conectado sin reiniciar ni repetir inferencia.
- El informe de costos registra 4.811 horas A100 de ventanas operativas en los jobs de la revision de recarga. Excluye los smoke anteriores y el intento de latencia fallido; no representa horas facturadas ni CU. No se dispone de CU o costo monetario real.
- Una semilla, 100 problemas de benchmarks publicos, dificultad heuristica, revision humana omitida (`reviewed=false`), posible contaminacion y baseline ampliado por terceros. Los parametros desplegados no igualan computo.
- Los tests de la serie 0.8B son distintos: no se comparan aciertos entre series como si estuvieran pareados.

## Artefactos

- `results/v2/analysis_reload/extracted/report.md`: informe original del analisis CPU, preservado.
- `results/v2/analysis_reload/extracted/final_table.csv` y `analysis.json`: tablas, contrastes y diagnosticos completos.
- `results/v2/analysis_reload/artifacts/analysis_v2_4b_confirmatory.zip`: analisis original, SHA256 `42a4cf2eac5c31a40370ba3f988a436008fc51ea07ef9c12e0ddd3e10feff740`.
- `results/v2/confirmatory_independent_audit.json` y `.md`: auditoria externa PASS.
- `results/v2/pilot_reload/decision.json` y `pilot_independent_audit.json`: go y auditoria del piloto.
- `results/v2/collected_finals_reload/`: exactamente siete ZIP finales, sin checkpoints duplicados.
- `results/v2/coordinator_reload_state.json`, `summary_reload.md` y `cost_estimate_reload.json`: estado completo, cierres e informacion operativa.
- `results/v2/canonical_notebooks.json`: hashes de originales y copias JSON.
