# Protocolo v2 · Una copia de Qwen3.5-4B con cuatro muestras + selector JevK5-9B frente al modelo de ~13B

**Proyecto:** JEV LLM · segunda serie experimental
**Fecha de diseño:** 30 de septiembre de 2026
**Estado:** serie completada el 1 de octubre de 2026: piloto go, 600 predicciones confirmatorias, 192 mediciones de latencia y análisis CPU auditado. Implementación v0.2.1 con enmienda autorizada de recarga por fase; se preserva el OOM del smoke original. Test y piloto congelados con semilla 20261001. Resultados en `docs/11_results_v2.md`; no se modifican hipótesis según estos resultados.
**Deriva de:** `protocolo_jev_qwen_v3.md` y `docs/protocolo_v3.1_enmienda.md`. Todo lo que este documento no modifica se hereda de ellos. El nombre "v2" identifica la **segunda serie experimental** de este proyecto; no guarda relación con los protocolos históricos v1/v2 que el v3 sustituyó.
**Resultados previos:** la serie con Qwen3.5-0.8B (v3 + v3.1, semilla 17, 600 ejecuciones) queda cerrada tal como se midió. Este protocolo no la corrige ni la reemplaza.

## 0. Resumen de la decisión

Se sustituye el generador de 0.8B por **una sola copia física de `Qwen/Qwen3.5-4B`**. En cada ronda, esa misma copia produce **cuatro muestras independientes**, cada una con su propia semilla, que se procesan en un mismo lote. No se cargan réplicas de pesos. JevK5-9B elige una de las cuatro propuestas con el mismo contrato, prompt nativo y calibración que en v3. El baseline sigue siendo el mismo modelo derivado de aproximadamente 13B.

**Pregunta principal:** ¿supera en exactitud el sistema "4B con cuatro muestras + selector" al modelo de ~13B con muestreo, sobre un **test nuevo** que ningún modelo ha visto en este proyecto?

La latencia pasa a ser una pregunta **secundaria**. Es de esperar que el 4B decodifique más lento que el 0.8B, y esta serie busca calidad, no velocidad.

### 0.1 Motivación y su carácter adaptativo

La motivación viene de un **análisis exploratorio posterior** a la serie de 0.8B. Ese análisis no forma parte de los resultados preregistrados.

- Con el 0.8B, incluso leyendo la respuesta de forma permisiva (último número de la salida), el mejor híbrido llegaba a 36 de 100, frente a 73 del modelo de ~13B con muestreo. El formato no explica la brecha.
- Entre cuatro soluciones completas, el techo "si se eligiera siempre bien" era de 38 de 100 con lectura permisiva. El selector alcanzó 36. **El cuello de botella era el generador, no el selector.**

Como esta decisión se tomó **después de abrir el test de v3**, ese test queda descartado para cualquier afirmación confirmatoria de esta serie. Se construye un test nuevo (§3).

## 1. Modelos fijados

| Papel | Modelo (Hugging Face) | Revisión | Parámetros del modelo lingüístico | Copias en GPU |
|---|---|---|---:|---:|
| Generador G4 | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 4,205,751,296 | **1** |
| Selector J | `alibiserikbay/JevK5-9B` | `d6521a18a86999190e9d775c915af3d6d6772fc4` | 8,953,803,264 | 1 |
| Baseline B | `DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking` | `717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe` | 12,413,584,128 | 1 |

- Los recuentos salen de las cabeceras safetensors publicadas en esas revisiones. Incluyen embeddings y cabeza de salida, cuentan una sola vez los pesos ligados y excluyen la torre visual y MTP. En G4 eso deja fuera 333,514,240 parámetros visuales y 120,599,552 de MTP.
- G4 es el checkpoint **postentrenado** (no `-Base`), con licencia Apache-2.0, arquitectura `Qwen3_5ForConditionalGeneration`, 32 capas (8 de atención completa) y embeddings ligados. Se carga solo la ruta de texto, sin MTP ni decodificación especulativa.
- **Importante:** la plantilla de chat de G4 activa el pensamiento por defecto cuando `enable_thinking` no está definido. Debe pasarse `enable_thinking=False` de forma explícita; el preflight lo comprueba y es bloqueante.
- El runtime JevK5 (`v0.3.0`, commit `6c6522fe…`), la temperatura de calibración 1.316 y la lectura de logits de letras no cambian.

### 1.0 Por qué Qwen3.5-4B y no un "Qwen3.8-4B"

A 30 de septiembre de 2026, **Qwen no ha publicado un modelo oficial de 4B en la familia 3.8**. Solo existen Qwen3.8-27B, Qwen3.8-Flash-Next (MoE de ~180B) y Qwen3.8-2.4T-A95B. Por eso, el 4B oficial más reciente es Qwen3.5-4B.

Se evaluó y **descartó** el destilado de terceros `empero-ai/Qwen3.8-4B-Distill` (revisión `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`), que es un Qwen3.5-4B reentrenado con ~45k razonamientos de Qwen3.8-2.4T. Los motivos:

- **Su propia ficha reporta peor rendimiento en GSM8K** que el base (78.5 % frente a 85.0 %, en modo razonamiento). Es justo la capacidad que esta serie necesita.
- Está entrenado para responder **siempre con un bloque `<think>`**. Eso choca con el modo sin pensamiento y el límite de 1,024 tokens del protocolo.
- Sus datos de entrenamiento no son públicos, lo que aumenta el riesgo de contaminación con benchmarks públicos de matemáticas.

Arquitectura, tokenizer y plantilla son idénticos a los de Qwen3.5-4B, así que se podría estudiar en una serie aparte sin cambiar el pipeline.

### 1.1 Presupuesto de parámetros (redefinido)

Con una sola copia, el criterio de emparejamiento pasa a ser el de **parámetros únicos desplegados**:

`P_desplegado = P_G4 + P_J = 13,159,554,560` frente a `P_B = 12,413,584,128`.

Por tanto, el baseline es un **5.67 % más pequeño** que el sistema híbrido: `100·(P_B/P_desplegado − 1) = −5.67 %`. El criterio de "cuatro réplicas" (`4·P_G4 + P_J ≈ 25.78B`) **ya no se cumple y no se afirma**.

La comparación es entre sistemas con número de parámetros residentes similar. No iguala FLOPs, pasos de inferencia, tiempo de GPU ni entrenamiento. El sistema híbrido **procesa más tokens** (cuatro muestras más la lectura del selector) y así se informa.

## 2. Preguntas e hipótesis

| Código | Hipótesis | Tipo |
|---|---|---|
| **H1** | La selección entre cuatro soluciones completas del 4B tiene **mayor exactitud** que el modelo de ~13B con muestreo | **Principal (confirmatoria)** |
| H2 | La selección por bloques de hasta 64 tokens del 4B tiene mayor exactitud que el modelo de ~13B con muestreo | Secundaria |
| H3 | La selección por pasos del 4B tiene mayor exactitud que el modelo de ~13B con muestreo | Secundaria |
| H4 | El selector aporta sobre el 4B solo (cada híbrido frente a una solución del 4B) | Secundaria |
| H5 | Latencia en la misma GPU (estudio aparte) | Secundaria, descriptiva |

Por qué la hipótesis principal es la selección sobre soluciones completas:
- Fue la variante con mayor aprovechamiento del techo de las cuatro propuestas en la serie exploratoria.
- Su cómputo es el más acotado: una sola lectura del selector.
- Esta elección se declara **antes** de construir el test nuevo.

## 3. Datos

### 3.1 Test nuevo (congelado antes de cualquier inferencia de esta serie)

- **100 problemas** con el mismo diseño y las mismas reglas de `data/build_dataset.py`:
  - 5 dominios × (8 fáciles, 8 medios, 4 difíciles).
  - Mismas fuentes y revisiones fijadas.
  - Mismos filtros: respuesta exacta, |respuesta| ≤ 999, sin dinero, sin porcentajes como respuesta, ≤ 512 tokens renderizados.
  - Mismas reglas de dificultad y de preferencia por competición en el tramo difícil, con los mismos rellenos declarados.
- **Exclusión estricta:** ningún problema de los 120 de la serie anterior (test y desarrollo) y ningún duplicado aproximado de ellos, según el mismo criterio de 8-gramas.
- Semilla de construcción: **20261001**.
- Corrección del filtro monetario antes de inferir: se excluyen también preguntas explícitas sobre dinero, como "How much money is used?", que el patrón heredado omitía. Las dos construcciones rechazadas (una por ese filtro y otra por un cambio no autorizado de semilla) quedan archivadas en `results/v2/preparation/`. La construcción final mantiene **20261001**; no se ha escogido según resultados de modelos.
- Disponibilidad del pool tras excluir los 120 usados, antes de aplicar el filtro de longitud y el de duplicados:

| Dominio | Fácil | Medio | Difícil (competición / relleno) |
|---|---:|---:|---|
| Aritmética | 442 | 207 | 0 / 8 (GSM8K ≥ 7 pasos) |
| Álgebra | 41 | 12 | 14 / 27 (MATH L4–5) |
| Proporciones y porcentajes | 24 | 23 | 0 / 4 (2 GSM8K + 2 MATH) |
| Teoría de números | 27 | 11 | 10 / 37 |
| Conteo y probabilidad | 51 | 34 | 6 / 109 |

Hay tres estratos ajustados: proporciones difíciles (4 disponibles para 4 plazas), teoría de números media (11 para 8) y álgebra media (12 para 8).

**Si algún estrato no se llena tras aplicar los filtros, se publica el N real reducido.** No se relajan filtros ni se reasignan dominios después de mirar.

- Archivos separados, igual que en v3 §3.3: `test_v2_inputs.jsonl` y `test_v2_gold.jsonl`, más `schedule_v2.json` y `SHA256SUMS`. La inferencia recibe solo los inputs.
- La revisión humana (segundo revisor y revisión ciega de 20 IDs preregistrados) debe completarse **antes** de la corrida confirmatoria. Si no se completa, se publica como limitación.
- El usuario autorizó continuar sin revisión humana: `reviewed=false` en todos los registros; no se presenta como un dataset revisado.

### 3.2 Conjunto piloto

- **40 problemas** tomados de lo que quede del pool **después** de construir el test v2: 8 por dominio, fáciles y medios.
- No se solapan ni con el test v2 ni con los datos anteriores.
- Se construyen en el mismo commit que el test v2 y se congelan con él.
- Selección determinista: 4 fáciles y 4 medios por dominio; si faltan medios, se completa con fáciles del mismo dominio, sin cambiar etiquetas. El piloto congelado contiene 5 fáciles y 3 medios en teoría de números, y 4+4 en los otros dominios.

### 3.3 Desarrollo

Se reutilizan los 20 problemas de desarrollo de la serie anterior. Sirven solo para depurar el software, calentar motores y el estudio de latencia; no para elegir configuraciones.

## 4. Matriz experimental

| Condición (descripción) | Código | Modelos residentes |
|---|---|---|
| Qwen3.5-4B solo: una solución con muestreo | `G4_SINGLE` | 1 × G4 |
| Modelo de ~13B: una solución con muestreo (**ancla**) | `B13` | 1 × B |
| Modelo de ~13B: una solución sin muestreo (control de sensibilidad) | `B13_GREEDY` | 1 × B |
| 4B + selector: elegir entre cuatro continuaciones cada bloque de hasta 64 tokens | `J64_G4` | 1 × G4 + J |
| 4B + selector: elegir entre cuatro continuaciones al terminar cada paso | `JSTEP_G4` | 1 × G4 + J |
| 4B + selector: elegir entre cuatro soluciones completas | `JFINAL_G4` | 1 × G4 + J |

- **Perfil fijado: semilla 17 en todos los brazos**, igual que en la serie anterior. Da 600 ejecuciones. Este perfil no se amplía según los resultados. Cualquier ampliación posterior se declara como réplica o análisis exploratorio.
- Diagnósticos offline sobre las cuatro soluciones completas de `JFINAL_G4`, sin generación adicional: selección uniforme esperada, voto por mayoría de respuestas normalizadas, techo con cuatro propuestas (oracle@4) y exactitud condicionada a que exista una propuesta correcta.

## 5. Configuración común (sin cambios)

Todo lo siguiente se mantiene exactamente igual que en v3 §5 y v3.1:

| Campo | Valor |
|---|---|
| Pesos | BF16, sin cuantización ni offload |
| Muestreo | `temperature=0.7`, `top_p=0.9`, `top_k=0` (desactivado), penalizaciones neutras |
| Modo de pensamiento | desactivado explícitamente en G4 y en B |
| Salida máxima | 1,024 tokens |
| Presupuesto total de propuestas híbridas | 4,096 tokens muestreados |
| Límite por bloque | 64 tokens |
| Límite por paso | 128 tokens, máximo 64 rondas |
| Contexto de G4 / B | 4,096 tokens |
| Entrada máxima del selector | 16,384 tokens, sin truncar |
| Timeout por ejecución | 600 s |

**Prompts:** se usan **sin modificar** los archivos `prompts/generador.txt`, `prompts/criterio_paso.txt` y `prompts/criterio_final.txt` (hashes en el manifiesto). El único factor que cambia es el generador, para poder atribuirle el efecto.

Queda **fuera de esta serie** cualquier variante de prompt, por ejemplo pedir explícitamente "¿cuál es el siguiente paso?". Si se quiere estudiar, será otro protocolo, con su propio desarrollo y un test separado.

## 6. Selector y algoritmos

- Contrato del selector: sin cambios (v3 §6 y v3.1 A3). Mismo JSON con `option_0…option_3`, mismos criterios, permutación con semilla derivada, argmax con empate a la primera posición y equivalencia con el runtime nativo como validación bloqueante.
- Algoritmos: sin cambios (v3 §7 y v3.1 A4). Mismas reglas de presupuesto, terminación por línea FINAL o EOS, sobreproducción, fronteras de paso y bucle por pasos con timeout.

## 7. Perfil de implementación: `1COPY-G4-VLLM-GRAPH`

- vLLM 0.30.0 con CUDA graphs, BF16 y el stack fijado de v3.1. Un motor G4 con `max_num_seqs=4`, que recibe las cuatro peticiones en el mismo lote, y un motor J en otro proceso.
- **Memoria planificada en una A100 de 40 GB:** pesos de G4 ≈ 7.8 GiB, J ≈ 16.7 GiB, caché KV explícita (G4 3 GiB, J 4 GiB), más contextos y grafos. Total ≈ 33 GiB. Es una estimación; el estrés del preflight es el que decide.
- **Si no cabe:** se detiene la corrida y se reporta. No se cuantiza, no se descarga a CPU y no se cambia de modelo sin un nuevo protocolo.
- **GPU:** A100 de 40 GB en **todos** los brazos, incluido el 4B solo, para reducir la heterogeneidad de hardware respecto a v3.1.
- `KV_CACHE_GB_G=3.0` es un parámetro de infraestructura fijado en este documento. Todo lo demás, igual que en v3.1.

## 8. Validaciones previas (preflight)

Las validaciones de v3.1 A6 siguen siendo obligatorias, con estas adiciones:

1. **Plantilla de G4 sin pensamiento** (bloqueante): el render debe terminar en `<think>\n\n</think>\n\n`, y se guardan el texto y su hash.
2. **Tokenizer de G4:** se guarda el hash y se verifica que los tokens de EOS/PAD y el conjunto de tokens con salto de línea se construyan para G4, sin reutilizar la caché del 0.8B.
3. **Estrés conjunto** (bloqueante): cuatro secuencias del 4B de longitud máxima más una entrada de ~16k tokens en el selector, con ambos motores residentes. Se informa la memoria máxima.
4. **Eager frente a CUDA graphs en G4 (A100):** se exige un umbral de acuerdo greedy ≥ 0.75. Si no se alcanza, la validación no bloquea, pero hay que ampliar la comparación a 20 prompts de desarrollo y **publicarla**. Además, en el piloto se repite esta comprobación con el 0.8B en A100 para aclarar la advertencia de 0.427 observada en L4 en la serie anterior.
5. Equivalencia del selector vLLM frente al runtime nativo (bloqueante), igual que en v3.1.

### 8.1 Enmienda de infraestructura: recarga de fases en latencia

El smoke con los cuatro brazos G4 pasó en A100 de 40 GB: cuatro salidas de 1,024 tokens y una entrada de 16,330 tokens en J, con pico NVML observado de 36.12 GiB. Sin embargo, el estudio de latencia falló al mantener B dormido mediante `sleep(level=1)`: B retenía 2.83 GiB de GPU y G4 no pudo reservar la caché fijada de 3 GiB. La sesión permaneció conectada; se conservaron artefactos y se verificó su liberación.

**Cambio autorizado por el usuario tras el OOM y antes del piloto:** descargar completamente de GPU y cerrar los motores de una fase antes de cargar los de la siguiente, en la **misma A100 de 40 GB**. Nunca se mantienen B y G4+J residentes a la vez. No hay cuantización, offload ni reducción de las cachés G3/J4/B3 GiB. Cuando se mide G4 solo, J se cierra y se vuelve a cargar/probar antes de medir un híbrido.

Descargas, cierre confirmado de procesos, cargas, probes del selector y recalentamiento quedan **fuera de `T_total`** y se registran en `swaps.jsonl` y `warmup.jsonl`. Las seis condiciones, los bloques contrabalanceados, los datos, las semillas y los algoritmos de resolución no cambian. Se publica por separado el costo de recarga; la latencia resultante es condicional a motores cargados y calientes, no un costo extremo a extremo de intercambio de modelos.

La enmienda se fija en `config/experiment_v2_reload.json`, código 0.2.1, `notebooks/v2-reload/` y `dist/v2-reload/`. La configuración, bundles 0.2.0 y todos los artefactos anteriores permanecen intactos. Se reutiliza únicamente evidencia de desarrollo completada y verificada; se repite el smoke de latencia fallido con un nuevo nombre de ejecución. Piloto y confirmatoria, si se autoriza por go, usan la nueva revisión consistente.

## 9. Piloto de viabilidad (preregistrado)

- **Brazos:** `JFINAL_G4`, `B13` y `G4_SINGLE`, sobre el conjunto piloto de 40 problemas con semilla 17.
- **Qué se mide:** formato válido, exactitud, oracle@4, fracción del techo que captura el selector, memoria, tiempos, timeouts y OOM.
- **Regla para pasar a la corrida confirmatoria** (se aplica tal cual):
  1. Todas las validaciones bloqueantes pasan.
  2. Menos del 5 % de ejecuciones con timeout o error de infraestructura.
  3. La exactitud de `JFINAL_G4` en el piloto es **mayor o igual** que la de `B13` en el piloto (estimación puntual).
- **Si no se cumple:** se detiene la serie y se publica el piloto como resultado negativo de viabilidad. Una corrida posterior a un no-go se etiqueta como **exploratoria**.
- El piloto **no** autoriza a cambiar prompts, hiperparámetros, presupuestos ni modelos. Solo se admiten reparaciones de infraestructura, que quedan en el registro de cambios.
- Los datos del piloto no entran en el análisis confirmatorio.
- La regla de errores se aplica tanto a cada brazo (40 casos previstos) como al agregado (120). Una ejecución incompleta o con artefactos no verificables nunca autoriza continuar.

## 10. Orden de ejecución

Igual que en v3.1: un notebook por condición, **lotes de dos VMs como máximo**, un operador duradero por VM y guard de sesión (pings WebSocket cada 15 s, comprobación de la asignación cada 20 s, cierre confirmado contra el servidor).

1. **Piloto:** lote [`JFINAL_G4`, `B13`], luego [`G4_SINGLE`]. Se aplica la regla del §9.
2. **Confirmatoria:** lotes [`B13`, `B13_GREEDY`], [`G4_SINGLE`, `J64_G4`] y [`JSTEP_G4`, `JFINAL_G4`].
3. **Estudio de latencia** en una sola A100: mismo diseño que el notebook 08 (20 problemas de desarrollo + 12 sintéticos, bloques contrabalanceados, cambio de modelos fuera del cronómetro), con G4.
4. **Análisis** en CPU, después del estudio de latencia.

Las latencias de los pasos 1 y 2 son **descriptivas**, porque vienen de VMs distintas. Solo el paso 3 aporta comparaciones de latencia en la misma GPU.

## 11. Evaluación

- **Principal:** el evaluador estricto de v3 §11, sin cambios. Exactamente una línea `FINAL:`, gramática numérica estricta y comparación racional exacta. Timeouts, truncados, EOS sin FINAL y límites de rondas cuentan como incorrectos.
- **Secundaria y declarada como exploratoria:** una lectura permisiva (último número de la salida) para separar fallos de formato de fallos de razonamiento. **Nunca se usa para contrastar hipótesis.**
- Se corrige de forma offline, después de guardar predicciones inmutables, y con una segunda implementación independiente del parser como auditoría, igual que en la serie anterior.

## 12. Estadística y reglas de conclusión

- **Unidad:** el problema. Bootstrap pareado por ID, estratificado por dominio, 10,000 remuestreos, semilla 271828.
- **H1 (principal):** diferencia de exactitud `JFINAL_G4 − B13`, en puntos porcentuales, con IC del 95 %. Se declara **"más preciso"** si el IC excluye 0. El umbral de interés práctico es **+5 puntos**, y se informa si la estimación puntual lo supera.
- **H2–H4:** familia fija de cinco comparaciones: `J64_G4` y `JSTEP_G4` frente a `B13`, y cada uno de `J64_G4`, `JSTEP_G4` y `JFINAL_G4` frente a `G4_SINGLE`. No se selecciona el mejor híbrido después de ver resultados. Se publican p-valores bilaterales exactos de McNemar ajustados por Holm (alpha 0.05), IC bootstrap del 95 % descriptivos e IC bootstrap simultáneos conservadores de Bonferroni del 99 % (0.05/5). Estos últimos no se denominan "IC de Holm". La hipótesis principal y su IC95 no cambian.
- **Control con el modelo de ~13B sin muestreo:** es de sensibilidad. No se elige el mejor baseline por problema, pero se informa con transparencia si el híbrido supera al ancla con muestreo y no a la versión sin muestreo.
- **Potencia:** con 100 problemas y una semilla, el IC del 95 % de una diferencia pareada tiene un semiancho típico de unos 7 puntos. En la serie anterior, "~13B sin muestreo frente a con muestreo" dio +8 puntos con IC95 [+1, +15]. **Diferencias reales menores de unos 8–10 puntos pueden quedar sin resolver.** Eso se reporta como incertidumbre, no como equivalencia.
- **Comparación con la serie de 0.8B:** solo descriptiva. Los tests son distintos, así que no hay pareo entre series. Se permite comparar diferencias respecto al ancla dentro de cada serie, siempre etiquetadas como tales.

## 13. Métricas y artefactos

Los mismos de v3 §12 y v3.1 (predicciones, métricas por ejecución, candidatos, decisiones, rondas, memoria, eventos, fallos, preflight, logs, manifiesto y zips). Se añaden:
- Tabla con **nombres descriptivos de las condiciones**, además del código.
- Tokens procesados por el sistema híbrido frente al ancla (proxy de cómputo, etiquetado como tal).
- Fracción del techo de cuatro propuestas que captura el selector.

## 14. Costo estimado (sin medir)

- Preparación por notebook en A100: 12–15 min (instalación, descarga de ~26 GB en los brazos con B, arranque y preflight). La descarga de G4 es de ~9.3 GB.
- Inferencia de los híbridos con 4B: estimada entre 2 y 3 veces la del 0.8B por muestra. Del orden de 15–30 min por brazo con 100 problemas.
- **Total estimado:** piloto ≈ 1.5–2 h de A100 (≈ 8–11 CU); confirmatoria ≈ 4–6 h (≈ 22–32 CU); latencia ≈ 45 min (≈ 4 CU). Hay que revisar el ETA real tras el piloto antes de lanzar la confirmatoria.

## 15. Limitaciones obligatorias

- La motivación del cambio de generador es adaptativa (§0.1). Se mitiga con un test nuevo y el preregistro, pero no desaparece.
- Una sola semilla y 100 problemas. Una sola tripleta de modelos. No se generaliza a "todos los Jev" ni a otros generadores.
- El emparejamiento es por parámetros únicos desplegados, no por cómputo: el sistema híbrido procesa más tokens.
- Entrenamientos distintos de G4, J y B. El baseline es un modelo ampliado por terceros. Todo se ejecuta sin pensamiento, en vLLM con CUDA graphs, y el muestreo no es invariante al lote.
- Los benchmarks son públicos: posible contaminación. La dificultad es heurística. Revisión humana pendiente si no se completa antes.
- Los tramos difíciles de aritmética y de proporciones siguen dependiendo de rellenos no competitivos.

## 16. Cambios de implementación previstos (antes de ejecutar)

1. `config/experiment_v2.json`: G → `Qwen/Qwen3.5-4B@851bf6e8…`, `KV_CACHE_GB_G=3.0`, perfil `1COPY-G4-VLLM-GRAPH` y lista de condiciones con los códigos `_G4`.
2. `data/build_dataset_v2.py`: excluir los 120 usados y sus duplicados aproximados, semilla 20261001, generar test v2 y piloto, y publicar el N real por estrato.
3. `jevlab`: versión **0.2.0** (entra en el `config_hash`), soporte de `SPLIT="pilot"` y `SPLIT="test_v2"`, y etiquetas descriptivas en el análisis. Sin cambios en los algoritmos, el selector ni el evaluador.
4. Notebooks regenerados para la serie v2 y bundles separados (el de inferencia sin gold, el de análisis con gold).
5. Tests locales: exclusión de datos, cuotas, hashes, ausencia de gold en los inputs y plantilla sin pensamiento de G4.
6. Smoke con `SPLIT="dev"` y `N_PROBLEMS=1` en cada notebook nuevo **antes** del piloto. Nunca sobre el test v2.

Decisiones operativas aprobadas: smoke reducido de los cuatro brazos que usan G4 y latencia mínima (2 dev + 4 sintéticos), más el chequeo eager/graph del 0.8B en A100. Se conservan los códigos internos `G_SINGLE`, `J64`, `JSTEP` y `JFINAL`; bajo `config/experiment_v2.json` equivalen respectivamente a `G4_SINGLE`, `J64_G4`, `JSTEP_G4` y `JFINAL_G4`. `DATA_SUBDIR="data/v2"` usa archivos `{test,pilot,dev}_{inputs,gold}.jsonl`; los inputs se empaquetan sin gold ni referencias. El operador aislado admite recuperaciones del guard durante 180 s sin reiniciar trabajo remoto, descarga checkpoints y confirma el cierre contra el servidor.

## 17. Registro de cambios de este protocolo

| Fecha (UTC) | Cambio | Motivo |
|---|---|---|
| 2026-09-30 | Versión inicial | Diseño de la serie con una copia de 4B, decidido tras el análisis exploratorio de la serie de 0.8B |
| 2026-09-30 | Se confirma Qwen3.5-4B oficial y se documenta el descarte de `empero-ai/Qwen3.8-4B-Distill` (§1.0) | No existe un Qwen3.8-4B oficial; el destilado rinde peor en GSM8K según su propia ficha y está entrenado para pensar siempre |
| 2026-09-30 | Aclaraciones preinferencia: familia secundaria fija de cinco contrastes, Holm en p-valores y Bonferroni en IC simultáneos; errores del piloto por brazo y agregado | Evitar selección posterior y etiquetado estadístico incorrecto |
| 2026-09-30 | Datos finales 100/40, filtro monetario corregido y semilla original 20261001 restaurada con auditorías de candidatos rechazados; revisión humana omitida por autorización | Corregir preparación antes de inferencia sin relajar cuotas ni filtros |
| 2026-09-30 | Implementación aislada 0.2.0, alias internos, smoke reducido y recuperación acotada del guard | Plan de ejecución autorizado y protección de las sesiones Colab |
| 2026-10-01 | Enmienda 0.2.1 autorizada: recarga exclusiva B frente a G4+J en la misma A100, fuera del cronómetro; repetir solo smoke de latencia | OOM real del estudio con B dormido; conservar los brazos de desarrollo que sí terminaron |
