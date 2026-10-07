# Enmienda v3.1 al protocolo `protocolo_jev_qwen_v3.md`

**Fecha:** 27–28 de septiembre de 2026 · **Estado:** congelada **antes** de construir el dataset y de ejecutar
cualquier caso de test. Cualquier cambio posterior se registra en `docs/06_desviaciones_y_cambios.md` con fecha.

Esta enmienda **no** modifica modelos, prompts, criterios del selector, presupuestos, semillas ni la estadística
del protocolo v3. Declara, con su justificación, las diferencias de implementación que impone el entorno
(vLLM en Google Colab Pro) y las decisiones de diseño pedidas por el investigador.

## A1. Perfil principal: `SHARED4-VLLM-GRAPH` (sustituye a `R4-BF16-EAGER`)

* **Qué cambia.** Los cuatro generadores son **una sola copia** de `Qwen/Qwen3.5-0.8B` servida por un motor vLLM
  que recibe **cuatro peticiones independientes por ronda**, cada una con su semilla derivada (§5.2) en el mismo
  batch (continuous batching). El protocolo v3 preveía `SHARED4` como perfil separado (§0, §8.3); aquí pasa a ser
  el perfil principal.
* **Por qué.** Es la forma de generar los 4 candidatos en paralelo con menor costo de cómputo: el decode está
  limitado por ancho de banda de memoria, y 4 secuencias en batch cuestan casi lo mismo que una. R4 en vLLM exige
  4 motores = 4 procesos CUDA (el §8.1 pide evitarlo salvo medirlo y documentarlo).
* **Consecuencia para la hipótesis de presupuesto.** `P_desplegado_R4 = 4·P_G + P_J` se reporta sólo como cálculo
  **nominal**. La VRAM y el inventario reales son `P_G + P_J`; nunca se infla la memoria con copias ficticias.
  Parámetros medidos desde cabeceras safetensors (ruta lingüística, sin visión ni MTP):
  `P_G = 752,393,024`, `P_J = 8,953,803,264` (P_B se registra en el manifiesto de B13).
* **Referencia R4 opcional.** Parámetro `RUN_R4_MEMORY_CHECK=True` en los notebooks híbridos: arranca 3 motores G
  extra junto a G+J, lee NVML y los apaga (solo memoria, sin corridas).

## A2. Un notebook por condición en VMs distintas; latencia confirmatoria en el notebook 08

* Los brazos `G_SINGLE`, `B13`, `B13_GREEDY`, `J64`, `JSTEP`, `JFINAL` se ejecutan en notebooks separados que
  pueden correr en paralelo en VMs distintas (petición del investigador). Esto rompe el contrabalanceo por bloques
  dentro de una misma sesión (§10.1) y compara latencias entre GPUs/hosts distintos (§9.1).
* **Mitigación.** (i) Cada ejecución guarda hora de pared UTC de inicio/fin y los tiempos `perf_counter` (T_total,
  primera salida aceptada, propuestas, selector, prefill, control). (ii) Las latencias y speedups de los notebooks
  01–06 se etiquetan **descriptivos**. (iii) El notebook **08_estudio_latencia** mide todas las condiciones en **una
  sola A100**, en bloques contrabalanceados (B antes/después alternando; orden rotado de los híbridos), sobre un
  conjunto de preguntas **distinto del test** (20 problemas de desarrollo + prompts sintéticos con bandas de longitud
  de salida controladas). El intercambio B ↔ G+J usa el *sleep mode* de vLLM fuera del cronómetro. **Las conclusiones
  de H-latencia salen del notebook 08.** (iv) Una sonda de calibración de hardware se registra en cada VM sólo como
  covariable de sensibilidad (nunca para normalizar).
* `G_SINGLE` corre en L4 (más barata); su latencia no entra en las seis comparaciones principales.

## A3. Selector JevK5-9B servido por vLLM

* Se conserva el contrato del §6: prompt nativo (`jevk5.prompt.messages` + plantilla con `enable_thinking=False`),
  criterio JSON con `option_0..option_3`, lectura de **logits de letras**, temperatura publicada **1.316**, decisión
  por argmax (empates → primera posición permutada), sin muestrear la decisión ni pedir razonamiento.
* **Implementación.** Una sola pasada de prefill en vLLM; se leen los *raw logits* de las letras A–D
  (`logprobs_mode="raw_logits"` + `logprob_token_ids`), se dividen por T y se aplica softmax. vLLM genera
  técnicamente un token que se descarta. Sin caché de prefijo interronda (§6.1).
* **Equivalencia verificada (preflight, bloqueante).** Un subproceso carga el runtime publicado
  `jevk5.JevK5(snapshot, graphs=False)` (transformers) y compara, sobre las mismas IDs de entrada, probabilidades y
  argmax. Tolerancia: el argmax debe coincidir salvo que el margen top-2 nativo sea ≤ 2 ulps de bf16. En el smoke:
  8/8 decisiones coinciden, |Δp| máx = 0.022.

## A4. Motor de inferencia y terminación

* vLLM **0.30.0** con **CUDA graphs** (perfil `VLLM-BF16-GRAPH`) en todas las condiciones; `ENFORCE_EAGER` queda como
  parámetro para una réplica eager. El preflight compara tokens greedy graph vs eager en dev (smoke: 100 % iguales).
* Todas las condiciones usan el mismo **bucle por pasos** (`add_request`/`step`/`abort_request`): terminación
  uniforme al completarse la línea FINAL, en EOS o por límite; TTFT medido; timeout real de 600 s.
* **Sobreproducción.** Tokens entregados después del token que completa la línea FINAL se cuentan como
  sobreproducción y se retiran del candidato que ve J y del prefijo. Tokens que el motor pudiera computar tras el
  `abort` y nunca entregar **no son observables** (limitación declarada).
* **Cachés.** No se clona estado KV/DeltaNet manualmente: cada ronda de J64/JSTEP reenvía `prompt + IDs aceptados`
  (sin re-tokenizar) y vLLM hace el prefill completo (prefix caching desactivado por defecto; para modelos híbridos
  es experimental). Esas relecturas se cuentan como `processed_prompt_tokens`. El test de clonado del §8.2 se
  sustituye por un **test de continuidad de tokens** (prefijos 1, 63, 64, 65, 127, 128: continuar desde el prefijo
  debe reproducir la continuación greedy de una sola petición) — smoke: 100 % de coincidencia.
* El §8.1 (streams) se sustituye por una **auditoría de batching**: 4 peticiones secuenciales vs en batch sobre 10
  prefijos de desarrollo (smoke: speedup ×2.17).
* **Memoria.** vLLM corre en subprocesos, por lo que `torch.cuda.max_memory_*` del notebook no aplica. Se reporta:
  memoria de pesos y KV del log de vLLM, KV dimensionada explícitamente con `kv_cache_memory_bytes` ("reservada"),
  y NVML muestreado a 10 Hz (pico observado, no exacto). No se llama `empty_cache()` por ronda.

## A5. Dataset muestreado de benchmarks públicos (sustituye a problemas originales)

* 100 test + 20 desarrollo muestreados de **GSM8K test**, **MATH test** (algebra, number_theory,
  counting_and_probability, prealgebra), **AMC12 2022–23** (`AI-MO/aimo-validation-amc`) y **AIME 2024–2025**, con
  revisiones fijadas por SHA (ver `docs/02_dataset.md`). 5 dominios × (8 fáciles, 8 medios, 4 difíciles).
* **Dificultad heurística** (no revisión humana): pasos `<<…>>` en GSM8K, nivel en MATH, competición = difícil.
* **Exclusiones preregistradas:** respuestas no exactas (radicales, π, listas…), respuestas ≥ 1000 (separador de
  miles), problemas cuya respuesta natural es dinero o porcentaje, `[asy]`, unidades/grados, bases, "nearest".
* **Contaminación:** es probable que GSM8K/MATH/AMC/AIME estén en el preentrenamiento de los modelos; J declara
  entrenamiento con el *train* de GSM8K (se usa sólo el *test*). Limitación obligatoria.
* **Revisión humana pendiente:** `reviewed=false`; el investigador completa `data/review_sheet.csv` y la revisión
  ciega de los 20 IDs preregistrados en `data/blind_review_ids.json` (§11).

## A6. Preflight integrado

Las validaciones del §14 corren dentro de cada notebook (parser, lock/hashes, plantillas, longitud de prompt,
semántica de muestreo, referencia nativa + equivalencia del selector, continuidad, batching, estrés de longitud
máxima incluida una entrada de ~16k en J, eager vs graph). Las bloqueantes detienen el notebook si fallan.

## A7. Versionado del código

`jevlab.__version__` forma parte del `config_hash`: una reparación del pipeline nunca reanuda ni se mezcla con
corridas del código anterior (§11: repetir todas las condiciones afectadas).
