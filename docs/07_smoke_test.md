# Prueba de humo (smoke test) — 28 sep 2026

**Objetivo:** ejecutar **cada notebook completo** sobre **un único problema** (`N_PROBLEMS=1`, `RUN_TAG="smoke"`,
semilla 17) para comprobar instalación, preflight, motores, bucle, artefactos, análisis y herramientas.
**No son resultados del experimento** (n=1; las cifras de exactitud no significan nada).

Artefactos: `results/smoke/a100/`, `results/smoke/l4/`, `results/smoke/analysis/` (salida de `07_analisis`),
`results/smoke/_superseded_v0.1.0/` (primera pasada, anterior a la corrección del EOS).

## Ejecución

| Notebook | VM | Forma | Código | Resultado | Duración total del notebook |
|---|---|---|---|---|---|
| 04_J64 | A100 40GB | celda a celda (depuración) y luego papermill | 0.1.1 | ok | 4.5 min |
| 05_JSTEP | A100 40GB | papermill (`launch_seq.sh`) | 0.1.1 | ok | 4.7 min |
| 06_JFINAL | A100 40GB | papermill | 0.1.1 | ok | 4.6 min |
| 02_B13 | A100 40GB | papermill | 0.1.1 | ok | 3.3 min |
| 03_B13_GREEDY | A100 40GB | papermill | 0.1.1 | ok | 3.4 min |
| 08_estudio_latencia | A100 40GB | papermill (2 dev + 4 sintéticos, bloques de 3, 6 condiciones → 36 corridas) | 0.1.1 | ok | ~18 min (motores B+J+G con sleep mode ≈ 10 min) |
| 01_G_SINGLE | L4 | papermill (`launch.sh`) | 0.1.1 | ok | ~2 min (tras primera instalación) |
| 07_analisis | CPU | papermill | 0.1.1 | ok | < 2 min |

Instalación del stack en una VM nueva: ~3.5 min. Los notebooks siguientes en la misma VM reutilizan paquetes,
pesos descargados y caché de compilación de vLLM.

## Preflight (todas las validaciones pasaron)

Ver tabla completa en `05_preflight_validacion.md`. Destacados: hashes de todos los archivos de los 3 modelos
coinciden con el Hub; equivalencia selector vLLM vs runtime JevK5 nativo 8/8 (|Δp| ≤ 0.022); continuidad de tokens
100 %; graph = eager 100 %; batching ×2.17; entrada de 16,232 tokens en J sin error.

## Memoria y parámetros medidos

| | Valor |
|---|---|
| P_G (lingüístico, embeddings ligados contados una vez) | 752,393,024 |
| P_J | 8,953,803,264 |
| Pesos cargados por vLLM | G 1.53 GiB · J 16.71 GiB · B 23.25 GiB |
| NVML tras arrancar motores (KV reservada incluida) | G+J 25.99 GiB · B 27.35 GiB · G sola (L4) 4.36 GiB |
| Pico NVML observado en casos | híbridos 26.4–28.8 GiB · B 27.4–29.6 GiB |
| Arranque de motores (con caché) | J ≈ 39 s · G ≈ 34 s · B ≈ 43 s (primera vez sin caché: ~3–4 min) |

## Caso de prueba (jev-t-003, GSM8K; gold = 26)

| Condición | Estado | Salida (final) | T_total |
|---|---|---|---|
| B13 | final | `FINAL: 26` ✔ | 1.38 s |
| B13_GREEDY | final | `FINAL: 26` ✔ | 1.14 s |
| G_SINGLE | eos_invalid | `...84 - 10 = 74` (sin FINAL) | 0.37 s (L4) |
| J64 | final | `FINAL: 36` ✘ | 0.99 s |
| JSTEP | eos_invalid | `Final answer: 26` (formato no admitido) | 1.69 s |
| JFINAL | eos_invalid | J eligió (p=0.98) la única candidata con razonamiento correcto, pero ninguna tenía línea FINAL | 3.62 s |

Estudio de latencia (misma A100, 6 ítems): speedup mediano vs B13 — J64 1.78, JFINAL 1.03, JSTEP 0.49,
G_SINGLE 1.78 (IC muy anchos con n=6; sólo valida el mecanismo).

## Hallazgos (qué falló, por qué, cómo se resolvió)

1. **EOS dentro de `token_ids` (bug, corregido → v0.1.1).** vLLM entrega `<|im_end|>` en los IDs; aparecía en el
   texto (`FINAL: 36<|im_end|>` → formato inválido) y en lo que veía J. Se elimina del texto y se cuenta como token
   muestreado. Se repitieron **todas** las condiciones con 0.1.1 (la pasada anterior quedó en `_superseded_v0.1.0`).
2. **torchaudio/torchcodec de Colab** (cu128) rompen `import transformers` tras instalar vLLM (torch cu130): la celda
   de instalación los desinstala.
3. **Logs de EngineCore perdidos**: vLLM configura logging al importarse; ahora se configura antes de cualquier
   `import vllm` y todo va a `logs/vllm.log`.
4. **Preflight bloqueó correctamente** una corrida de J64 en la que la GPU seguía ocupada por otro kernel (la
   referencia nativa de J no cabía): el notebook se detuvo antes de medir nada.
5. **Procesos `VLLM::EngineCore` huérfanos** retienen ~26 GB si se mata papermill: `launch_seq.sh` y `kill_gpu.sh`
   liberan la GPU por PID.
6. **CLI de Colab:** conexiones frágiles mientras la VM compila (reintentos añadidos); el CLI 0.7.0 dio por perdida
   una sesión viva al expirar el token (~1 h) y dejó una VM huérfana facturando (liberada vía API); actualizado a 0.7.4.
7. **Comportamiento del modelo (no es bug; no se ajusta nada sobre el test):** con el prompt del protocolo, el 0.8B a
   menudo omite la línea `FINAL:`, escribe LaTeX/líneas en blanco o degenera en conteos (`11\n12\n13...`); B lo sigue
   bien. Es esperable que G_SINGLE e híbridos acumulen `eos_without_final`/`truncated`. El protocolo prohíbe cambiar
   el prompt por resultados; se reportará como categoría de fallo.
8. **Banda sintética `xlong` del notebook 08** (300 líneas) superaba el tope de 64 rondas de JSTEP (`max_rounds`).
   Rediseñada en v0.1.2 (5/15/30/60 líneas "k squared is k*k"); cambio sólo del notebook 08, validado con test local.

## Costo del smoke y estimación de la corrida real

* Smoke (incluida la depuración): ~2.9 h de A100 + ~0.5 h de L4 + minutos de CPU ≈ **16–18 CU**.
* Corrida mínima (100 problemas, semilla 17 + greedy), estimación a partir del smoke: inferencia de 2–15 s por
  problema según condición y longitud → ~1–1.5 h de A100 para los 5 notebooks en secuencia (`launch_seq.sh`,
  ~4 min fijos por notebook) + ~10 min de L4 + ~30–40 min de A100 para el notebook 08 → **≈ 8–12 CU**.
  El perfil recomendado (3 semillas) ≈ 20–30 CU. Con salidas largas (truncados a 1024 tokens) el tiempo crece;
  revisa el ETA en `progress.json` tras los primeros 5 problemas.
