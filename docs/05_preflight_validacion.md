# Preflight y validaciones (§14 + enmienda A6)

Todas corren dentro de cada notebook de condición (`PREFLIGHT_MODE`): `full` (todas, siempre), `light` (reutiliza
las pesadas cacheadas en `/content/jev_llm/preflight_cache/<stack_hash>/` si existen en la misma VM y stack), `off`
(sólo las baratas). `ABORT_ON_PREFLIGHT_FAIL=True` detiene el notebook ante una bloqueante.

| Validación | Paso §14 | Bloqueante | Qué verifica | Resultado en el smoke (A100) |
|---|---|---|---|---|
| `lock` | 1–2 | sí | commit == SHA fijado; SHA-256 de **cada archivo** (pesos incluidos) vs LFS del Hub y `SHA256SUMS` de JevK5; arquitectura; parámetros lingüísticos desde cabeceras | ok; P_G=0.752B, P_J=8.954B |
| `parser_selftest` | 3 | sí | fracciones, decimales, `2/4`, menos Unicode, `1/0`, texto extra, FINAL parcial, FINAL en EOS | 16/16 |
| `chat_template` | 3 | sí | plantilla publicada renderiza `<think>\n\n</think>\n\n` con `enable_thinking=False`; texto y hash guardados | ok (G y B) |
| `prompt_length` | 3 | sí | prompts renderizados de los problemas seleccionados ≤ 512 tokens | máx 150 (N=1) |
| `sampling_semantics` | 3 | sí | `top_k=0` = desactivado en vLLM; semillas < 2⁶³ | ok |
| `selector_native_reference` | 5 | sí | runtime JevK5 publicado (subproceso) sobre 8 decisiones sintéticas (cortas, a mitad de frase, duplicadas, ~3k tokens); IDs idénticos a `prompt_text()` | ok, T=1.316 |
| `selector_equivalence` | 5 | sí | vLLM vs nativo con las mismas IDs: argmax (tolerancia 2 ulps bf16), |Δp| | 0 discrepancias; |Δp| máx 0.022 |
| `continuity` | 4 | no | continuar desde prefijos de 1/63/64/65/127/128 tokens reproduce la continuación greedy de una sola petición | 100 % |
| `batching_audit` | 6 | no | 4 peticiones secuenciales vs en batch (10 prefijos dev, 64 tokens) | 13.6 s vs 6.3 s → ×2.17 |
| `stress` | 6 | sí | 4 (o 1) generaciones de longitud máxima con `ignore_eos`; entrada de J cercana a 16,384 tokens; logits finitos; pico NVML | ok; J 16,232 tokens; pico 28.6 GiB |
| `eager_vs_graph` | 8.3 | no | tokens greedy con CUDA graphs vs motor eager (3 prompts × 64) | 100 % |
| `r4_memory` (opcional) | 8.3 | no | memoria con 4 copias de G + J residentes (sin corridas) | no ejecutado (opcional) |

Nota: continuidad, batching y eager/graph son diagnósticos (no bloqueantes) porque bf16 y el orden de batch pueden
cambiar legítimamente tokens; sus números se publican. El muestreo de vLLM no es invariante al batch: misma semilla
no garantiza la misma salida entre GPUs/stacks (§5.2).
