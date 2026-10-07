# Dataset (ficha técnica)

Construido por `data/build_dataset.py` (determinista, semilla `20260927`). Reproducir: `python data/build_dataset.py`.

## Fuentes (revisiones fijadas)

| Fuente | Repo / archivo | Revisión | Uso |
|---|---|---|---|
| GSM8K | `openai/gsm8k` · `main/test` | `740312add88f781978c0658806c59bc2815b9866` | aritmética multietapa; proporciones (palabras clave) |
| MATH | `EleutherAI/hendrycks_math` · test de algebra, number_theory, counting_and_probability, prealgebra | `21a5633873b6a120296cce3e2df9d5550074f4a3` | álgebra, teoría de números, conteo/probabilidad, proporciones |
| AMC12 2022–23 | `AI-MO/aimo-validation-amc` · train (83) | `69d78a4a2c840e82d69af6bc742bda09005f6316` | tramo difícil |
| AIME 2024–2025 | copias locales de `math_agent` (`data/raw/aime/*.jsonl`) | SHA-256 en `dataset_manifest.json` | tramo difícil |

## Diseño

* 5 dominios: `arithmetic`, `algebra`, `ratios_percentages`, `number_theory`, `counting_probability`.
* **Test (100):** por dominio 8 fáciles + 8 medios + 4 difíciles. **Dev (20):** por dominio 2 + 1 + 1. Disjuntos.
* **Dificultad (heurística, enmienda A5):**
  * GSM8K: 2–3 pasos `<<…>>` = fácil; 4–5 = medio; ≥ 7 = difícil (sólo como relleno de aritmética difícil).
  * MATH: nivel 1–2 = fácil; 3 = medio; 4–5 = difícil (relleno).
  * AMC/AIME: siempre difícil. Dominio por palabras clave + 16 correcciones manuales leyendo sólo el enunciado
    (`DOMAIN_OVERRIDES` en el script); se excluyen geometría, trigonometría, opciones múltiples y dinero.
* **Preferencia del tramo difícil:** competición primero; si no alcanza, relleno declarado (aritmética → GSM8K ≥ 7
  pasos; proporciones → MATH prealgebra L4–5). Resultado real en `dataset_manifest.json → counts`:
  aritmética difícil = 1 AMC + 3 GSM8K; proporciones difícil = 1 AMC + 3 MATH; el resto 4/4 competición.
* **Filtros:** respuesta exacta (entero, `\frac{p}{q}`, decimal finito) convertida a `Fraction`; |respuesta| ≤ 999;
  sin dinero (`$`, dollars, cents), sin "what percent", sin `[asy]`, unidades/grados, bases, "nearest", notación
  científica, intervalos, pares ordenados; sin duplicados aproximados (8-gramas); prompt renderizado ≤ 512 tokens
  con los tokenizers y plantillas de G y B (máx. observado: 292).
* **IDs opacos** (`jev-t-001`, `jev-d-001`) barajados: no revelan fuente, dominio ni dificultad.
* **Tipos de respuesta:** 111 enteras, 9 fracción/decimal.

## Archivos

| Archivo | Contenido | ¿En el bundle de inferencia? |
|---|---|---|
| `test_inputs.jsonl`, `dev_inputs.jsonl` | `id`, `problem`, `language` | sí |
| `schedule.json` | 10 bloques de 10 balanceados por dominio (semilla 20260927) y orden | sí |
| `test_gold.jsonl`, `dev_gold.jsonl` | `gold_numerator/denominator`, `gold_answer`, `reference_solution`, `domain`, `difficulty`, `difficulty_rule`, `template_group`, `source*`, `prompt_tokens`, `reviewed=false` | **no** (sólo análisis) |
| `blind_review_ids.json` | 20 IDs de test (4/dominio) para la revisión ciega del §11, elegidos antes de cualquier salida | no |
| `review_sheet.csv` | hoja para el segundo revisor (unicidad, enunciado, solución, dificultad) | no |
| `dataset_manifest.json` | fuentes, exclusiones por motivo, conteos, tipos | sí |
| `SHA256SUMS` | hashes de todos los anteriores | — |

## Limitaciones

Contaminación probable (benchmarks públicos); dificultad no validada por humanos; la aritmética "difícil" de GSM8K
es menos difícil que la de competición; el filtro de dinero/porcentajes sesga GSM8K hacia problemas sin moneda;
la clasificación de dominio de competición es por palabras clave con correcciones manuales.
