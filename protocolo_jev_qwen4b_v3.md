# Protocolo v3: tercera ronda, 500 problemas nuevos

Fecha de diseno: 2026-10-02. Estado: implementacion y preparacion; no existe un test v3 congelado ni resultados v3. El nombre identifica la tercera serie, NO el antiguo `protocolo_jev_qwen_v3.md` de la serie 0.8B. Las dos rondas anteriores permanecen cerradas e inmutables.

## 1. Objetivos y preregistro

Comparacion co-principal: una copia oficial Qwen3.5-4B produce cuatro soluciones completas; JevK5-9B selecciona una. Se contrasta contra el modelo DavidAU de aproximadamente 13B con decodificacion greedy, tanto en exactitud como en latencia controlada.

El estudio se motiva adaptativamente por v2. No se reutilizan sus tests como confirmatorios. No se escogen baselines por pregunta ni se seleccionan semillas, fuentes o reglas segun resultados. Quinientos problemas mejoran precision, pero no garantizan significancia ni generalizacion.

## 2. Modelos y perfiles

Revisiones completas y configuracion en `config/experiment_v3.json`:

| Papel | Modelo | Revision |
|---|---|---|
| G | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| J | alibiserikbay/JevK5-9B | d6521a18a86999190e9d775c915af3d6d6772fc4 |
| B | DavidAU/Qwen3.5-13B-GLM-4.7-Flash-DeepSeek-Polaris-Grande-Deep-Thinking | 717b561ac319a0ba9f9a0b1ed30dca2e1cc6c6fe |
| O | Qwen/Qwen3.5-9B oficial postentrenado | c202236235762e1c871ad0ccb60c8ee5ba337b9a |

BF16, texto solamente, sin cuantizacion, offload, MTP ni especulacion. CUDA graphs, vLLM 0.30.0. Runtime JevK5 v0.3.0 fijado a `6c6522fe5462a05fdb82bceeb0e8c624c11f1517`; calibracion 1.316. Paquete experimental aislado `jevlab.v3` version 0.3.0; el paquete historico conserva su version.

Una sola copia fisica de G y cuatro solicitudes simultaneas, no cuatro copias de pesos. Los parametros linguisticos se cuentan desde cabeceras, excluyendo vision/MTP y contando una sola vez pesos ligados. El control oficial 9B no iguala parametros desplegados del hibrido. Ninguna comparacion iguala FLOPs o entrenamiento.

## 3. Dataset y frescura

500 test, 50 piloto adicional, 20 dev originales reutilizados solo para depuracion. Idioma ingles. Test por dominio genuino revisado: 30 de nivel bajo, 40 medio, 30 alto SEGUN EL NIVEL DE FUENTE, para aritmetica multietapa, algebra, proporciones/porcentajes, teoria de numeros y conteo/probabilidad. Piloto por dominio: 3/4/3 de esos niveles. Codigos internos easy/medium/hard de sampling_tier no implican esa misma dificultad intrinseca.

Semilla de construccion 20261002, sin alternativas. Excluir la union de 260 problemas usados, duplicados exactos, aproximados y variantes numericas/plantillas. Registrar por separado candidatos identificables de las dos construcciones v2 rechazadas: no se confunden con problemas que tuvieron inferencia. No alterar fuentes previas ni recrear su freeze.

Solo problemas naturalmente existentes con identificador de origen, revision/hash, atribucion, licencia y respuesta de referencia. Fuentes iniciales: GSM8K TEST, MATH TEST de los cuatro dominios existentes, AMC/AIME heredados y ASDiv oficial. Fuentes adicionales pueden adquirirse antes de revisar/congelar, con adaptadores, pin y licencia registrados. No inventar problemas ni modificar sus numeros para llenar cupos. No usar GSM8K train como test independiente: el selector declara haberlo usado en entrenamiento. MATH train no esta autorizado como relleno automatico.

Corregir explicitamente las politicas de elegibilidad: `$...$` matematico no implica dinero; cantidades monetarias/porcentuales se permiten si se solicita un numero exacto en una unidad inequivoca. Gold entero, fraccion o decimal finito, magnitud <=10^9. No convertir porcentajes a otra escala ni modificar la cantidad preguntada. Hasta 1024 tokens renderizados bajo G/B/O. Excluir imagenes necesarias, elecciones multiples, pruebas abiertas, aproximaciones, respuestas multiples o simbolicas no soportadas. El formato de salida sigue siendo una unica linea FINAL numerica estricta.

La dificultad es provisional hasta revision agentica de estructura, razonamiento y cantidad de operaciones necesarias, nunca segun que modelo acierte. No cambiar dominios/dificultad para satisfacer cuotas. Mantener auditoria de clasificacion y de rechazos. Si un estrato no tiene suficientes candidatos validos, detener la congelacion y ampliar fuentes; NO publicar menos de 500 como si fueran 500.

Enmienda autorizada ANTES de inferencia: el usuario elige 'Balance por nivel de fuente' tras comprobar que solo 2 de los primeros 100 candidatos son intrinsecamente dificiles. Se conserva sampling_tier de la fuente/proxy original para el muestreo 30/40/30; difficulty conserva el juicio intrinseco real, sin cuotas forzadas. Publicar ambas distribuciones. No afirmar 150 problemas realmente dificiles. El dominio siempre es el genuino revisado. Las dos soluciones ciegas que validen matematica/unidades y concuerden en etiquetas intrinsecas pueden aprobar ordinariamente aunque difieran de la pista de fuente; los desacuerdos reales entre revisores requieren adjudicacion. No reescribir revisiones previas; las dos migraciones y sus evidencias se conservan.

## 4. Revision exclusivamente agentica

El usuario autoriza solo revision agentica, no humana. Dos revisores reales en contextos separados resuelven primero el enunciado sin gold, fuente, etiqueta provisional, cuotas o dictamen ajeno. Preservar la salida ciega antes de entregar la referencia a cada mismo revisor para una segunda fase de comparacion. Adjuntar hashes sin reescribir soluciones/veredictos.

Revisar los 550 seleccionados: claridad, unicidad, unidad, solucion, respuesta racional, dominio y dificultad. Discrepancias requieren adjudicacion documentada; no sobreescribir el gold solo por consenso de agentes. Los no resueltos se rechazan y sus veredictos se conservan. No fabricar IDs de ejecucion, evidencia o aprobaciones. El contrato verificable esta en `data/v3/REVIEW_CONTRACT.md`.

Enmienda preinferencia 2026-10-03: las etiquetas FUENTE son provisionales, no verdad absoluta. Para dos soluciones ciegas matematicamente correctas que discrepen solo en clasificacion o en una unidad inherentemente adimensional, un tercer agente real puede adjudicar etiquetas finales con razonamiento y hashes de los dictamenes originales. No puede cambiar gold, numeros, enunciados ni reparar soluciones ciegas falsas/no resueltas. Todos los vetos originales permanecen visibles y se distingue aceptacion ordinaria de aceptacion adjudicada. La regla se aplica uniformemente sin dar cuotas al adjudicador, antes de la seleccion; no se reclasifica para llenar cupos. Preparacion/politica previas se archivan por bytes antes de migrar; los paquetes/candidatos y revisiones originales no se reescriben.

`agent_reviewed=true`, `human_reviewed=false`, `reviewed=false` en gold test/piloto; dev reutilizado no se presenta como nuevamente revisado. Las revisiones agenticas pueden cometer errores correlacionados: esta limitacion no desaparece por tener dos agentes.

Tras revisar, reservar todos los estratos del test ANTES del piloto. Seleccionar deterministamente y congelar entradas, gold, politica, registro de exclusiones, revision y hashes. La inferencia solo recibe entradas ciegas y un manifiesto PUBLICO sin respuestas/evidencia de revision. Las soluciones y referencias quedan fuera de ese bundle.

## 5. Matriz y semillas

| Condicion | Semillas de calidad | Ejecuciones test |
|---|---|---:|
| G_SINGLE: 4B solo con muestreo | 17,29,43 | 1500 |
| G_GREEDY: 4B solo greedy | referencia 17, muestreo efectivo None | 500 |
| JFINAL: 4B + selector de cuatro soluciones completas | 17,29,43 | 1500 |
| B13: modelo ~13B con muestreo | 17,29,43 | 1500 |
| B13_GREEDY: modelo ~13B greedy, ancla | referencia 17, muestreo efectivo None | 500 |
| Q9_GREEDY: oficial 9B greedy | referencia 17, muestreo efectivo None | 500 |

Total 6000 inferencias de calidad. Bloques/pasos quedan fuera de esta serie. Greedy se ejecuta una vez por problema y se referencia frente a tres semillas, nunca se contabiliza como tres muestras independientes. Rechazar semillas repetidas.

Namespace fijo de propuestas `G4_FINAL4_V3`, ramas 0..3; identidad de pool separada de politica/condicion. Guardar propuestas durables ANTES de J, incluso parciales o si J falla. Los controles offline reutilizan exactamente ese pool: rama original 0 antes de permutacion, uniformidad esperada, pluralidad de respuestas racionales validas TERMINALES, oracle@4. Desempate de voto: menor rama original representada; sin votos validos: rama 0 con correccion normal. Nunca usar gold para votar ni asignar latencia de generacion a controles calculados offline.

## 6. Limites y validaciones

Temperatura 0.7, top_p0.9, top_k0 desactivado; penalizaciones neutras. Greedy temperatura0, seedNone. Pensamiento desactivado explicitamente en todos los generadores. Salida2048 por solucion, propuestas8192 por pool, contexto4096, prompt1024, Jentrada16384 sin truncamiento, timeout600s. A10040GB; caches G3/J4/B3/O3 GiB. Estas diferencias respecto a v2 impiden atribuir cambios entre rondas solo al dataset.

Bloqueantes: identidad de pesos/tokenizers/plantillas; parser; semantica de muestreo; longitudes; no pensamiento; full snapshot hashes; selector nativo versus vLLM; cuatro entradas maximas y salidas2048 con J16k residente; cada perfil standalone; ausencia de B/O con G+J durante recargas; cierre confirmado de procesos. Version desconocida o validacion omitida falla cerrado. Diagnosticos eager/graph y batching se publican; desacuerdo greedy<0.75 se amplia a20dev y se informa, no se confunde con cambio de respuestas.

No cuantizar, offload, cambiar hardware/caches/presupuesto/modelos para superar OOM sin enmienda explicita anterior al test. Prefijo cache desactivado. Manifestar opciones efectivas del motor, paquete y entorno.

## 7. Latencia controlada

Semilla de seleccion20261003: 100 preguntas del test, 20/dominio genuino revisado en cuotas6/8/6 POR NIVEL DE FUENTE; dificultad intrinseca real informada aparte. IDs congelados ANTES de inferir. JFINAL y B13_GREEDY: tres semillas, tres repeticiones temporales reales por semilla, 900 solicitudes por sistema y1800 en total. Las repeticiones no cambian semillas de propuesta; registrar hashes de salidas y posibles divergencias.

Una misma A100/UUID para todos los pares; bloques contrabalanceados. Descargas, cierre, cargas, probes y calentamiento quedan fuera de T_total y se informan aparte. B exclusivo frente a G+J exclusivo; descarga completa y verificacion de salida de procesos, NO sleep con pesos residentes. No incluir oro en entradas de latencia.

Para cada problema y sistema, mediana de sus nueve tiempos; razon B/J. Estimador principal de speedup: exp(media de log(razon)) sobre100 problemas. Reportar ademas p50/p95, tokens, truncamientos, timeout, formato/correccion del subconjunto y ventanas de recarga. No seleccionar solo aciertos ni pares rapidos. El ETA tecnico determina viabilidad antes de confirmar; si se necesita particionar entre GPUs, registrar una enmienda del diseno y del analisis antes de abrir el test, no afirmar luego un unico UUID.

## 8. Estadistica y conclusiones

Unidad: problema. Calidad: media de tres semillas del brazo estocastico por cada uno de500 problemas, frente a referencia greedy unica. Latencia:100 clusters; nunca1800 observaciones independientes. Bootstrap pareado estratificado por dominio,10000 remuestreos,semilla271828; mantener semillas/repeticiones juntas.

Dos objetivos co-principales: diferencia de exactitud JFINAL-B13_GREEDY y speedup de esa pareja. IC97.5% confirmatorios Bonferroni sobre2, IC95 descriptivos. Mas exacto si limite inferior>0; mas rapido si limite inferior de speedup>1; superioridad conjunta exige AMBOS. Interes practico: punto de calidad>=5pp y reduccion de tiempo>=10% (speedup>=1/0.9), informar si el IC completo lo apoya por separado. No interpretar ausencia de significancia como equivalencia.

Familia secundaria FIJA de cinco: JFINAL-G_SINGLE, JFINAL-voto compartido, G_GREEDY-Q9_GREEDY, JFINAL-Q9_GREEDY, G_SINGLE-B13. P-valores bilaterales aproximados de bootstrap centrado/agrupado, correccionHolm5. No aplicar McNemar a medias fraccionarias de3semillas ni a1500filas como independientes. Uniformidad y oracle son diagnosticos, no brazos desplegables.

Antes de inferencia test, analisis de potencia sobre escenarios de diferencias/discordancia relevantes; reportar incertidumbre si500 no alcanza. No incrementar N o seleccionar semillas segun p-valores. Correccion estricta offline de predicciones inmutables, auditor independiente numerico y recalculo independiente de intervalos/p-valores. Todos los fallos de modelo, truncamientos y timeouts cuentan como incorrectos; incidentes de plataforma se conservan y se recupera solo trabajo no terminado, sin borrar fallos ni reintentar por calidad. No excluir preguntas despues de resultados.

## 9. Piloto tecnico y presupuesto

50 preguntas por los seis perfiles:150por cada estocastico y50por cada greedy,600ejecuciones. Criterio tecnico: bloqueantes pasadas, cobertura completa y timeout/infraunion<5% por brazo y agregado; NO criterio de ganar al baseline. Publicar formato/tokens/tiempo/memoria y ETA de test/latencia sin ajustes de calidad.

El go tecnico NO autoriza presupuesto confirmatorio. Requerir autorizacion posterior del usuario, ligada al hash del config y al informe piloto. CU/costo monetario no se inventan: distinguir tiempo observado, extrapolacion y facturacion no disponible. Particiones y deadlines se fijan antes del test.

Autorizacion explicita del usuario 2026-10-03: ejecutar hasta resultados con TOPE TOTAL de 25 horas A100 acumuladas, incluyendo smoke, piloto, test y latencia, no 25 horas adicionales tras piloto. La confirmacion ligada al informe/config se materializa despues de verificar go tecnico y ETA dentro del saldo, bajo esta autorizacion condicional ya otorgada. Un ledger estable por experimento conserva gasto y reservas aunque cambie el formato de un JSON. Al agotar saldo se preservan artefactos y se cierran VMs propias; no se amplia el techo sin nueva autorizacion. No equivale a conocer CU/costo facturado.

## 10. Operacion Colab

Maximo2asignaciones simultaneas, todasA10040GB; una GPU en latencia. CLI aislado por sesion, wrapper ejecutable y COLAB_SESSION_CONFIG; Python del CLI para APIs/guard/cierre. Guard WebSocket15s/backend20s, monitor25s y controles2/5/10/30min. Recuperacion180s acotada; transportetimeout no significa notebook muerto. Operador duradero, hostWSL y coordinador independientes del turno del agente. No confiar en daemon automatico del CLI0.7.4.

Descargar checkpoints, verificar hashes/cobertura/notebooks/ZIPs, liberar inmediatamente cuando termine y confirmar ausencia del endpoint en servidor. Nunca detener sesiones ajenas ni repetir trabajo remoto vivo. Una perdida real de VM requiere preservacion, identificacion del trabajo incompleto y registro de recuperacion. No modifica semillas/perfiles. Deadline inicial4h por VM, revisado con ETA del piloto antes de confirmar; sin extender cargos silenciosamente. Al terminar, cero asignaciones propias; las externas se declaran y no se tocan.

## 11. Implementacion y entregables

Namespaces separados: `data/v3`, `config/experiment_v3.json`, `src/jevlab/v3`, `notebooks/v3`, `dist/v3`, `results/v3`. No sobrescribir el antiguo protocolo v3 ni v2, sus datos, notebooks, bundles o resultados. Pipeline: adquisicion/revision -> freeze -> tests/bundles -> smoke dev -> piloto tecnico -> autorizacion de presupuesto ->6000calidad ->1800latencia ->CPUanalisis/auditoria ->cierre.

Versiones, politicas y schemas explicitos. Verificador exacto de500IDs/3semillas o referencia500,1800mediciones reales y pools con ramas completas o fallos explicitamente contabilizados. Manifiesto publico sin gold ligado por hashes; evidencia detallada de revision separada en analisis. Informe con nombres descriptivos y flags de revision humana false.

## 12. Limitaciones obligatorias

Motivacion adaptativa; solo una tripleta principal y un control oficial; benchmarks publicos y posible contaminacion; fuentes/licencias y dificultad heterogeneas; revision agentica sin humana; cuatro propuestas consumen mas computo; distintas politicas de muestreo; latencia condicionada a motores calientes; efectos de batch/precision; datasets y presupuestos distintos entre rondas. No afirmar conclusiones para todos los modelos ni todos los problemas matematicos.

## Registro de cambios

2026-10-02: protocolo de tercera serie registrado antes de test. Usuario fija500test+50piloto,controloficial9B,solo revisionagentica y autoriza comenzar implementacion. Estado de capacidad/revision en `data/v3/PREPARATION_REPORT.md`; datos no congelados hasta cumplir requisitos reales.

2026-10-03: usuario requiere ejecutar todo hasta resultados y autoriza 25 horas A100 totales. Se implementa adjudicacion de clasificacion/unidades con evidencia preservada antes de cualquier inferencia y se endurecen presupuesto acumulativo e integridad de artefactos. Hipotesis, semillas, pesos, cuotas y presupuestos de tokens no cambian.

Enmienda de muestreo preinferencia: el usuario autoriza balance por nivel de fuente, manteniendo500test+50piloto y100por dominio. sampling_tier y dificultad intrinseca revisada se separan; se publican ambas sin afirmar que nivel alto equivale a dificil real. Modelos, hipotesis, semillas, tokens y presupuesto GPU no cambian.
