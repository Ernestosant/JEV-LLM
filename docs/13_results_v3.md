# Resultados de la tercera ronda (v3)

## Alcance

La tercera ronda termino: es una **pequena prueba de concepto**, no un avance definitivo ni una demostracion general de superioridad de arquitecturas. Bajo el protocolo fijado, el hibrido obtuvo mayor exactitud que el baseline principal; no demostro ser mas rapido ni superar al voto de las mismas cuatro propuestas.

Este informe describe resultados terminados sin reescribir el [protocolo preregistrado](../protocolo_jev_qwen4b_v3.md), los datos ni los resultados originales. Los estados de preparacion en [12_implementation_v3.md](12_implementation_v3.md) son fotografias historicas, no el estado final. Las [limitaciones y el trabajo futuro](14_limitations_and_future_work.md) forman parte de la interpretacion, no son salvedades opcionales.

## Diseno y cobertura

| Componente | Cobertura efectiva | Unidad de analisis |
|---|---|---|
| Calidad confirmatoria | 6000 casos fisicos, exactamente 500 IDs de problemas | 500 problemas, no 6000 observaciones independientes |
| Piloto tecnico separado | 50 IDs adicionales, 600 casos | No entra en la calidad confirmatoria |
| Latencia controlada | 1800 mediciones reales sobre 100 IDs del test | 100 problemas, nueve tiempos por sistema y problema |

Los brazos estocasticos usan las semillas 17, 29 y 43; cada referencia greedy se ejecuta una vez por problema. JFINAL genera cuatro soluciones completas mediante solicitudes simultaneas a **una sola copia fisica** de Qwen3.5-4B y usa JevK5-9B para seleccionar una. No son cuatro copias de pesos. Los 1500 pools compartidos contienen 6000 posiciones de candidatos registradas y estan completos; no deben confundirse con los 6000 casos de la matriz de calidad.

La comparacion principal, fijada antes del test, es JFINAL frente a B13_GREEDY. B13 designa el modelo personalizado DavidAU de aproximadamente 13B, no un Qwen oficial de ese tamano. Q9_GREEDY es el control oficial Qwen3.5-9B; tampoco iguala los parametros desplegados del hibrido. Identidades y revisiones completas constan en el protocolo.

Se usaron BF16, vLLM 0.30.0, pensamiento desactivado y hasta 2048 tokens por solucion. La latencia pareada usa la misma A100 de 40 GB para ambos sistemas: tres semillas por tres repeticiones temporales, 900 mediciones por sistema. Se toma la mediana de sus nueve tiempos por problema. Carga, descarga, probes y calentamiento quedan fuera del cronometro de caso; no es una medicion de servicio frio ni de costo integral de despliegue.

El consumo acumulado registrado fue **56752.900384 segundos GPU, equivalentes a 15.76469455 horas A100**, dentro del techo total de 25 horas A100 y del maximo de dos asignaciones simultaneas. Es tiempo acumulado de recursos, no duracion de pared, factura monetaria ni CU facturadas.

## Exactitud

| Condicion | Casos fisicos | Problemas | Exactitud (%) |
|---|---:|---:|---:|
| G_SINGLE: 4B con muestreo | 1500 | 500 | 91.0 |
| G_GREEDY: 4B greedy | 500 | 500 | 92.2 |
| JFINAL: cuatro propuestas + selector | 1500 | 500 | 94.7333 |
| B13: modelo personalizado con muestreo | 1500 | 500 | 84.2 |
| B13_GREEDY: baseline principal | 500 | 500 | 84.0 |
| Q9_GREEDY: 9B oficial greedy | 500 | 500 | 89.2 |

En brazos estocasticos, la exactitud se resume primero por problema sobre tres semillas. Greedy no se triplica como si fueran tres muestras independientes. Fallos de modelo y truncamientos cuentan como incorrectos; no se excluyen para favorecer el resultado.

## Efectos coprincipales

Bootstrap pareado por cluster de problema, estratificado por dominio, con 10000 remuestreos y semilla 271828. Los dos intervalos confirmatorios del 97.5% aplican Bonferroni; son aproximados.

| Efecto | Estimacion | IC 97.5% Bonferroni | Lectura |
|---|---:|---|---|
| Exactitud JFINAL - B13_GREEDY | +10.7333 pp | [7.5333, 14.1333] pp | Evidencia de mayor exactitud en este test |
| Speedup geometrico pareado B13_GREEDY / JFINAL | 1.00192494 | [0.88386781, 1.12642550] | No demuestra mayor velocidad |

Un speedup mayor que 1 favorece a JFINAL. El intervalo de latencia incluye 1: no se demuestra superioridad en latencia ni **superioridad conjunta**. Tampoco se demuestra una reduccion de tiempo del 10%; ni siquiera la estimacion puntual alcanza el umbral `1 / 0.9`. La diferencia de calidad supera el umbral practico de 5 pp, incluido su limite inferior confirmatorio. Ausencia de significancia en latencia no demuestra equivalencia.

## Controles compartidos

| Control offline sobre el mismo pool | Exactitud (%) |
|---|---:|
| Rama original 0 fija, antes de permutacion | 90.7333 |
| Voto/pluralidad de cuatro respuestas validas (VOTE4SHARED) | 94.6 |
| JFINAL | 94.7333 |

El voto usa respuestas numericas terminales validas, sin gold, y desempata por la menor rama original. JFINAL - voto compartido es **+0.1333 pp**, con IC95 descriptivo **[-0.5333, 0.8] pp** y p ajustado Holm **0.7653**. Por tanto, **no hay evidencia de ventaja del selector sobre este voto**. La mejora respecto a una respuesta unica no identifica por si sola una contribucion propia de la critica/seleccion aprendida.

Los controles reutilizan exactamente los candidatos de JFINAL, no constituyen ejecuciones independientes. No se les asigno latencia: estas exactitudes no prueban ahorro de tiempo o costo. Oracle@4 (96.6%) es un diagnostico offline que usa gold, no un sistema desplegable.

## Anomalia de tamano y errores

Que G_GREEDY alcance 92.2%, B13_GREEDY 84.0% y Q9_GREEDY 89.2% no prueba que los modelos pequenos sean generalmente mejores. El baseline DavidAU es una construccion personalizada merge/upscale con componentes de modelos de 9B y destilaciones; no es un control oficial de la misma familia emparejado por tamano o entrenamiento. Las comparaciones tampoco igualan FLOPs, entrenamiento ni capacidad maxima del fabricante.

El contraste secundario 4B greedy - 9B oficial greedy es +3 pp, pero su p ajustado Holm es 0.087991: no se rechaza a nivel 0.05 en la familia secundaria fijada de cinco contrastes. No debe rescatarse como hallazgo confirmatorio por su intervalo descriptivo sin correccion.

| Errores standalone greedy | Numero incorrecto con FINAL valido | FINAL invalido sin truncamiento | Truncamiento | Total incorrecto |
|---|---:|---:|---:|---:|
| 4B | 24 | 0 | 15 | 39 |
| DavidAU ~13B | 55 | 5 | 20 | 80 |

La brecha no se explica solo por truncamientos: tambien hay mas respuestas numericas incorrectas del baseline. Esto no determina la causa; presupuesto, modo, entrenamiento y backend requieren controles separados. Cambiar ahora a thinking, aumentar tokens o ajustar muestreo no reemplazaria esta comparacion: seria un estudio nuevo.

## Auditoria y limites

El parser estricto independiente reviso las **6000 predicciones y obtuvo cero desacuerdos**. Es evidencia de concordancia de extraccion/correccion numerica, no de que todos los gold sean verdaderos ni de equivalencia de los backends.

El test contiene 394 problemas intrinsecamente faciles, 96 medios y 10 dificiles; 340 proceden de MATH. El balance 150/200/150 es por nivel de fuente, no por dificultad real. Los 550 problemas de test/piloto tuvieron revision agentica, no humana. Benchmarks publicos, contaminacion de entrenamiento y duplicados semanticos no excluidos limitan la generalizacion.

La validacion de G incluyo solo tres prefijos cortos eager frente a graph. B13 y Q9 omitieron explicitamente esa comprobacion por memoria. El estres de continuidad paso, pero no prueba equivalencia de exactitud nativa. La recuperacion J1 dispone de evidencia local byte-exact con `sourceSha` y manifiesto, no de SHA remoto de transporte persistido. En LATV3 la vinculacion de memoria a la fase H2 se acredita por una proyeccion en memoria; el resumen bruto global B3 no fue modificado. Estas brechas no se convierten en validaciones completas por haber terminado la ejecucion.

## Resumenes publicos

Enlaces destinados a la seleccion curada de `public_results/v3`, no a artefactos operativos:

- [Resumen final en espanol](../public_results/v3/resumen_es.md).
- [Informe analitico final](../public_results/v3/report.md).
- [Comparacion coprincipal estructurada](../public_results/v3/primary_comparison.json).

La publicacion de esos resumenes no implica publicacion, aprobacion operativa ni permiso de redistribucion del bundle bruto, dataset completo o logs de nube. No se incluyen identificadores de ejecucion, rutas personales ni secretos. La conclusion defendible sigue siendo local: **mejor exactitud que el baseline fijado, sin ventaja demostrada de latencia ni del selector frente al voto compartido**.
