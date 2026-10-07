# Limitaciones y trabajo futuro

## Interpretacion

JEV-LLM v3 es una **pequena prueba de concepto terminada**, no una demostracion definitiva de un nuevo paradigma. Los [resultados v3](13_results_v3.md) respaldan mayor exactitud de JFINAL frente al baseline personalizado greedy bajo condiciones fijadas; no respaldan superioridad en latencia, superioridad conjunta ni ventaja del selector frente al voto de sus mismos candidatos.

Estas limitaciones aclaran el alcance de los resultados, sin modificar el [protocolo original](../protocolo_jev_qwen4b_v3.md), datos congelados, predicciones o analisis originales. La motivacion fue adaptativa tras rondas previas; el piloto y las rondas historicas no entran como evidencia confirmatoria de este test.

## Validez externa

- **Dificultad limitada:** 394/500 problemas son intrinsecamente faciles, 96 medios y solo 10 dificiles. Los 150 problemas de nivel alto de fuente no equivalen a 150 problemas realmente dificiles. En latencia hay 78 faciles, 21 medios y uno dificil.
- **Concentracion de fuentes:** 340/500 problemas proceden de MATH. Cinco dominios con 100 problemas cada uno no garantizan diversidad equivalente de estilos, procedencia o capacidades matematicas.
- **Fuentes publicas:** excluir identidades previamente usadas y aplicar deduplicacion no excluye contaminacion del entrenamiento ni todos los duplicados semanticos/plantillas. Un test nuevo para esta serie no es necesariamente nuevo para los modelos.
- **Referencias sin revision humana:** test y piloto fueron revisados por agentes (`agent_reviewed=true`, `human_reviewed=false`). Dos soluciones ciegas y adjudicacion documentada reducen algunas dudas, pero pueden compartir errores. Cero desacuerdos entre parsers sobre 6000 predicciones no establece verdad matematica del gold.
- **Tarea estrecha:** problemas en ingles con respuesta numerica exacta y formato FINAL estricto. No hay evidencia general para pruebas abiertas, respuestas simbolicas, otros idiomas o tareas no matematicas.

## Modelos y presupuesto

El baseline DavidAU ~13B es un modelo personalizado merge/upscale con componentes 9B y destilaciones, no un oficial de mayor tamano emparejado con el 4B por familia y entrenamiento. JevK5-9B tiene entrenamiento distinto, y el control Qwen3.5-9B oficial no iguala los parametros del hibrido. Contar parametros no iguala computo: cuatro propuestas, selector, batching y entrenamiento difieren.

El 4B greedy obtuvo +3 pp respecto al 9B oficial greedy, pero Holm p=0.087991 no permite rechazar el contraste secundario a 0.05. Esto no establece una ley inversa de escalado. El peor resultado del ~13B tampoco se reduce a truncamientos: sus errores standalone fueron 55 numeros incorrectos, cinco FINAL invalidos y 20 truncamientos, frente a 24, cero y 15 en el 4B.

Todos los generadores tuvieron pensamiento desactivado y un limite de 2048 tokens por solucion. Esa igualdad define esta comparacion, pero no mide la mejor capacidad de cada fabricante/modelo. Modos thinking, presupuestos mayores y muestreo apropiado por modelo deben estudiarse de nuevo, con reglas y validaciones previas, no usarse para cambiar post hoc al ganador del experimento original.

## Atribucion del selector

El voto offline sobre los mismos cuatro candidatos obtuvo 94.6%, frente a 94.7333% de JFINAL. Diferencia +0.1333 pp, IC95 [-0.5333, 0.8], Holm p=0.7653: **no hay ventaja demostrada del selector**. No demostrar diferencia tampoco prueba equivalencia. La rama fija compartida obtuvo 90.7333%; compararla con JFINAL muestra el beneficio agregado de seleccionar entre candidatos, no identifica critica aprendida frente a mayoria.

Estos controles son condicionados al pool de JFINAL; no tienen latencia medida ni son replicas independientes. Oracle@4 usa gold y no es desplegable. Un estudio causal de la critica necesita separar generacion, voto, seleccion y revision, con identico pool y contabilidad de sus costos.

## Backend y trazabilidad

| Brecha observada | Evidencia disponible | Lo que no permite afirmar |
|---|---|---|
| G: eager frente a graph sobre tres prefijos cortos | Comprobacion acotada de G | Equivalencia de respuestas completas en toda la distribucion |
| B13 y Q9: comprobacion eager/native omitida por memoria | Omision explicita; estres de continuidad pasado | Equivalencia de exactitud con ejecucion nativa independiente |
| Recuperacion J1 | Recuperacion local byte-exact con `sourceSha` y compromisos de manifiesto | Verificacion criptografica de transporte remoto: su SHA no quedo persistido |
| LATV3: memoria de fase H2 | Evidencia real de H2 vinculada por proyeccion solo en memoria | Que el resumen bruto global B3 ya represente H2 o haya sido corregido |

El estres de memoria/continuidad verifica que una ruta puede ejecutarse, no que preserve exactitud nativa. La concordancia del parser verifica otra capa distinta. No deben agruparse como una unica aprobacion total.

La evidencia J1 sostiene recuperacion local, pero falta una prueba remota de transporte durable; no se puede reconstruir retrospectivamente como si hubiera quedado registrada. En LATV3 se debe mantener la distincion entre evidencia por fase y resumen global sin mutar el artefacto bruto. Son brechas de auditoria que deben declararse, no ocultarse ni corregirse reescribiendo historia.

## Latencia y estadistica

El speedup geometrico pareado fue 1.00192494, IC97.5% [0.88386781, 1.12642550]. No demuestra superioridad en latencia ni reduccion del 10%, y no demuestra equivalencia. La conclusion conjunta requiere que ambos objetivos coprincipales superen sus umbrales; aqui solo la calidad lo hace.

Las 1800 mediciones reales son nueve por sistema y problema, en una misma A100 de 40 GB; la unidad inferencial son 100 clusters de problema. Las 6000 predicciones de calidad tampoco equivalen a 6000 problemas independientes. Bootstrap e intervalos son aproximados y la familia secundaria usa Holm sobre cinco contrastes fijados, no seleccionados segun resultados.

La latencia es condicional a motores calientes. Carga, descarga, probes y calentamiento quedan fuera del reloj por caso. Tiempos de brazos ejecutados en corridas distintas son descriptivos, no sustituyen el estudio pareado. Una unica campana no caracteriza toda la variabilidad de runtime. Las 15.76469455 horas A100 acumuladas (56752.900384 segundos) no son ahorro financiero demostrado ni facturacion; el techo fue 25 horas con un maximo de dos asignaciones simultaneas.

## Prioridades futuras

Cada ampliacion debe ser un estudio separado, con preregistro, presupuesto y criterios de exclusion previos; no un cambio retrospectivo del protocolo original.

1. **Validacion independiente native/eager:** elegir y fijar un subconjunto antes de ver sus resultados; comparar respuestas completas, tokens y decisiones bajo motores nativo, eager y graph, incluyendo B13 y Q9. Si memoria impide una ruta, declarar el limite o disenar otra prueba separada, no darla por pasada con estres de continuidad.
2. **Controles oficiales emparejados:** comparar tamanos de una misma familia oficial y modos equivalentes, con revisiones completas y plantillas verificadas. Separar el efecto de tamano del de merge/upscale, distilacion y entrenamiento; evaluar tambien modos recomendados por fabricante en una matriz nueva.
3. **Presupuestos y muestreo justificados:** estudiar curvas calidad/costo con mas tokens y thinking cuando corresponda, y ajustes de muestreo sensatos por modelo validados en desarrollo/piloto independiente. No escoger temperaturas arbitrarias ni afinarlas sobre el test final.
4. **Holdout realmente dificil y novedoso:** incorporar suficientes problemas genuinamente dificiles y nuevas fuentes, revision humana independiente de referencias, unidades y soluciones, y un estudio explicito de contaminacion y duplicados semanticos. Reportar incertidumbre de procedencia; no prometer que frescura elimina toda contaminacion.
5. **Critica frente a mayoria:** aislar voto, seleccion aprendida y critica/revision sobre candidatos compartidos, con desempates preregistrados y sin gold en decisiones. Medir tambien latencia, tokens, memoria y costo de cada politica; incluir comparaciones a presupuesto de computo comparable.
6. **Variabilidad de runtime:** repetir bloques contrabalanceados en el mismo hardware con condiciones documentadas y separacion de tiempos calientes, arranque y recargas. Mantener repeticiones dentro del cluster de problema; no convertir repeticiones temporales en nuevos problemas.
7. **Mayor potencia y replica:** dimensionar una muestra mayor segun efectos y discordancias relevantes, antes de inferir; preregistrar analisis pareado por cluster y multiplicidad corregida. No aumentar N, cambiar semillas o parar al obtener un p favorable. Replicar con otra seleccion independiente antes de generalizar.
8. **Auditoria durable y publicacion acotada:** registrar en futuras recolecciones SHA remoto y local persistidos, y evidencia de memoria explicitamente indexada por fase. Para esta serie, conservar y explicar J1/LATV3 sin mutar originales. Publicar solo resumenes curados y revisar licencias y privacidad antes de cualquier ampliacion de materiales.

## Publicacion y conclusion

Consultar el [resumen publico final](../public_results/v3/resumen_es.md), el [informe analitico](../public_results/v3/report.md) y la [comparacion coprincipal](../public_results/v3/primary_comparison.json), dentro de la seleccion curada prevista para `public_results/v3`. Estos enlaces no implican permiso para redistribuir el dataset completo, logs de nube ni el bundle bruto. Tampoco representan una aprobacion operativa general. La documentacion publica no necesita rutas personales, endpoints, UUIDs de GPU, identificadores de host ni valores de entorno o credenciales.

El resultado util de esta pequena prueba es una mejora de exactitud acotada y una lista concreta de controles pendientes. **No hay evidencia de que el selector supere al voto compartido, de que el hibrido sea mas rapido, ni de que modelos pequenos sean universalmente superiores.** Las prioridades futuras buscan probar esas preguntas por separado, no transformar el resultado original en un avance definitivo.
