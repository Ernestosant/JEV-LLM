# Resultados agregados de la prueba de concepto v3

Esta carpeta contiene una seleccion publica de los resultados terminados: 500 problemas de test, 6000 casos fisicos de calidad y 1800 mediciones de latencia sobre 100 problemas. El piloto de 50 problemas y 600 casos se analiza separadamente.

- [Resumen en espanol](resumen_es.md): resultados principales y conclusiones verificadas.
- [Informe analitico](report.md): metodologia, composicion de la muestra y contrastes.
- [Tabla de exactitud](final_table.csv): seis brazos fisicos y controles derivados del mismo pool.
- [Comparacion coprincipal](primary_comparison.json): efectos de calidad y latencia, con intervalos.
- [Contrastes secundarios](secondary_comparisons.csv): familia fijada con ajuste Holm.
- [Controles compartidos agregados](shared_controls_summary.csv): voto y rama fija sobre el mismo pool.

Son agregados numericos y texto de informe. No son un dataset redistribuible, ni contienen predicciones individuales, respuestas de referencia, datos operativos, credenciales, variables de entorno, notebooks ejecutados o pesos.

Las tablas de controles compartidos reutilizan propuestas de JFINAL; **no agregan casos fisicos ni problemas independientes**. Oracle@4 usa gold y no es un metodo desplegable. Los tiempos de corridas de calidad distintas son descriptivos; solo el estudio LATENCY pareado sustenta la comparacion de velocidad.

Leer [resultados y alcance](../../docs/13_results_v3.md) y [limitaciones y trabajo futuro](../../docs/14_limitations_and_future_work.md) antes de generalizar.
