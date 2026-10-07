# Implementacion de la tercera ronda

> **Nota de cierre:** esta página conserva la fotografía de preparación anterior al test. La ronda v3 ya terminó: 500 problemas, 6.000 casos de calidad y 1.800 mediciones de latencia, con análisis real y recursos liberados. Ver [resultados finales](13_results_v3.md) y [limitaciones y trabajo futuro](14_limitations_and_future_work.md). Las tareas y límites operativos de esta fotografía histórica no son instrucciones vigentes de publicación o ejecución.

## Estado histórico de preparación

Protocolo y software v3 implementados localmente; preparacion y revision de datos en curso. **No hay un test v3 congelado, resultados experimentales v3 ni GPU Colab asignada por esta implementacion.**

La tercera ronda esta definida en `protocolo_jev_qwen4b_v3.md` y `config/experiment_v3.json`: 500 test nuevos, 50 piloto adicional, 100 preguntas del test reservadas para latencia, tres semillas y revision exclusivamente agentica. No se modifica la serie historica del archivo `protocolo_jev_qwen_v3.md`.

## Implementado

- Namespace `src/jevlab/v3`, version 0.3.0. El runtime historico conserva 0.2.1 y sus bundles permanecen intactos.
- Seis condiciones, incluidos 4B greedy y Qwen3.5-9B oficial greedy; tres semillas distintas para los brazos estocasticos y una referencia por pregunta para greedy.
- Pools de cuatro propuestas durables antes de llamar a J, namespace de semillas separado de la politica y trazas de fallos/terminacion. Los controles de rama 0, pluralidad, uniformidad esperada y oracle reutilizan propuestas sin fingir nueva inferencia o latencia.
- Limites 2048/8192, maximo 1024 tokens de entrada, estres sintetico cerca de ese maximo y selector cerca de 16k. Validaciones obligatorias aunque un formulario indique light/off; versiones desconocidas fallan cerrado.
- Estudio de latencia de 1800 solicitudes reales: JFINAL y modelo grande greedy, 100 preguntas, tres semillas y tres repeticiones, recarga exclusiva de fases y costos de carga fuera del cronometro.
- Analisis por cluster de problema, dos IC co-principales del 97.5%, IC95 descriptivos y cinco contrastes secundarios con bootstrap centrado y Holm. No tratar semillas/repeticiones como preguntas independientes.
- Notebooks separados en `notebooks/v3/`, un bundle exclusivamente de desarrollo y gates de datos, piloto tecnico, presupuesto y ETA antes de asignar recursos.
- Operador aislado con guard WS15s/backend20s, polling25s, recuperacion180s, limite4h y cierre confirmado. La confirmatoria requiere autorizacion real ligada a config y piloto; los controles de plan no crean dicha autorizacion.

## Preparacion real

El pool inicial fue ampliado con fuentes naturales y politicas semanticas globales antes de empezar revisiones. No se bajaron las cuotas ni se cambio la semilla. Hay **3422 candidatos provisionales**, cinco familias de fuentes y 137 pares de paquetes ciegos/referencia. Las 550 plazas propuestas tienen capacidad provisional, con 88 candidatos prioritarios de reserva; esto no equivale a 550 preguntas aprobadas.

Se excluyen 260 identidades usadas y 113 identidades adicionales de construcciones previas rechazadas. Registro y deduplicacion exacta, aproximada y de variantes numericas quedan en `data/v3/staging/`.

Fuentes nuevas adquiridas con revision/hash y licencia: ASDiv oficial (CC BY-NC 4.0) y OpenStax oficial (CC BY-NC-SA 4.0). Estos permisos son para uso no comercial con atribucion y, donde corresponde, ShareAlike; no se asume permiso para explotacion comercial. GSM8K train no fue adquirido ni usado. Las claves/procedencia de competicion heredadas siguen necesitando verificacion, no se presentan automaticamente como oficiales.

## Revision efectuada

Primeros cuatro paquetes: **100 candidatos** revisados por ocho ejecuciones reales de agentes, dos por paquete. Cada agente resolvio a ciegas, preservo su salida y luego comparo la referencia sin cambiar la respuesta ciega. Se registraron 400 eventos: 200 soluciones ciegas y 200 comparaciones.

Resultado bajo el contrato actual: **12 candidatos con doble aprobacion; 88 vetados**. Entre 82 rechazados por ambos hay 70 discrepancias de clasificacion y ocho desacuerdos de gold; los grupos se solapan. Otros seis tienen veto unilateral. Hay ademas casos no resueltos o con dudas de unidad. Estos datos NO prueban que 88 respuestas fuente sean erroneas: muchas discrepancias son de dominio o dificultad provisional.

Todos los dictamenes y sus identidades/hashes se preservan en `data/v3/staging/reviews/`; el contador vigente es `review_progress.json`. No se convierten falsos en verdaderos ni se rellenan soluciones para satisfacer un validador. La publicacion sigue bloqueada hasta 550 aprobaciones reales en las cuotas exactas. Adjudicar desacuerdos y/o revisar reservas es trabajo pendiente; no corresponde corregir etiquetas selectivamente para llenar cupos.

`data/v3/PREPARATION_REPORT.md` es la fotografia de la preparacion anterior a las revisiones. El estado posterior se consulta en el registro de revision, no se modifica retrospectivamente la preparacion sellada.

## Comandos locales

```powershell
python -B tools/build_notebooks_v3.py
python -B tools/make_bundle_v3.py --dev
python -B tools/colab/run_v3.py --plan --phase test
python -B -m pytest tests -q -p no:cacheprovider
```

El plan es offline y devuelve `blocked_dataset_review`. El bundle de desarrollo no contiene gold, test, piloto ni evidencia privada de los revisores. El constructor de bundles completos y el runtime no fabrican un sello de datos para sortear el bloqueo.

Verificacion local final de esta etapa: **1023 tests pasados**, cero fallos y 13 avisos de deprecacion existentes en el analisis Matplotlib historico. Los notebooks se regeneraron y el bundle dev se verifico tras fijar el hash del contrato de revision en la configuracion. Los tests usan motores/backends falsos; no prueban memoria o exactitud en CUDA real. Estado estructurado: `results/v3/implementation_status.json`.

Para registrar una revision realmente emitida:

```powershell
python -B tools/submit_reviews_v3.py --help
```

El registrador adjunta solo los hashes permitidos por el contrato, conserva todos los veredictos y comprueba idempotencia. No debe asignarse un ID ficticio de agente a una salida generada por otro procedimiento.

## Pendiente

1. Adjudicar discrepancias con evidencia y continuar revisando candidatos de reserva. Si las cuotas revisadas quedan cortas, planificar nuevas fuentes sin alterar datos previos ni resultados.
2. Completar y sellar exactamente 500 test, 50 piloto y 100 IDs de latencia, con politica y procedencia verificadas.
3. Construir bundles completos y volver a validar cobertura, politica, hashes y ausencia de gold.
4. Ejecutar smoke y piloto tecnico en Colab; CUDA real, memoria y equivalencia nativa v3 todavia no estan medidos.
5. Presentar presupuesto y particionamiento/ETA al usuario. No lanzar confirmatoria automaticamente por obtener go tecnico.
6. Ejecutar, analizar y auditar solo tras esa autorizacion, manteniendo maximo dos VMs y cierre confirmado.
