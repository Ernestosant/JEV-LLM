# Supervision Y Cierre De Colab

El ciclo de vida es independiente del monitoreo experimental a los 2/5/10/30 minutos.

- Pings WebSocket: cada 15 segundos, conectando SOLO al kernel existente del CLI.
- Comprobacion de la asignacion en el servidor: cada 20 segundos, con timeout de red.
- Renovacion de conexion con credenciales de proxy actualizadas: antes de los 50 minutos.
- Cancelacion del guard: comprobada cada segundo mientras espera el siguiente control.
- Cierre: unassign de la asignacion exacta, seguido de list_assignments para confirmar su
  ausencia. Hasta tres intentos limitados; una respuesta de red fallida NO demuestra cierre.
- Registro: results/lifecycle/<SESSION>/session_lifecycle.jsonl, sin tokens ni credenciales.

## Uso

Los scripts launch.sh y launch_seq.sh inician el guard antes de subir/ejecutar el notebook.
stop.sh libera la VM y comprueba el servidor. Si una VM nueva falla durante el lanzamiento,
un trap intenta liberarla; si no puede confirmar el cierre, devuelve error y una advertencia.

Para herramientas que aislan el estado del CLI, exporta COLAB_SESSION_CONFIG con la misma
ruta que usa tu ejecutable wrapper. El valor por defecto es ~/.config/colab-cli/sessions.json.
COLAB_PYTHON permite indicar el Python del entorno donde esta instalado el CLI.

No se crean kernels de repuesto ni se ejecutan celdas ficticias para mantener actividad.
El guard espera hasta que el lanzamiento haya creado el kernel del CLI, comprueba la
asignacion y conserva su conexion. No garantiza disponibilidad ni evita cuotas o limites
de Colab. Un fallo de ping/observacion se registra como estado desconocido.

## Verificacion Real

Prueba de integracion 2026-09-29 en una VM temporal CPU, sin modelos:

- Conexion WebSocket confirmada y cinco observaciones de salud espaciadas 20 segundos.
- Pings configurados a 15 segundos en el cliente WebSocket instalado.
- Cancelacion a las 22:34:13 UTC; guard detenido a las 22:34:14 UTC.
- Asignacion ausente confirmada en el primer intento, a las 22:34:14 UTC.
- Proceso del guard ausente y listado global de sesiones vacio despues del cierre.
- Evidencia: results/first_two_20260929/guard_check/session_lifecycle.jsonl.

El primer intento de usar el endpoint HTTP TFE keep-alive devolvio HTTP400 en esta cuenta;
se descarto esa implementacion y se verifico la conexion WebSocket del CLI. No se presenta
ese intento como un keep-alive satisfactorio.

Los experimentos G_SINGLE/B13 ya habian terminado y sus VMs estaban liberadas antes de
la prueba de integracion del nuevo guard; no se alteraron ni se repitieron sus resultados.
