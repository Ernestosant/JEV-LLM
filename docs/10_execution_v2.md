# Serie v2: ejecucion y recuperacion

El protocolo vigente es `../protocolo_jev_qwen4b_v2.md`, incluida la enmienda de recarga por fase del apartado 8.1. El estado real se consulta en `results/v2/coordinator_reload_state.json`; este documento no sustituye los registros de ejecucion.

## Archivos congelados

- Test nuevo: 100 problemas; piloto: 40; desarrollo: 20 reutilizados. Semilla de construccion 20261001, semilla de inferencia 17. `data/v2/SHA256SUMS` sella los archivos.
- Configuracion vigente: `config/experiment_v2_reload.json`, codigo 0.2.1, BF16, A100 de 40 GB, una copia del generador Qwen3.5-4B y cuatro peticiones independientes por ronda hibrida.
- Notebooks: `notebooks/v2-reload/`; bundles: `dist/v2-reload/`. Inferencia sin gold; analisis separado con gold.
- `config/experiment_v2.json`, `notebooks/v2/`, `dist/v2/` y los resultados de 0.2.0 permanecen como evidencia historica. No se regeneran ni se reanudan bajo el codigo nuevo.
- Revision humana omitida por autorizacion del usuario: `reviewed=false`. No presentar los datos como revisados.

## Continuidad

Cada VM tiene configuracion de CLI, wrapper ejecutable, operador duradero y guard propios. El guard mantiene WebSocket cada 15 s y consulta la asignacion cada 20 s. El operador sondea cada 25 s, descarga checkpoints durante la corrida, ejecuta controles de progreso y limita la VM a cuatro horas.

Un error de transporte no prueba que el notebook haya muerto. La recuperacion del guard esta acotada a 180 s; no reinicia el kernel ni la inferencia. El cierre solo se considera completo tras confirmar la ausencia del endpoint en el servidor.

El coordinador se ejecuta en un proceso WSL independiente del terminal del agente, con limite global de 12 horas y dos asignaciones como maximo. Las asignaciones ajenas consumen capacidad, pero nunca se detienen. Un proceso anfitrion WSL evita el cierre prematuro de sus hijos durante el arranque.

No cerrar WSL, suspender el equipo ni modificar codigo, datos o bundles mientras haya trabajo activo. Los pings no garantizan inmunidad a una revocacion del servicio, perdida de red o apagado local.

## Secuencia

1. Revalidar evidencia historica de desarrollo: tres hibridos, 4B solo y diagnostico 0.8B en A100. No repetir esas inferencias.
2. Repetir solamente la latencia minima que fallo: 2 preguntas de desarrollo y 4 sinteticas, 36 mediciones. Cerrar completamente B antes de cargar G4+J, y viceversa. Cargas, probes y recalentamiento fuera de `T_total`.
3. Piloto: seleccion entre soluciones completas y modelo grande con muestreo, 40 casos cada uno; despues 4B solo, 40 casos.
4. Aplicar `tools/pilot_decision.py` con la configuracion vigente. No-go detiene el experimento. No autoriza cambios de prompt, semillas, pesos o presupuestos.
5. Solo con go: confirmatoria en pares [grande con muestreo, grande greedy], [4B solo, seleccion por bloques], [seleccion por pasos, seleccion final]. Cien casos por brazo.
6. Latencia completa: una A100, 192 filas, con recargas fuera del cronometro. Analisis CPU con exactamente siete ZIP finales; auditoria independiente.

## Comprobar y recuperar

```powershell
# Solo lectura: asignaciones reales del servidor.
wsl -e colab sessions
```

Consultar primero `coordinator_reload_state.json`, `coordinator_reload_events.jsonl` y el `status.json` del job. El estado distingue ejecucion, verificacion y liberacion. Cada job conserva `execution_handover.json`, logs, checkpoints, ZIP finales y notebook ejecutado.

El comando siguiente solo inspecciona la preparacion; no asigna VMs:

```bash
python3 tools/colab/run_v2.py --plan --amend-latency-reload
```

No lanzar un segundo coordinador mientras el existente este vivo. Una recuperacion usa `--run --amend-latency-reload`, los mismos archivos de estado y sus pruebas de propiedad; no sustituye trabajos fallidos ni repite inferencia para mejorar resultados. Si aparece una entrega ambigua, revisar el proceso y la asignacion antes de decidir cualquier accion.

## Incidentes conservados

- `results/v2/preparation/`: candidato inicial rechazado por filtro monetario incompleto y candidato rechazado por cambio no autorizado de semilla. La seleccion final usa la semilla original.
- `results/v2/smoke/hybrid/`: tres notebooks completados, verificados y liberados. Estres observado de 36.12 GiB, entrada J de 16,330 tokens; acuerdo graph/eager 1.0 en tres prompts iniciales.
- `results/v2/smoke/single_lat/`: 4B solo completado y verificado por `gsingle_verification.json`; el operador combinado sigue marcado como fallido por el OOM posterior de latencia. No se ha fabricado una finalizacion exitosa de la secuencia.
- `results/v2/smoke/diagnostic_08b/`: diagnostico de desarrollo terminado; no se usa para conclusiones de calidad.
- El OOM de latencia no fue una desconexion: B dormido retenia 2.83 GiB y G no podia reservar sus 3 GiB de KV. La nueva configuracion mantiene las mismas caches y cambia exclusivamente la gestion de residencia entre fases.

Los tiempos de recarga se publican aparte. La latencia medida es condicional a motores cargados y calientes, no el costo extremo a extremo de intercambiar modelos. Las CU y el costo monetario no se inventan: las estimaciones operativas identifican explicitamente cuando solo se dispone de tiempo observado.
