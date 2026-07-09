# Integrar el log BPI Challenge 2013 (incidents, colaborativo) en la experimentación

## Contexto

`data/logs/BPIChallenge2013/` ya contiene un log colaborativo completo derivado del
sub-log *incidents* de BPI Challenge 2013 (Volvo IT VINST): el XES original
(`original/`), un conversor a vocabulario colaborativo (`collab_convert.py`), el XES
colaborativo resultante, su conversión a OCEL 2.0 (`.sqlite`/`.jsonocel`, generada con
el mismo conversor genérico `src/mapping/collab_xes_to_ocel.py` que usan los cuatro
logs de Predict-Collab), visualizaciones OC-DFG/OCPN, y documentación de diseño ya
redactada (`informacion.md`, `description.tex` — una subsección lista para paper —,
`metrics.json`). Todo esto está sin trackear en git y **no está cableado** en
`src/ocpm_eval/config.py` ni mencionado en ningún README.

**Evaluación de pertinencia.** Es una adición valiosa: aporta validez externa (log
real, no simulado sobre modelos BPMN) y escala (7.554 casos / 65.533 eventos vs.
~100 casos / hasta ~1.800 eventos en los logs actuales), permitiendo métricas RQ3 más
robustas. El esquema OCEL2 generado es estructuralmente compatible (mismas tablas
núcleo `event`/`object`/`event_object`/`object_object`/`object_Collaborationcase`
/`object_Participant`/`object_Participantprojection`/`object_Message`), verificado por
inspección directa del `.sqlite`.

Sin embargo, se detectó una incompatibilidad concreta: el `collab_convert.py` actual
implementa un "modelo de mensaje independiente" que solo mintea eventos
`ReceiveTask` (nunca `SendTask`), porque el log fuente no registra el lado emisor de
un traspaso de línea. Como `ocpm_tasks/labels.py` define 5 de las 14 tareas
(`NE-NPaM`, `NV-TNM`, `NV-NMPr`, `NV-NMPa`, `OB-M`) en términos de `is_send`
(no `is_send or is_receive`), y `OB-M` (parte del subconjunto representativo
`RQ3_SUBSET`) lanza `ValueError` cuando su parámetro auto-resuelto (actividad de
send más frecuente) es `None` —justo lo que ocurre sin eventos `is_send`—, el log
tal cual está hoy **rompería** `run_rq2`/`run_rq3` para este log (el `try/except`
por log en `rq2_fidelity.run_rq2`/`rq3_pipeline.run_rq3` absorbe la excepción, pero
pierde silenciosamente TODOS los resultados de BPI2013, no solo `OB-M`).

Se descartó "arreglarlo" fabricando un send genérico o solo cambiando el lado de
evaluación. En su lugar, **se revisa el conversor** para sintetizar, de forma
explícitamente marcada, un evento `SendTask` correlacionado por cada `ReceiveTask`
detectado, atribuido a la línea anterior y usando los atributos organizacionales
(`org:resource`/`org:group`/`org:role`) que el propio evento `Queued` entrante ya
registra — justificado por el hallazgo empírico de `informacion.md` (98,2% de esos
eventos `Queued` con cambio de línea son ejecutados por el recurso del handler
anterior). Esto no es fabricar datos no observados: es una re-atribución
determinística de un dato ya presente en el evento, declarada explícitamente (no
oculta) vía un atributo residual `collab:synthesized="true"`, siguiendo la política
que el propio script ya declara ("FLAG but do NOT silently correct"). Los resultados
de este log se mantienen en una etapa y CSVs separados de los cuatro logs de estudio
(decisión del usuario), ya que no comparte procedencia con el corpus reusado de
Delgado et al. (2025).

## 1. Revisar `collab_convert.py` para sintetizar el lado *send*

Archivo: `data/logs/BPIChallenge2013/collab_convert.py`

- En `transform()` (rama `elif is_line_change:` dentro de `if status ==
  TRANSFER_STATUS:`, líneas ~178-189): además de re-etiquetar el evento actual como
  `ReceiveTask`, sintetizar **un evento adicional**:
  - `elem_type="SendTask"`, `collab:participant=prev_part`,
    `collab:fromParticipant=prev_part`, `collab:toParticipant=part`, mismo
    `msg_id=f"msg{mid:07d}"` que el `ReceiveTask` pareado (para que ambos queden
    correlacionados como una sola `Message` con dos observaciones).
  - `src`: copia superficial del evento fuente `e`, con `organization involved`
    sobrescrito a `prev_part` (los demás campos preservados —`org:resource`,
    `org:role`, `org:group`, países— ya describen factualmente al emisor real).
  - Atributo residual `collab:synthesized="true"` **solo** en este evento sintetizado
    (no en el `ReceiveTask`, que sigue siendo una observación real).
  - Mismo `time:timestamp` que el evento `Queued` disparador; insertarlo en
    `enriched`/`local_events[prev_part]` inmediatamente antes del `ReceiveTask`, para
    que el orden de inserción (la convención ya vigente en el proyecto desde el
    commit "rely on insertion order instead of sequenceNumber") lo ordene primero en
    cualquier sort estable por timestamp (relevante para `rq2_fidelity._read_xes_cases`,
    que reordena por `time:timestamp` con `mergesort` estable).
  - Incrementar `eid`/`stats["n_events"]`; añadir contador `n_send_synthesized` y
    extender `elem_type_events` en `summarize()` con `"SendTask"`.
- En `write_collab_xes()` (líneas ~261-310): serializar el `msg_id` (hoy calculado en
  `message_id` pero **nunca escrito** al XES) y el nuevo `collab:synthesized`, para
  ambos lados del par. Esto habilita correlación explícita (mismo mecanismo
  `corr_attr` que ya usa ToyCollab con `msgId`) y permite que el mapeo genérico
  pueble `sender`/`receiver` en `object_Message`.
- Actualizar el docstring del módulo y de `write_collab_xes()` (hoy dicen "no
  fabricated send... SendTask is therefore never emitted") para describir el
  diseño revisado y su justificación.
- Resolver la nota "must be reconciled with the authoritative collaborative-XES
  extension" — verificado: `elemType`/`participant`/`fromParticipant`/`toParticipant`
  ya coinciden exactamente con `src/mapping/support/collab.xesext`.
- `chmod 644 collab_convert.py metrics.json` (hoy están en 600, a diferencia de sus
  archivos hermanos).

## 2. Regenerar los artefactos derivados

- Re-ejecutar `collab_convert.py` apuntando a los mismos paths ya usados
  (`original/BPI_Challenge_2013_incidents.xes.gz` → `BPI2013_incidents_collaborative.xes`
  + `metrics.json`).
- Re-ejecutar `python src/mapping/collab_xes_to_ocel.py
  data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.xes
  data/logs/BPIChallenge2013/BPI2013_incidents_collaborative` (mismo conversor R2
  genérico que los demás logs) para regenerar `.sqlite`/`.jsonocel` y, si se desea,
  las visualizaciones OC-DFG/OCPN.
- Verificar en el `.sqlite` regenerado: aparece una tabla de evento tipo `SendTask`
  equivalente, `object_Message` tiene `sender` y `receiver` poblados, y
  `elem_type_events`/`n_events` en el `metrics.json` regenerado reflejan los ~4.051
  eventos sintetizados añadidos.

## 3. Cablear el log en el pipeline de evaluación (etapa y config separados)

Archivo: `src/ocpm_eval/config.py`
- Agregar `real_world_ocel_logs() -> List[LogSpec]` devolviendo un único
  `LogSpec("BPI2013", "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.sqlite",
  "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.xes")`, espejando el
  patrón ya usado por `predictcollab_ocel_logs()`/`toy_ext_log()` — deliberadamente
  fuera de `predictcollab_ocel_logs()`.

Archivo: `src/ocpm_eval/rq2_fidelity.py`
- Añadir parámetro `out_name: str = "rq2_fidelity.csv"` a `run_rq2` (hoy
  hardcodeado), igual que ya existe en `run_rq3`.

Archivo: `src/ocpm_eval/run_evaluation.py`
- Añadir una nueva etapa tras RQ3-full, usando `replace(cfg, logs=real_world_ocel_logs())`
  (mismo patrón que `full_cfg` ya usa para el catálogo completo):
  ```python
  print("\n########## RQ2/RQ3 — real-world validation (BPI2013) ##########")
  bpi_cfg = replace(cfg, logs=real_world_ocel_logs())
  results["rq2_bpi2013"] = run_rq2(bpi_cfg, out_name="rq2_fidelity_bpi2013.csv")
  results["rq3_bpi2013"] = run_rq3(bpi_cfg, out_name="rq3_results_bpi2013.csv")
  ```
- Dado que BPI2013 es ~36x más grande que el log más grande actual, cronometrar la
  primera corrida; si la extracción de features de OCPA/ajuste de RandomForest es
  lenta, dejar esta etapa como opt-in (parámetro de función) en vez de correr
  siempre dentro de `python -m ocpm_eval.run_evaluation`.
- El `try/except` por log ya existente queda como red de seguridad: los casos borde
  que persistan sin síntesis posible (p. ej. `Queued` en primera posición de CI, sin
  predecesor observado — igual que hoy, por diseño) seguirán marcados como
  "unmatched", no corregidos.

## 4. Documentación — texto

- `data/logs/README.md`: actualizar el párrafo de "Message detection" en la sección
  `BPIChallenge2013/` para describir el diseño revisado (send sintetizado y
  justificado, ya no "no fabricated send").
- `README.md` (raíz):
  - Añadir `BPIChallenge2013/` al árbol de estructura del repo (~línea 76-81).
  - Añadir una fila a la tabla "Evaluation stages" (~línea 213-218) para la nueva
    etapa de validación real-world y sus CSVs de salida.
  - Corregir de paso el texto obsoleto "feasibility demo on a Healthcare variant" en
    la fila de RQ-EXT (inconsistencia preexistente, debería decir "toy log").
- `data/logs/BPIChallenge2013/description.tex` / `informacion.md`: reescribir el
  párrafo "Message detection under the independent message model" para reflejar el
  diseño sintetizado, y agregar una oración explícita de amenaza a la validez: el
  lado *send* de cada traspaso es un evento **derivado/sintetizado**, no registrado
  directamente, justificado por la concordancia empírica del 98,2% — al mismo nivel
  que la ya declarada inferencia de `fromParticipant`. Actualizar la tabla de
  métricas embebida con los números regenerados.

## 5. Verificación

- Correr la nueva etapa BPI2013 end-to-end: confirmar que no crashea, que
  `OB-M`/`NV-NMPr`/`NV-NMPa`/`NE-NPaM`/`NV-TNM` producen métricas no degeneradas, y
  que `rq2_fidelity_bpi2013.csv`/`rq3_results_bpi2013.csv` tienen filas completas
  para las 14/6 tareas respectivamente.
- Confirmar que `rq2_fidelity.csv`, `rq3_results.csv` y `rq3_results_full.csv` (los
  cuatro logs de estudio) no cambian de contenido — sin regresión en los resultados
  ya usados para el paper.
- Revisar permisos (`collab_convert.py`, `metrics.json`) y limpiar `.DS_Store` antes
  de considerar la carpeta lista para commitear.
