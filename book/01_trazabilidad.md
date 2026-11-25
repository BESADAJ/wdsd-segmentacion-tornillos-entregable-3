# Trazabilidad de correcciones del Entregable 2 (Bloque A)

```{admonition} Pendiente
:class: warning
Falta incorporar las observaciones del docente al E2. La tabla recoge por ahora los faltantes que el propio equipo declaró en el informe del E2 (`informe/pendientes.md` y el checklist del Apéndice B) y lo que se hizo con cada uno.
```

| N.° | Observación (fuente) | Acción realizada | Impacto en resultados | Estado |
|---|---|---|---|---|
| 1 | Auditoría de fuga de la partición agrupada (E2 §10.1) | Ejecutada (`auditoria_fuga_e2.py`): sin grupos compartidos ni duplicados exactos; AUC adversarial 0,33. Falla el criterio de casi gemelos: 12 % de test, con límite de 5 % | Se reporta el F1 con y sin esas imágenes; no cambia el ranking | Hecho (E2) |
| 2 | Determinismo documentado | Ultralytics corre en modo determinista y MMDetection no; se documenta en el E2. En el E3, Mask R-CNN se reentrenó con 10 semillas en la 3050: la diferencia con la 3060 es +0,004 (p = 0,42) | Ninguno en el ranking | Hecho |
| 3 | Réplica del paper guía y su limitación | Réplica parcial de Kalitsios et al. (2022) y verificación de la fuga de la partición oficial en 2 de 3 modelos | Justifica la partición agrupada | Hecho (E2) |
| 4 | Hoja de extracción y calidad de los 64 estudios (§3.1) | — | — | Pendiente (equipo) |
| 5 | Ficha por modelo: DOI, diagrama, innovación y matriz comparativa (§3.3–3.4) | — | — | Pendiente (equipo) |
| 6 | Techo de desempeño con reanotación (§12) | — | — | Pendiente: hay que reanotar una muestra |
| 7 | Formato A.3 de las tablas de pruebas | Las tablas del capítulo 7 del E3 incluyen p exactos y ajustados, IC y tamaños de efecto | — | En curso |
| 8 | Dockerfile y ejecución con un solo comando | Dockerfile del dashboard (E3); el entrenamiento sigue por pasos | — | Parcial |

## Ranking antes y después

El benchmark del E2 queda **congelado**: no se reentrenó ningún rival. El único cambio es que el E3 añade el modelo original y un control de Mask R-CNN con 10 semillas en otra GPU.

| Posición | E2 (F1 por grupo) | E3 (F1 por grupo) |
|---|---|---|
| 1 | Mask R-CNN 0,726 | **SGW 0,738** |
| 2 | HTC 0,724 | HTC 0,724 |
| 3 | Cascade 0,723 | Cascade 0,723 · Mask R-CNN (3050, 10 semillas) 0,723 |
| 4 | MS R-CNN 0,716 | MS R-CNN 0,716 |

El resto del orden no cambia (tabla 7.1).
