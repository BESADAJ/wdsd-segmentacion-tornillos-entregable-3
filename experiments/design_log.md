# Bitácora de diseño del modelo original (E3 §7.2)

Se registran todas las variantes evaluadas, incluidas las descartadas. Las métricas de validación son de la **partición de desarrollo** (train/val del E2). La partición de confirmación y test no se usan para diseñar.

| N.° | Variante (cambio respecto a la anterior) | Métrica | Decisión | Motivo y fecha |
|---|---|---|---|---|
| 0 | Mask R-CNN del E1/E2 con asignación por IoU (referencia) | val F1 por grupo del E2: ver `corridas.csv` del E2 | Referencia | Línea de partida. 2026-11-23 |
| 1 | SGW con κ = 1, c = 4 en RPN y RoI | Sin entrenar: prueba de sobreajuste de un lote sintético (256 px, red sin preentrenar). La pérdida inicial fue 14,7, contra 3,8 con IoU | Descartada | Con κ = 1 demasiadas anclas cuentan como positivas: los umbrales del IoU pierden su significado. Lleva a calibrar κ analíticamente. No se usó ningún dato de WDSD. 2026-11-23 |
| 2 | SGW con κ₀ = 0,45 (calibrado con el IoU en el límite de objetos grandes), c = 4, en RPN y RoI; búsqueda κ ∈ {0,225; 0,45; 0,9} | Pendiente: piloto de asignación y búsqueda en val | En curso | Calibración derivada a priori, sin datos (`book/05_modelo_diseno.md` §2.2). 2026-11-23 |

| — | Piloto de asignación (no es una variante; sin entrenar, solo train) | RoI: fracción de propuestas positivas en el tercil pequeño 0,44 (IoU) frente a 0,85 (κ₀); RPN casi igual | Se mantiene la rejilla declarada | La predicción 1 se cumple en RoI y no en RPN. 2026-11-23 |

N_d hasta la fecha: 2 variantes (1 descartada sin datos reales).
