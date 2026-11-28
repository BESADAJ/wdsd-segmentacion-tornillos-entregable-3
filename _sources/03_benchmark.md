# Modelos benchmark, selección y exportación (Bloque B)

## 1. Benchmark congelado

Los 14 modelos del E2 (Top 10, Mask R-CNN, YOLOv8, SOLOv2 y el clásico) quedan congelados con su configuración elegida en validación. Su definición, hiperparámetros y presupuesto de ajuste (3 tasas × 2 épocas) están en el informe del E2, capítulos 3 y 4.

`data/tabla_maestra.csv` es la tabla maestra que usan el libro y el dashboard. Para cada modelo da:
- la métrica media ± DE y su IC 95 % (t sobre semillas);
- parámetros, GFLOPs y latencia en GPU (E2);
- calibración (ECE);
- rango de Friedman y decisión frente a SGW.

## 2. Criterio de selección

- **Por modelo:** la configuración de hiperparámetros con mayor F1 por grupo medio en validación (regla del E2, empate a 0,005 hacia el valor por defecto).
- **Mejor modelo global:** el de mayor F1 en validación entre los finalistas de Friedman.
  - En el E2, Mask R-CNN; en el E3, SGW, con F1 medio en val de 0,808 frente a 0,794 del control (10 semillas cada uno).
  - Ambos pertenecen al grupo de finalistas que Nemenyi no distingue (capítulo 7).
- **Semilla exportada:** para SGW y Mask R-CNN, la de mayor F1 por grupo en val entre las 10. Para los dos fue la semilla 48.
  - Su F1 en test no se usó para elegir.
  - Hay sesgo del ganador: elegir el máximo de 10 en val sobreestima su desempeño en val, no en test.

## 3. Política de reentrenamiento

El enunciado pide reentrenar en el entrenamiento completo con el número de épocas fijado en validación. Todos los modelos, incluido el original, usan la misma política:
- 6 épocas sobre las 744 imágenes de train;
- se queda el *checkpoint* de la mejor época según el AP de val.

**Desviación declarada:** no se reentrenó sobre train + val, para no perder la selección de época y umbral en val y para mantener la misma política en los 14 rivales congelados.

## 4. Exportación, ida y vuelta y costo

`training/exportar_onnx.py` exporta el grafo completo de torchvision (preprocesado, RPN, RoI y máscaras) a ONNX (opset 17), con entrada fija de 3×720×1280.
- **Compresión:** los pesos se guardan en FP16 y se pasan a FP32 al cargar, lo que deja el archivo en 84 MB frente a 168. Las dos alternativas habituales fallaron:
  - la conversión completa a FP16 rompe la inferencia de tipos de los nodos `If`;
  - la cuantización INT8 dinámica no tiene kernel `ConvInteger` en CPU.
- **Ida y vuelta:** 10 imágenes fijas de test, más una de la galería que se prueba en CI. Tolerancias: |Δscore| < 10⁻³, |Δcaja| < 0,5 px, IoU de máscara ≥ 0,95 y el mismo número de detecciones sobre el umbral.
- **Métrica del exportado:** F1 por grupo en las 108 imágenes con ONNX Runtime FP16, frente al modelo en memoria, con prueba TOST de equivalencia (δ = 0,20).
- **Costo en CPU:** latencia mediana y p95 (lote 1), memoria pico, parámetros y GFLOPs.

Resultados en `results/exportacion/costo_cpu.csv` e `ida_y_vuelta.csv`; metadatos en `artifacts/*.metadata.json`.

**Tabla 3.1.** Exportación. Semilla 48 de cada modelo, CPU de 6 hilos.

| Modelo | F1 test (PyTorch, CPU, FP32) | F1 test (ONNX) | Máx. \|Δscore\| | Máx. \|Δcaja\| | IoU mín. de máscara | Latencia PyTorch | Latencia ONNX (mediana / p95) |
|---|---|---|---|---|---|---|---|
| SGW | 0,734 | 0,734 | 0,0002 | 0,007 px | 1,00 | 2,29 s | 1,76 / 1,92 s |
| Mask R-CNN | 0,707 | 0,707 | 0,0004 | 0,004 px | 1,00 | 2,28 s | 1,65 / 1,80 s |

- **Equivalencia:** el exportado reproduce exactamente la métrica primaria del modelo en memoria. La diferencia es 0, y la prueba TOST con δ = 0,20 da p < 0,001.
- **Diferencia con la tabla confirmatoria:** el F1 de la semilla 48 en CPU con FP32 (0,734) difiere del de su corrida en GPU con precisión mixta (0,741). La diferencia viene de la precisión numérica, no de la exportación.
- **Umbral en el grafo:** el grafo exportado aplica el umbral de score de val. Las detecciones sobre el umbral son las mismas y el grafo no materializa las máscaras de las que se descartan.

Los otros 12 modelos del E2 se muestran en el dashboard con sus resultados precalculados:
- 7 corren en MMDetection y 2 en los repositorios oficiales de YOLOv7 y YOLOv9, y su exportación a ONNX no es directa;
- todos juntos no caben en la memoria del plan gratuito de Render.
