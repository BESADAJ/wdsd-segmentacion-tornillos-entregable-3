# Ficha del modelo: SGW (Mask R-CNN con asignación de Wasserstein normalizada por tamaño)

**Versión 1.0.0 · 2026-11-25 · Ivan Berrio, Jose David Castillo y Jaime Besada (curso de Deep Learning, Entregable 3)**

## Descripción

Mask R-CNN ResNet-50-FPN (torchvision) preentrenado en COCO.
- **Anclas:** de 8 a 128 px.
- **Asignación de positivos:** en RPN y en la cabeza RoI no se usa el IoU, sino la similitud SGW: s = exp(−W₂/D), donde W₂ es la distancia de Wasserstein-2 entre las gaussianas de las cajas y D = κ(√(w·h) + c) es la escala propia de cada tornillo.
- **Hiperparámetros propios:** κ = 0,45 y c = 4 px.
- **En inferencia** la red es idéntica a Mask R-CNN: SGW solo cambia qué propuestas se entrenan como positivas.

- **Código del mecanismo:** `src/original_model.py`. Pruebas: `tests/test_original_model.py`. Teoría: capítulo 5 del libro.
- **Artefacto:** `artifacts/sgw_k0.45.onnx`, ONNX opset 17 con pesos almacenados en FP16 y cómputo en FP32. Metadatos en `artifacts/sgw_k0.45.metadata.json`.
- **Selección:** de las 10 semillas finales (42–51), la de mayor F1 por grupo en validación.

## Uso previsto

- Localizar y segmentar tornillos en fotografías cenitales de aparatos electrónicos en desensamblado, de 1280×720 px, como apoyo a un sistema robótico o a un operador.
- Investigación y docencia sobre detección de objetos diminutos.

## Usos no recomendados

- **Como único sensor en un robot sin verificación:** pierde cerca de 6 de cada 10 tornillos del tercil más pequeño.
- **En imágenes muy distintas de WDSD** (otra resolución, otra iluminación u otros objetos): la app avisa cuando la entrada está fuera de dominio.
- Contar tornillos con garantías estrictas.

## Datos de entrenamiento

WEEE Disassembly Screw Dataset (VCL, CERTH-ITI), sin licencia declarada.
- **Partición agrupada por escena:** 744 imágenes de entrenamiento, 93 de validación y 108 de prueba (14 grupos).
- **Tamaño de los tornillos:** el 98,8 % es pequeño según COCO; el lado mediano es de 10,6 px.

## Métricas

F1 de instancia por imagen (IoU de máscara ≥ 0,5), promediado por grupo de escena, en test. Media ± DE sobre 10 semillas:

| Modelo | F1 por grupo | Recall del tercil pequeño* | Recall del tercil grande* |
|---|---|---|---|
| SGW | 0,738 ± 0,014 | 0,385 | 0,757 |
| Mask R-CNN (misma GPU) | 0,723 ± 0,018 | 0,302 | 0,776 |

\* Todos los tornillos juntos, por semilla (exploratorio). Por grupo, que es la unidad preregistrada, la diferencia en el tercil pequeño es +0,012 y no es significativa.

- **R3:** no se cumple. SGW no supera de forma significativa a Mask R-CNN, HTC, Cascade ni MS R-CNN, y supera por más de δ = 0,20 solo a 4 de 14 rivales.
- **Calibración:** el score de detección no es una probabilidad calibrada.
- **Intervalo del conteo:** el intervalo conforme al 90 % del número de tornillos (± 2) cubre el 82 % de las imágenes de test, porque las escenas no son intercambiables.

## Costo

- **Parámetros:** 43,9 millones.
- **Cómputo:** unos 392 GFLOPs por imagen.
- **Latencia en GPU:** 88 ms (RTX 3060).
- **Latencia en CPU:** ver `results/exportacion/costo_cpu.csv`.

## Limitaciones y riesgos

- La ganancia en tornillos diminutos se concentra en una escena (grupo 6), así que no está demostrado que se generalice.
- SGW se ajusta más al entrenamiento que Mask R-CNN: la brecha entre train93 y val es 0,098 frente a 0,059. La brecha entre val y test es igual en ambos.
- Imágenes subexpuestas, desenfocadas o con ruido degradan mucho el desempeño (capítulo 8).
- **Riesgo operativo:** un falso negativo deja un tornillo sin retirar y puede dañar piezas al forzar el desensamblado. Se recomienda verificación humana.

## Ética y privacidad

- Las imágenes no contienen personas.
- La app procesa las imágenes subidas en memoria y no las guarda.
- El dataset no declara licencia: no se redistribuyen sus imágenes completas.

## Cómo citar

Ver `CITATION.cff`.
