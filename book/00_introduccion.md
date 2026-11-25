# Introducción

## Resumen ejecutivo

**Problema.** Desensamblar residuos electrónicos con robots exige localizar cada tornillo.
- En 2022 se generaron 62 millones de toneladas de estos residuos y solo el 22,3 % se recicló de forma documentada (Baldé et al., 2024).
- En el WEEE Disassembly Screw Dataset (WDSD), un tornillo mide en mediana 10,6 px de lado.
- En el Entregable 2, ninguno de los 14 modelos comparados superó un recall de 0,37 en el tercil de tornillos más pequeños.

**Modelo original.** Proponemos SGW, una asignación de propuestas para Mask R-CNN que reemplaza el IoU por una similitud de Wasserstein gaussiana normalizada por la escala de cada tornillo, con un piso de 4 px.
- Demostramos que su sensibilidad a errores de posición está acotada para cualquier tamaño, mientras que la del IoU crece sin límite al encoger el objeto.

**Resultados**, con 10 semillas y un protocolo preregistrado:
- SGW alcanza 0,738 de F1 de instancia por grupo, el mejor del benchmark, frente a 0,723 de Mask R-CNN.
- La diferencia no es significativa (Holm), y R3 no se cumple.
- El recall de tornillos pequeños sube de 0,30 a 0,39 contando todos los tornillos, pero la prueba preregistrada por escena no lo confirma: la ganancia se concentra en una escena.

## Contribuciones

1. SGW: una similitud de asignación normalizada por objeto y con piso, aplicada en RPN y RoI de Mask R-CNN, con calibración analítica de κ (capítulo 5).
2. El teorema 1 y la proposición 2, que comparan la estabilidad de SGW y del IoU ante errores de posición, con verificación numérica (capítulo 5).
3. Una evaluación preregistrada con 10 semillas pareadas, familia completa de 14 rivales, ablación y análisis de robustez (capítulos 7 y 8).
4. Un dashboard desplegado con inferencia ONNX en CPU, aviso fuera de dominio e intervalo conforme del conteo (capítulo 9).

## Objetivos

- Corregir y congelar el benchmark del E2 (Bloque A).
- Exportar los mejores modelos (Bloque B).
- Diseñar, demostrar y validar un modelo original (Bloques C y D).
- Desplegarlo en un dashboard (Bloques E y F).

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 1 | Trazabilidad de correcciones del E2 |
| 2 | Datos: EDA y auditoría de fuga |
| 3 | Benchmark, selección y exportación |
| 4 | Revisión de literatura y novedad |
| 5 | Modelo original: diseño y teoría |
| 6 | Implementación |
| 7 | Protocolo y validación estadística |
| 8 | Evaluación ampliada y robustez |
| 9 | Dashboard y despliegue |
| 10 | Conclusiones |
| 11 | Reproducibilidad |

**Dashboard:** URL pública en Render, pendiente de desplegar (capítulo 9).
