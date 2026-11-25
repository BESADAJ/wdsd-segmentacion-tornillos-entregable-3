# Evaluación ampliada y robustez

## 1. Robustez a perturbaciones y a la escala (predicción 4)

- **Qué se hizo:** `experiments/robustez_e3.py` evalúa los 20 modelos finales de SGW y Mask R-CNN (10 semillas cada uno) sobre las 108 imágenes de test, con el umbral de val de cada corrida.
- **Perturbaciones:** las mismas tres de ruido y tres de desenfoque del E2, con la misma semilla de ruido por imagen, más la escala.
  - En escala, la red procesa la imagen al 75 % y al 50 %, cambiando también el tamaño interno de torchvision, y las máscaras se devuelven a la resolución original.
- **Control:** a escala 1,0 se reproduce exactamente la tabla confirmatoria (0,738 y 0,723).

**Tabla 8.1.** F1 por grupo en test bajo perturbación. Media de 10 semillas; d = SGW − Mask R-CNN.

| Perturbación | Nivel | Mask R-CNN | SGW | d | Semillas con SGW mejor |
|---|---|---|---|---|---|
| Escala | 1,00 | 0,723 | 0,738 | +0,016 | 8/10 |
| Escala | 0,75 | 0,592 | 0,623 | +0,031 | 7/10 |
| Escala | 0,50 | 0,124 | 0,144 | +0,019 | 8/10 |
| Ruido | σ = 0,02 | 0,700 | 0,712 | +0,012 | 9/10 |
| Ruido | σ = 0,05 | 0,621 | 0,624 | +0,003 | 5/10 |
| Ruido | σ = 0,10 | 0,482 | 0,449 | −0,034 | 3/10 |
| Desenfoque | σ = 1 px | 0,650 | 0,615 | −0,035 | 2/10 |
| Desenfoque | σ = 2 px | 0,221 | 0,164 | −0,057 | 1/10 |
| Desenfoque | σ = 4 px | 0,012 | 0,011 | −0,001 | 4/10 |

**Predicción 4: no se cumple.** La ventaja de SGW no crece al reducir la escala.

| Contraste | Interacción por grupo | IC 95 % | p de decisión (Holm) |
|---|---|---|---|
| d(0,75) − d(1,0) | +0,015 | [−0,012; 0,047] | 0,53 |
| d(0,50) − d(1,0) | +0,004 | [−0,020; 0,025] | 0,53 |

Al 50 %, los dos modelos colapsan (F1 de 0,12 a 0,14): con el lado mediano en unos 5 px, los tornillos quedan por debajo del piso c = 4 px y del paso de P2. Por eso ninguna asignación puede rescatarlos.

**Hallazgo nuevo (exploratorio): SGW es menos robusto al desenfoque y al ruido fuerte.** Con desenfoque de σ = 1–2 px pierde entre 0,035 y 0,057 más que Mask R-CNN, en 8 a 9 de 10 semillas.

Una explicación coherente con el mecanismo: SGW entrena como positivas propuestas más alejadas del tornillo. El clasificador aprende así a aceptar evidencia más difusa y, con el desenfoque, produce más falsos positivos y máscaras peor delimitadas. Es una hipótesis para trabajo futuro, no algo verificado.

Comparado con los 13 modelos del E2 (`data/robustez.csv`, página del modelo original en el dashboard):
- SGW y Mask R-CNN están entre los más robustos al ruido leve;
- toda la familia de dos etapas cae con el desenfoque.

## 2. Tres conjuntos adicionales

Se registraron antes de descargarlos (tag `e3-conjuntos`). Las desviaciones que aparecieron al prepararlos se declararon antes de entrenar (protocolo §6):

| Conjunto | Naturaleza | Imágenes (train/val/test) | Objetos | Área mediana | Unidad estadística en test |
|---|---|---|---|---|---|
| BBBC039 | Núcleos en fluorescencia | 100 / 50 / 50 | 23.606 | 627 px | Imagen (n = 50) |
| NuInsSeg | Núcleos en H&E, 31 órganos | 513 / 54 / 98 | 34.835 | 465–819 px | Imagen (n = 98); partición por órgano |
| MinneApple | Manzanas en huerto | 547 / 89 / 34 | 23.064 | 486–897 px | Imagen (n = 34); partición por secuencia |

- **Mismo protocolo que en WDSD:** SGW (κ = 0,45, c = 4) y Mask R-CNN, 3 semillas, sin ajuste por conjunto.
- **Tope de detecciones:** las 100 detecciones por imagen recortan a los dos modelos en BBBC039 y NuInsSeg, que tienen hasta 364 objetos por imagen.
- **Tamaño:** en los tres conjuntos los objetos son mucho más grandes que los tornillos. La hipótesis del mecanismo predice poca o ninguna ventaja: son una prueba de que SGW no empeora fuera de su nicho, más que de que mejora.

```{admonition} Resultados pendientes
:class: note
Las 18 corridas están en curso (`training/cola_extra.py`). Al terminar, se completan la tabla de consistencia (SGW mejor en al menos 2 de 3 y no peor en ninguno) y las pruebas por conjunto con Holm.
```

## 3. Calibración, incertidumbre, eficiencia y errores

- **Calibración:** el score de detección de la familia Mask R-CNN tiene un ECE de 0,04–0,06 en el E2 (temperatura y Platt en el capítulo 5 del E2). SGW comparte arquitectura; su ECE no se recalculó.
- **Incertidumbre:** intervalo conforme por partición del número de tornillos por imagen, calibrado en val (93 imágenes). Al 90 % el radio es de ± 2 tornillos, con una cobertura empírica en test del 82 %, por debajo de la nominal. Val y test son escenas distintas y la intercambiabilidad no se cumple: el intervalo se muestra en el dashboard con esta advertencia.
- **Eficiencia** (`results/exportacion/costo_cpu.csv`):
  - SGW tiene el mismo costo de inferencia que Mask R-CNN: 43,9 M de parámetros y 383 GFLOPs por imagen.
  - En CPU, con 6 hilos, la mediana es de unos 1,8–1,9 s por imagen con ONNX Runtime, frente a 2,9 s con PyTorch.
  - El entrenamiento es un 4 % más lento.
- **Errores:** los diagnósticos por imagen (TP, FP y FN, y Dice frente a F1) de los 15 modelos están en la página de Modelos del dashboard.
  - La ganancia de SGW en tornillos pequeños se concentra en el grupo 6 (capítulo 7 §3).
