# Protocolo y validación estadística

Este capítulo aplica el protocolo preregistrado (`experiments/protocolo.md`, tag `e3-protocolo` del 2026-11-23) a las 50 corridas finales de SGW: semillas 42–51 × 5 variantes, todas en la misma RTX 3050, sin fallos técnicos ni exclusiones. Todas las cifras salen de `experiments/analisis_confirmatorio.py`, que escribe `results/confirmatorio/`. El F1 recalculado desde las predicciones coincide con el de cada `resumen.json` (diferencia máxima 2 × 10⁻¹⁶).

## 1. Estándar, equidad y familia de contrastes

- **Unidad de independencia:** el grupo de escena (14 en test). El F1 de cada grupo es la media de sus imágenes, promediada sobre las semillas del modelo.
- **Equidad del presupuesto:**
  - SGW tiene el mismo presupuesto que Mask R-CNN en el E2 (6 épocas, lote 2, tasa 0,01, coseno, AMP, volteo, pesos de COCO).
  - Su único hiperparámetro propio, κ, se buscó con 3 valores y 2 épocas, igual que la tasa de los rivales del E2.
  - El control Mask R-CNN se volvió a entrenar con las mismas 10 semillas y en la misma GPU.
  - Frente al Mask R-CNN del E2 (3060) no hay diferencia de hardware: +0,004, p = 0,42 en las semillas 42–46.
- **Esfuerzo de diseño declarado:** N_d = 2 variantes (bitácora de diseño), una de ellas descartada sin usar datos reales. Los rivales del E2 no tuvieron diseño propio: es la asimetría que pide declarar el enunciado.
- **Familia R3:** SGW frente a 14 rivales, que son el Top 10 del E2, Mask R-CNN, YOLOv8, SOLOv2 y el clásico.
- **Tres pruebas por contraste**, unilaterales (SGW > rival):
  - Wilcoxon pareado por grupo;
  - *bootstrap* pareado por grupo (10 000 remuestreos, IC 95 % percentil);
  - t pareado sobre las 10 semillas, solo frente a Mask R-CNN, que es el único rival con semillas pareadas.
- **Corrección y decisión:** Holm sobre los 14 contrastes en cada prueba. Decide el máximo de los p ajustados.
- **Relevancia:** diferencia ≥ δ = 0,20, con el IC inferior > 0.

## 2. Resultado principal (R3)

**Tabla 7.1.** F1 de instancia por grupo en test. Media ± desviación entre semillas.

| Modelo | F1 por grupo | DE | Semillas |
|---|---|---|---|
| **SGW (κ = 0,45, c = 4)** | **0,738** | 0,014 | 10 |
| Mask R-CNN (control, 3050) | 0,723 | 0,018 | 10 |
| Mask R-CNN (E2, 3060) | 0,726 | 0,012 | 5 |
| HTC | 0,724 | 0,017 | 5 |
| Cascade Mask R-CNN | 0,723 | 0,006 | 5 |
| MS R-CNN | 0,716 | 0,010 | 5 |
| PointRend | 0,701 | 0,019 | 5 |
| YOLOv7 | 0,638 | 0,030 | 5 |
| CondInst | 0,607 | 0,009 | 5 |
| YOLO26 | 0,569 | 0,020 | 5 |
| YOLOv8 | 0,566 | 0,032 | 5 |
| Mask2Former | 0,545 | 0,143 | 5 |
| YOLO11 | 0,530 | 0,047 | 5 |
| YOLOv9 | 0,405 | 0,038 | 5 |
| SOLOv2 | 0,231 | 0,021 | 5 |
| Clásico | 0,018 | 0,004 | 5 |

**Tabla 7.2.** Familia R3: diferencia SGW − rival en F1 por grupo, con 14 grupos.

| Rival | Diferencia | IC 95 % | Grupos a favor | p Holm Wilcoxon | p Holm bootstrap | p t (semillas) | δ de Cliff | d pareado | Decisión |
|---|---|---|---|---|---|---|---|---|---|
| Mask R-CNN | +0,015 | [−0,003; 0,035] | 9 | 0,291 | 0,190 | 0,059 | 0,09 | 0,42 | equivalente dentro de δ |
| Cascade | +0,015 | [−0,027; 0,053] | 9 | 0,291 | 0,382 | — | 0,08 | 0,19 | equivalente dentro de δ |
| HTC | +0,014 | [−0,021; 0,045] | 7 | 0,291 | 0,382 | — | 0,08 | 0,22 | equivalente dentro de δ |
| MS R-CNN | +0,022 | [−0,006; 0,048] | 10 | 0,266 | 0,191 | — | 0,12 | 0,41 | equivalente dentro de δ |
| PointRend | +0,037 | [0,013; 0,070] | 9 | 0,033 | 0,001 | — | 0,12 | 0,65 | superior, < δ |
| YOLOv7 | +0,100 | [0,049; 0,148] | 13 | 0,016 | 0,001 | — | 0,36 | 1,03 | superior, < δ |
| CondInst | +0,131 | [0,076; 0,192] | 13 | 0,007 | 0,001 | — | 0,31 | 1,13 | superior, < δ |
| YOLO26 | +0,169 | [0,110; 0,233] | 13 | 0,007 | 0,001 | — | 0,36 | 1,37 | superior, < δ |
| YOLOv8 | +0,172 | [0,124; 0,221] | 13 | 0,007 | 0,001 | — | 0,34 | 1,79 | superior, < δ |
| Mask2Former | +0,193 | [0,120; 0,264] | 13 | 0,002 | 0,001 | — | 0,42 | 1,37 | superior, < δ |
| YOLO11 | +0,208 | [0,151; 0,268] | 14 | 0,001 | 0,001 | — | 0,47 | 1,78 | **cumple R3** |
| YOLOv9 | +0,333 | [0,255; 0,414] | 14 | 0,001 | 0,001 | — | 0,63 | 2,11 | **cumple R3** |
| SOLOv2 | +0,507 | [0,321; 0,689] | 14 | 0,001 | 0,001 | — | 0,79 | 1,39 | **cumple R3** |
| Clásico | +0,720 | [0,565; 0,849] | 14 | 0,001 | 0,001 | — | 0,97 | 2,55 | **cumple R3** |

**Veredicto R3: no se cumple.** SGW es significativamente mejor que 10 de los 14 rivales. Supera por más de δ solo a 4: YOLO11, YOLOv9, SOLOv2 y el clásico. Frente a los cuatro modelos de dos etapas más fuertes (Mask R-CNN, Cascade, HTC y MS R-CNN), la diferencia es pequeña y no significativa: el IC queda dentro de ±δ. Es lo que el protocolo había anticipado: superar a Mask R-CNN por δ exigía un F1 de 0,93.

**Friedman, Iman-Davenport y Nemenyi** sobre los 15 modelos y los 14 grupos:
- χ² = 146,9 (p = 3 × 10⁻²⁴); F de Iman-Davenport = 38,8 (p = 4 × 10⁻⁴⁷).
- **SGW tiene el mejor rango medio (2,96).** La diferencia crítica de Nemenyi es CD = 5,73 (q = 3,39).
- Por Nemenyi, SGW no se distingue de Mask R-CNN, HTC, MS R-CNN, Cascade, PointRend, YOLOv7 y CondInst.
- Diagrama: `results/confirmatorio/diferencia_critica.png`.

## 3. Hipótesis del mecanismo (H-M) y predicciones secundarias

**H-M:** el recall del tercil pequeño (área < 75 px, cortes del E2) de SGW supera al de Mask R-CNN por más de 0,05.

| Cálculo | Diferencia | IC 95 % | Resultado |
|---|---|---|---|
| **Por grupo**, la unidad preregistrada (12 grupos con tornillos pequeños) | +0,012 | [−0,040; 0,062] | p Wilcoxon 0,93; p *bootstrap* 0,94: **no se confirma** |
| Todos los tornillos juntos, por semilla (exploratorio) | +0,083 | — | 9 de 10 semillas a favor; p de t contra 0,05 = 0,0004 |

**H-M no se confirma con el análisis preregistrado.** La discrepancia entre las dos filas tiene una causa concreta. La ganancia de recall pequeño viene casi entera del grupo 6:
- tiene 59 de los 155 tornillos pequeños de test;
- en él, SGW sube el recall de 0,22 a 0,41;
- aporta 11 de los 13 aciertos extra por semilla.

Los demás grupos tienen entre 1 y 8 tornillos pequeños, con un recall que casi no cambia (o que ya era alto). Promediar por grupo da a cada grupo el mismo peso, y así diluye la ganancia. El efecto existe y es estable entre semillas, pero depende de una escena. No se puede afirmar que se generalice a todas.

**Predicciones secundarias:** Holm entre las tres; por grupo, ninguna se confirma.

| Predicción | Por grupo | Todos los tornillos, por semilla (exploratorio) |
|---|---|---|
| P2: la ganancia se concentra en el tercil pequeño | +0,000 [−0,120; 0,094] | +0,102; 10/10 semillas |
| P3: quitar SGW de la cabeza RoI pesa más que quitarlo de RPN | −0,021 [−0,085; 0,027] | +0,055; 9/10 |
| P5: sin piso (c → 0) se pierde la ganancia | −0,008 [−0,073; 0,043] | +0,073; 9/10 |

P1 ya se había verificado en el piloto (capítulo 5). P4 (escala) se evalúa en el capítulo 8.

**Recall por tercil** (todos los tornillos juntos; media ± DE sobre 10 semillas):

| Modelo | Pequeño | Mediano | Grande |
|---|---|---|---|
| Mask R-CNN | 0,302 ± 0,035 | 0,662 ± 0,029 | 0,776 ± 0,026 |
| SGW | 0,385 ± 0,033 | 0,656 ± 0,020 | 0,757 ± 0,025 |

SGW cambia recall entre tamaños: gana en los tornillos pequeños y cede un poco en los grandes (−0,019).

## 4. Ablación

Familia propia: cada variante frente a SGW completo, bilateral, con Holm sobre 3 contrastes.

| Variante | Δ F1 por grupo | IC 95 % | p de decisión | Recall pequeño |
|---|---|---|---|---|
| sgw_rpn (SGW solo en RPN) | −0,013 | [−0,030; 0,004] | 0,30 | 0,328 |
| sgw_roi (SGW solo en RoI) | −0,009 | [−0,021; 0,002] | 0,22 | 0,383 |
| sgw_c0 (sin piso) | −0,015 | [−0,032; 0,001] | 0,30 | 0,312 |

Las tres variantes quedan por debajo de SGW completo, pero ninguna diferencia es significativa. Mirando todos los tornillos juntos, el patrón es coherente con el diseño:
- la cabeza RoI aporta casi toda la ganancia de recall pequeño (0,383 frente a 0,385 del modelo completo);
- quitar el piso la reduce (0,312), cerca del control (0,302).

## 5. Supuestos, potencia y sobreajuste

- **Supuestos:**
  - Las diferencias SGW − Mask R-CNN por grupo son compatibles con normalidad (Shapiro p = 0,59; asimetría 0,50).
  - Las diferencias por semilla no lo son (Shapiro p = 0,008), así que el t sobre semillas es solo orientativo.
  - La decisión usa el máximo de los tres p, que ya es conservador.
  - Las varianzas entre semillas son homogéneas (Levene p = 0,39).
- **Potencia (exploratorio):** el análisis de potencia no se registró antes de las corridas y se declara así.
  - Con la DE de las diferencias por grupo (0,037) y 14 grupos, la diferencia mínima detectable con potencia 0,90 es 0,031 a α = 0,05 y 0,044 a α/14 (peor caso de Holm).
  - Para δ = 0,20 la potencia es prácticamente 1.
  - Es decir: el estudio tiene potencia para detectar una diferencia de 0,04 frente a Mask R-CNN. La diferencia observada (0,015) es menor que eso.
- **Sobreajuste** (`sobreajuste_por_corrida.csv`):
  - La brecha entre train93 y val es mayor en SGW: 0,098 frente a 0,059 (t pareado p = 0,006). SGW ajusta más el entrenamiento, coherente con que asigna más propuestas positivas.
  - La brecha entre val y test es la misma: 0,070 frente a 0,071 (p = 0,92). El ajuste extra no se traduce en peor generalización relativa.
  - En 4 de las 10 corridas de SGW y en 3 de las 10 de Mask R-CNN, el AP de val sigue subiendo en la última época: los dos están algo subentrenados, igual que en el E2.

## 6. Qué se concluye

1. **SGW es el mejor modelo del benchmark en rango medio y en F1 medio,** pero su ventaja sobre los modelos de dos etapas (+0,015 frente a Mask R-CNN) no es significativa y está muy por debajo de δ. R3 no se cumple, como se había anticipado.
2. **El mecanismo hace lo que la teoría predice cuando hay muchos tornillos diminutos:** más recall en el tercil pequeño en 9 de 10 semillas. Pero esa ganancia se concentra en una escena, y la prueba preregistrada por grupo no la confirma.
3. **Los resultados negativos se reportan tal cual:** H-M por grupo y las predicciones 2, 3 y 5 por grupo. No se cambió la unidad de análisis después de ver los datos. La versión agregada por tornillo se presenta solo como exploratoria.
