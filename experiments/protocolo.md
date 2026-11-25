# Protocolo experimental preregistrado del modelo original (E3 §8.1)

**Estado: CONGELADO el 2026-11-23** con el tag de Git `e3-protocolo`, antes de cualquier corrida final (las finales son las únicas que tocan test).

Al congelar, la búsqueda de κ (solo val) estaba en curso: su resultado no se conocía. κ lo elige `training/cola_e3.py` aplicando la regla de §4, sin intervención manual.

Quedaban **abiertos**:
- la elección de los 3 conjuntos adicionales (§6): registrada el 2026-11-25 con el tag `e3-conjuntos`, antes de descargarlos;
- el análisis de potencia con las semillas piloto (§5): **no se registró antes de las corridas finales**. Se reporta como exploratorio (diferencia mínima detectable), y así se declara.

Ninguno de los dos cambia lo que aquí se fija.

## 1. Estándar y alcance (E3 §2), con las desviaciones declaradas de antemano

El equipo se compromete con el estándar del E3. Por cómputo (una RTX 3050 de 6 GB) y por plazo (2–3 días), se declaran **antes** de ver resultados confirmatorios estas desviaciones:

| Requisito del E3 | Qué se hace | Desviación |
|---|---|---|
| ≥ 10 semillas para toda la familia | SGW, Mask R-CNN y las 3 ablaciones: 10 semillas (42–51) en la misma GPU (3050). Resto del Top 10: las 5 semillas del E2 (42–46, en la 3060), más las que Ivan alcance a correr | El resto de la familia queda con 5 semillas y en otro hardware |
| Familia completa con Holm | Los 13 rivales del E2 (Top 10 + 3 modelos base) más el *baseline* clásico | Ninguna en la composición de la familia |
| 3 conjuntos adicionales | Por definir (§6) | Probablemente solo SGW frente a Mask R-CNN |
| Potencia ≥ 0,90 por contraste | Análisis de potencia con las semillas piloto (§5) | — |

## 2. Hipótesis

- **Confirmatoria principal (R3):** el F1 de instancia por grupo en test de SGW es mayor que el de cada rival, con una diferencia mayor que δ = 0,20 (heredado del E1). **Se espera que no se cumpla** frente a los modelos de dos etapas: superar a Mask R-CNN (0,726) por δ exige llegar a 0,93. Tope esperado: 4,0.
- **Confirmatoria del mecanismo (H-M, el hallazgo alternativo de §14.5.9):** el recall de SGW en el tercil pequeño (área < 75 px, cortes del E2) supera al de Mask R-CNN por al menos **0,05**. El margen se fija hoy, antes de entrenar.
- **Predicciones secundarias:** las 2 a 5 de `book/05_modelo_diseno.md` §4. La predicción 1 ya se verificó en el piloto.
- **Exploratorios:** todo lo demás. Se reporta con la etiqueta "exploratorio".

## 3. Métrica, datos y partición

- **Métrica primaria:** F1 de instancia por imagen (IoU de máscara ≥ 0,5), promediado por grupo de escena. Es la del E1/E2, sin cambios.
- **Secundarias:** recall y F1 por tercil de área, AP de máscara, PQ y la brecha entre train93 y val.
- **Partición:** la agrupada del E1 (744 / 93 / 108), congelada. Los JSON COCO están en `D:\wdsd_datos\datos_inst\coco`, generados con `experiments/preparar_datos.py`, el mismo código del E2. Su SHA-256 (primeros 16 caracteres) incluye las rutas absolutas de las imágenes:

| Archivo | SHA-256 | Contenido |
|---|---|---|
| train | `41b75d7da240c2d5` | 744 imágenes, 3.530 tornillos |
| val | `90d51a51d0e0eb4c` | 93 imágenes, 479 tornillos |
| test | `e03f737ebd0a9001` | 108 imágenes, 389 tornillos |
| train93 | `419ba6ae8a98e015` | submuestra de train para medir la brecha de sobreajuste |
- **Separación de desarrollo y confirmación (E3 §7.2):** las decisiones de diseño solo usaron train (piloto) y val (búsqueda de κ). Test se usa una sola vez por corrida final, con todo ya fijado.

## 4. Modelo y presupuesto

- **Presupuesto:** idéntico al de Mask R-CNN en el E2: 6 épocas, lote 2, SGD, tasa 0,01 (factor 2, heredado de la búsqueda del E2), coseno hasta el 1 %, AMP, solo volteo horizontal y pesos de COCO.
- **Hiperparámetro propio:** κ, elegido entre κ₀ × {0,5; 1; 2} = {0,225; 0,45; 0,9}. Se usa la semilla 42 y ⌈6/3⌉ = 2 épocas, con el mismo presupuesto (T = 3) que la búsqueda del E2.
  - Regla: máximo F1 por grupo en val; si κ₀ queda a 0,005 o menos del máximo, gana κ₀ (misma regla de empate que el E2).
  - El κ elegido se usa en todas las variantes.
- **Fijo a priori:** el piso c = 4 px.
- **Corridas finales:** semillas 42–51 (10) para `sgw`, `maskrcnn` (control en la misma GPU) y las tres ablaciones (`sgw_roi`, `sgw_rpn`, `sgw_c0`). En total son 50 corridas, todas en la RTX 3050.
  - Se corren por rondas de semilla, así que si el plazo corta la cola, las semillas terminadas quedan completas y pareadas.
  - Si no terminan las 10 semillas, se reporta el número alcanzado. No se eligen semillas a posteriori.
- **Ablación** (familia propia, con Holm): `sgw_rpn`, `sgw_roi` y `sgw_c0` frente a `sgw`, con las mismas semillas.

## 5. Pruebas estadísticas

- **Unidad de independencia:** el grupo de escena (14 en test). Para cada modelo, el F1 por grupo se promedia sobre sus semillas.
- **Tres pruebas por contraste:**
  1. Wilcoxon pareado por grupo;
  2. *bootstrap* pareado por grupo (10.000 remuestreos), con IC 95 %;
  3. t pareado sobre las semillas pareadas, en el contraste SGW frente a Mask R-CNN (misma GPU).

  Se complementa con Friedman e Iman-Davenport con Nemenyi y diagrama de diferencia crítica.
- **Corrección:** Holm sobre toda la familia. La decisión usa el máximo de los p ajustados de las tres pruebas.
- **Tamaño del efecto:** δ de Cliff y d de Cohen pareado, con IC.
- **Potencia:** con σ̂_d de las semillas piloto (la desviación entre semillas de Mask R-CNN en el E2 es 0,012). Se reporta la diferencia mínima detectable.

## 6. Conjuntos adicionales (registrado el 2026-11-25, tag `e3-conjuntos`)

Se registra antes de descargar los datos y antes de cualquier corrida de SGW o Mask R-CNN en ellos. Ninguno se usó para diseñar SGW: κ = 0,45 y c = 4 px quedan fijos.

| Conjunto | Naturaleza | Fuente y licencia | Unidad de grupo | Partición |
|---|---|---|---|---|
| BBBC039 | Núcleos U2OS en fluorescencia (Hoechst), 200 campos de 520×696, 16 bits | Broad Bioimage Benchmark Collection (Ljosa et al., 2012), CC0 | La imagen (no hay metadatos de escena) | La oficial del archivo `metadata` (entrenamiento, validación, prueba) |
| NuInsSeg | Núcleos en histología H&E, 665 parches de 512×512 de 31 órganos humanos y de ratón | Mahbod et al., *Scientific Data* 2024, Zenodo 10518968, CC BY 4.0 | El órgano (especie + tejido) | Agrupada por órgano, 70/15/15 con `random.Random(0)` sobre los órganos |
| MinneApple | Manzanas en huerto, 670 imágenes de 1280×720 con máscaras | Häni et al., 2019 (arXiv:1909.06441); réplica de AgML en Hugging Face, MIT | La secuencia de captura (prefijo de fecha y hora del archivo); si no se puede recuperar, la imagen | Agrupada por secuencia, 70/15/15 con `random.Random(0)` |

**Reglas, fijadas hoy:**
- **Modelos y presupuesto:** SGW (κ = 0,45, c = 4) y Mask R-CNN con la configuración exacta del WDSD: 6 épocas, lote 2, tasa 0,01, coseno, AMP, volteo, anclas de 8–128 px, `min_size` 720 / `max_size` 1280 y 100 detecciones por imagen.
  - Sin ajuste por conjunto.
  - El tope de 100 detecciones recorta por igual a los dos modelos en las imágenes más densas; se reporta.
- **Semillas:** 42, 43 y 44 por modelo y conjunto (18 corridas). Desviación declarada: menos de 10 semillas, por cómputo.
- **Preprocesamiento:** cada imagen se guarda como PNG RGB de 8 bits.
  - Las de 16 bits de BBBC039 se reescalan con los percentiles 0,5–99,5 de cada imagen.
  - Las máscaras de instancia se guardan como RLE sin comprimir en COCO.
  - Si una réplica perdió las instancias (máscara binaria), se separan por componentes conexos y se declara.
- **Métrica y prueba:** la misma que en el WDSD: F1 de instancia por grupo en test, IoU de máscara ≥ 0,5.
  - Por conjunto: Wilcoxon pareado por grupo, *bootstrap* pareado por grupo y t pareado sobre semillas.
  - Familia de 3 contrastes con Holm; decide el máximo de los p ajustados.
- **Consistencia (E3 §8):** SGW tiene mayor F1 medio en al menos 2 de los 3 conjuntos y no es significativamente peor en ninguno.
- **Secundario:** recall del tercil pequeño, con cortes por percentiles de las áreas de val de cada conjunto.

**Desviaciones encontradas al preparar los datos (2026-11-25), antes de cualquier corrida en ellos:**
- **MinneApple:** la réplica de AgML trae máscaras binarias, así que se separan por componentes conexos, como estaba previsto.
  - Solo hay 10 secuencias de captura. La partición agrupada queda en 7/2/1 secuencias (547/89/34 imágenes).
  - Se mantiene esa partición para evitar fuga, pero con 1 secuencia en test no hay prueba por grupo posible.
  - La unidad estadística en test pasa a ser la imagen (n = 34). Son imágenes de una misma secuencia, correlacionadas, y se reporta como amenaza.
- **BBBC039:** los núcleos no son pequeños (área mediana 627 px, frente a 72 px de los tornillos) y hay hasta 231 por imagen.
  - El tope de 100 detecciones recorta a los dos modelos en las imágenes densas. Se mantiene, porque cambiarlo sería ajustar por conjunto.
  - Sirve como control: un conjunto donde la hipótesis del mecanismo predice poca o ninguna ventaja.
- `preparar_extra.py` guarda el resumen y el SHA-256 de cada JSON en `resumen_preparacion.json` de cada conjunto.

## 7. Reglas de exclusión

Igual que en el E2: solo por fallo técnico (caída, falta de memoria o pérdida NaN), con un reintento con la misma semilla. Toda exclusión se reporta.
