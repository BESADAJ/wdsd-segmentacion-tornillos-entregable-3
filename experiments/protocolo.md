# Protocolo experimental preregistrado del modelo original (E3 §8.1)

**Estado: CONGELADO el 2026-11-23** con el tag de Git `e3-protocolo`, antes de cualquier corrida final (las finales son las únicas que tocan test).

Al congelar, la búsqueda de κ (solo val) estaba en curso: su resultado no se conocía. κ lo elige `training/cola_e3.py` aplicando la regla de §4, sin intervención manual.

Quedan **abiertos**, y se registrarán con su propia fecha antes de usarlos:
- la elección de los 3 conjuntos adicionales (§6);
- el análisis de potencia con las semillas piloto (§5).

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

## 6. Conjuntos adicionales (por decidir antes del congelamiento)

Candidatos: segmentación de instancias de objetos pequeños, con máscaras y licencia abierta (núcleos celulares tipo NuInsSeg o DSB2018; MinneApple). Se eligen antes de ver cualquier resultado de SGW en ellos y no se usan para diseñar.

## 7. Reglas de exclusión

Igual que en el E2: solo por fallo técnico (caída, falta de memoria o pérdida NaN), con un reintento con la misma semilla. Toda exclusión se reporta.
