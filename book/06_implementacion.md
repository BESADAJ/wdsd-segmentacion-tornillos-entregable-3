# Modelo original: implementación

## 1. Código

| Archivo | Qué contiene |
|---|---|
| `src/original_model.py` | Implementación del mecanismo |
| `training/entrenar_sgw.py` | Entrenamiento: reutiliza sin cambios el trabajador de Mask R-CNN del E2 y solo reemplaza el constructor del modelo |
| `training/cola_e3.py` | Cola de las corridas |

El contenido de `src/original_model.py`, con la complejidad de cada parte:

| Pieza | Qué hace | Complejidad |
|---|---|---|
| `parametros_gaussianos(cajas)` | Lleva (x1, y1, x2, y2) a p = (cx, cy, w/2, h/2) | O(N) |
| `distancia_w2(ref, cajas)` | Matriz de W₂ entre G cajas reales y B propuestas: la norma euclídea de p(g) − p(b), con ε = 10⁻¹² dentro de la raíz para que el gradiente sea finito | O(G·B), igual que `box_iou` |
| `SimilitudSGW(kappa, piso)` | Módulo sin parámetros entrenables; s = exp(−W₂/D(g)) con D(g) = κ(√(w_g h_g) + c). Valida κ > 0 y c > 0 y maneja entradas vacías. Incluye `cota_lipschitz()` | — |
| `_asignar_propuestas_sgw` | Copia de la lógica de `RoIHeads.assign_targets_to_proposals` de torchvision, con la similitud en lugar de `box_iou` | — |
| `aplicar_sgw(modelo, ...)` | Reemplaza `rpn.box_similarity` y enlaza `_asignar_propuestas_sgw` como método de `roi_heads`. Admite activar solo RPN, solo RoI o cambiar el piso (ablación) | — |
| `mask_rcnn_base` | El Mask R-CNN del E1/E2, con un respaldo idéntico si el código del E2 no está disponible (CI) | — |

**Costo.** Por par, SGW hace del orden de 15 operaciones elementales (restas, cuadrados, una raíz y una exponencial), del mismo orden que el IoU.
- Mediana del tiempo de entrenamiento por corrida en la misma GPU: 30,2 min para SGW y 29,0 min para Mask R-CNN (+4 %). La media de SGW (31,3 min) sube por la corrida de la semilla 50, que tardó 38,9 min por una causa no identificada (posible suspensión breve o regulación térmica de la GPU).
- En inferencia, SGW no cuesta nada: la red es la misma.

## 2. Adaptación frente a la fuente

**NWD** (Wang et al., 2021) define NWD = exp(−W₂/C), con una C global ajustada al conjunto. Se aplica en la asignación de RPN, en NMS y en la pérdida de un detector de cajas. SGW cambia cuatro cosas:

1. **Escala por objeto con piso** en lugar de C global. Esto da la cota de Lipschitz uniforme del teorema 1.
2. **κ calibrado** para que s coincida con IoU = 0,5 en el límite de objetos grandes. Así se mantienen los umbrales de torchvision: 0,7/0,3 en RPN y 0,5 en RoI.
3. **Se aplica también en la cabeza RoI** (muestreo de las propuestas positivas que entrenan las máscaras), no solo en RPN.
4. **No se toca NMS ni la pérdida:** el cambio es solo de asignación, para atribuir el efecto con la ablación.

No se copió código de NWD ni de MMDetection. La declaración de procedencia y de uso de IA está en `experiments/procedencia.md`.

## 3. Pruebas unitarias

`tests/test_original_model.py` tiene 17 pruebas. Pasan en CPU, y la de sobreajuste solo corre si hay CUDA:
- **Rango e identidad:** s ∈ [0, 1] y s(g, g) = 1.
- **W₂ es euclídea en p.**
- **Cota de Lipschitz**, con 4 pares (κ, c).
- **Gradiente no nulo** con cajas disjuntas, donde el IoU es 0.
- **Identidad de escala** y límite NWD.
- **Tolerancia en tornillos diminutos:** con un desplazamiento de 2 px, IoU < 0,5 ≤ s.
- **Calibración** con el IoU en objetos grandes (±0,06).
- **Entradas inválidas y vacías.**
- **Integración con Mask R-CNN:**
  - la asignación queda reemplazada;
  - todos los parámetros reciben gradiente;
  - el entrenamiento es reproducible con la misma semilla;
  - guardar y cargar conserva las salidas;
  - el modelo sobreajusta un lote pequeño.

Además:
- `tests/test_exportacion.py` hace la ida y vuelta del ONNX;
- `tests/test_dashboard.py` prueba la app y la consistencia de sus cifras con el informe.

## 4. Ablación

La ablación (capítulo 7 §4) muestra que las tres variantes (SGW solo en RPN, solo en RoI y sin piso) quedan por debajo de SGW completo, entre 0,009 y 0,015 de F1, sin diferencias significativas tras Holm.

Contando todos los tornillos juntos:
- la cabeza RoI aporta casi toda la ganancia en tornillos pequeños (recall 0,383, frente a 0,385 del modelo completo);
- sin el piso, el recall pequeño cae a 0,312, cerca del control (0,302).

## 5. Interpretabilidad del mecanismo

La explicación natural de SGW es **a quién asigna positivos**. El piloto de asignación (capítulo 5 §6, solo con train y sin entrenar) mostró:
- en la cabeza RoI, la fracción de propuestas positivas en el tercil pequeño sube de 0,44 con IoU a 0,85 con SGW;
- en RPN apenas cambia.

Esto es coherente con la ablación: el efecto viene de la cabeza RoI.

En inferencia, la arquitectura es la de Mask R-CNN, así que los mapas de oclusión del E2 (fidelidad y estabilidad entre semillas) valen para la familia. En el dashboard se comparan las máscaras de SGW y de Mask R-CNN sobre las mismas imágenes.
