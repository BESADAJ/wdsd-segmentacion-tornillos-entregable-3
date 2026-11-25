# Revisión de literatura y novedad

## 1. Búsqueda de novedad (2026-11-25)

Se buscó si la asignación SGW ya existe. SGW usa una similitud de Wasserstein entre gaussianas de cajas, normalizada por la escala de cada objeto y con piso, y la aplica en RPN y en RoI de Mask R-CNN para segmentación de instancias. La búsqueda se repite antes de la entrega final, con fecha.

| N.° | Cadena | Fuentes | Resultado relevante |
|---|---|---|---|
| 1 | Wasserstein distance label assignment Mask R-CNN instance segmentation tiny objects RPN | Web (arXiv, ar5iv, ECCV) | NWD, NWD-RKA, RFLA, SimD. Ninguno en segmentación de instancias ni en Mask R-CNN |
| 2 | screw detection deep learning WEEE disassembly instance segmentation small screws | Web (NSF PAR, SciTePress, arXiv) | Detectores de cajas (YOLOv4, CNN con Hough). Ninguna asignación propia para tornillos diminutos |
| 3 | scale-adaptive normalized Wasserstein distance per-object scale tiny object detection label assignment | Web (arXiv) | NWD; CoSWA-YOLOv12 (peso de NWD que depende del tamaño); pérdidas adaptativas a la escala |
| 4 | "Gaussian Wasserstein" OR "NWD" label assignment "two-stage" tiny objects RFLA DotD | Web (arXiv) | NWD, NWD-RKA, DotD, RFLA, SimD |
| 5 | WDSD WEEE Disassembly Screw Dataset Kalitsios Mask R-CNN screws | Web (arXiv, Roboflow, ACM) | Ningún trabajo además del paper guía usa WDSD con una asignación modificada |

**Bola de nieve:** desde NWD hacia adelante (NWD-RKA, RFLA, SimD, CoSWA-YOLOv12) y hacia atrás (DotD, GIoU).

**Pendiente:**
- la verificación por un tercero (un integrante que no diseñó SGW repite las cadenas 1, 3 y 5);
- repetir las cadenas en Google Scholar y Semantic Scholar.

## 2. Tabla comparativa con métodos afines

| Método | Similitud | Normalización por tamaño | Piso | Dónde se aplica | Tarea | Demostración |
|---|---|---|---|---|---|---|
| IoU (Mask R-CNN) | Área de intersección sobre área de unión | Implícita; sensibilidad 2/a, no acotada (proposición 2) | — | RPN y RoI | Cajas y máscaras | — |
| DotD (Xu et al., 2021) | Distancia entre centros sobre el tamaño medio de los objetos del conjunto | Constante del conjunto | No | Asignación | Cajas | No |
| NWD (Wang et al., 2021) | exp(−W₂/C), con W₂ entre gaussianas de cajas | **Constante global C** del conjunto | No | RPN, NMS, pérdida | Cajas | No |
| NWD-RKA (Xu et al., 2022) | NWD con asignación por ranking (top-k) | Constante global | No | Asignación | Cajas | No |
| RFLA (Xu et al., 2022) | KL entre campo receptivo gaussiano y objeto | Del campo receptivo | No | Asignación | Cajas | No |
| SimD (Shi et al., 2024) | exp(−(sim_loc + sim_forma)); diferencias divididas por (w_g + w_a)/m | **Por par**, con constantes m y n del conjunto | No | RPN (Faster, Cascade, DetectoRS) | Cajas | No |
| CoSWA-YOLOv12 (Issah y Mukamakuza, 2026) | (1 − r)·CIoU + r·NWD, con r una sigmoide del tamaño absoluto | Compuerta por el tamaño absoluto (τ = 56 px); NWD con C global | No | Asignador de YOLO (una etapa) | Cajas y máscaras (parásitos) | No |
| **SGW (este trabajo)** | exp(−W₂/D(g)), con D(g) = κ(√(w_g h_g) + c) | **Por objeto**: escala propia de cada objeto real | **Sí, c = 4 px** (paso de P2) | **RPN y cabeza RoI** de Mask R-CNN | **Máscaras de instancia** | **Sí**: teorema 1 (rango, Lipschitz uniforme, gradiente donde IoU = 0, límite NWD) y proposición 2 |

## 3. Declaración de novedad y clasificación de la diferencia

**Qué no es nuevo:**
- modelar las cajas como gaussianas y compararlas con Wasserstein (NWD);
- reemplazar el IoU en la asignación de una arquitectura de dos etapas (NWD, SimD);
- normalizar por el tamaño (SimD por par con una constante del conjunto; CoSWA con una compuerta).

**Qué es propio de SGW** (diferencia de mecanismo, no solo de aplicación):
1. **La escala de normalización es la del objeto real, con piso:** D(g) = κ(√A_g + c).
   - En NWD, C es global. En SimD, la escala depende del par y de constantes del conjunto, y no tiene piso.
   - Sin el piso, la sensibilidad de la similitud a un error de 1 px crece sin límite al encoger el objeto, igual que con el IoU.
   - El piso da la cota de Lipschitz 1/(√2 κ c), uniforme en el tamaño. Es la propiedad que se demuestra (capítulo 5).
2. **κ se calibra analíticamente** con el IoU en el límite de objetos grandes (κ₀ = 0,45), sin datos. Así los umbrales 0,7/0,3 de RPN y 0,5 de RoI conservan su significado.
   - SimD calcula m y n sobre train; NWD ajusta C al conjunto.
3. **SGW se aplica a la vez en RPN y en la cabeza RoI de Mask R-CNN para segmentación de instancias.**
   - Ninguno de los métodos encontrados se evaluó en segmentación de instancias de dos etapas.
   - CoSWA trata máscaras, pero en un detector de una etapa y sin normalizar la escala de W₂.
4. **Demostración formal:** ninguno de los trabajos afines demuestra propiedades de su métrica.
   - CoSWA afirma que la similitud de Wasserstein es invariante a la escala, pero con C global no lo es: NWD(λb₁, λb₂) = exp(−λW₂/C).
   - SGW sí cumple la identidad de escala s_c(λg, λb) = s_{c/λ}(g, b) (teorema 1d).

**Clasificación:** mecanismo **transferido** (NWD, de detección aérea) con **adaptación algorítmica propia** (normalización por objeto con piso, calibración analítica y aplicación en las dos etapas de segmentación de instancias) y **propiedad demostrada**. Por la tabla de topes del enunciado, R1 y R2 se cumplen y R3 no (capítulo 7): el tope es 4,0.

**Relación con el paper guía** (Kalitsios et al., 2022: Mask R-CNN ResNet-101 en WDSD, con asignación por IoU):
- SGW conserva la familia y la arquitectura del paper guía y cambia solo la asignación.
- La limitación del paper verificada en el E2 (fuga por la partición aleatoria) no la ataca SGW: la corrige la partición agrupada.
- SGW ataca la limitación medida en el E2 sobre los 14 modelos: el recall del tercil pequeño no pasa de 0,37 en ninguno.

## Referencias de este capítulo

- Wang, J., Xu, C., Yang, W. y Yu, L. (2021). A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. arXiv:2110.13389.
- Xu, C., Wang, J., Yang, W., Yu, H., Yu, L. y Xia, G.-S. (2022). Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark. *ISPRS J. Photogramm. Remote Sens.* arXiv:2206.13996.
- Xu, C., Wang, J., Yang, W. y Yu, L. (2021). Dot Distance for Tiny Object Detection in Aerial Images. CVPR Workshops.
- Xu, C. et al. (2022). RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection. ECCV. arXiv:2208.08738.
- Shi, S., Fang, Q., Xu, X. y Zhao, T. (2024). Similarity Distance-Based Label Assignment for Tiny Object Detection. IROS. arXiv:2407.02394.
- Issah y Mukamakuza (2026; iniciales por completar). CoSWA-YOLOv12: Scale-Invariant Tiny Object Detection and Segmentation of Malaria Parasites. arXiv:2609.29527.
- Kalitsios, G., Lazaridis, L., Psaltis, A., Axenopoulos, A. y Daras, P. (2022). Vision-Enhanced System for Human-Robot Disassembly Factory Cells: Introducing a New Screw Dataset. ICRCV. DOI 10.1109/ICRCV55858.2022.9953178.
