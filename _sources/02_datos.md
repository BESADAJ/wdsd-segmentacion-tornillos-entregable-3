# Datos: EDA y auditoría

El EDA completo por instancias está en el E1 (`notebooks/02_eda_instancias.ipynb`) y en el E2 (`informe/eda_trazabilidad.md`). Aquí se resume lo que condiciona el modelo original. Las cifras salen de `data/eda_*.csv`, generados por `experiments/datos_dashboard.py`, y se pueden explorar con filtros en la página EDA del dashboard.

## Data Card

| Campo | Valor |
|---|---|
| Fuente | WEEE Disassembly Screw Dataset (WDSD), Visual Computing Lab, CERTH-ITI (Kalitsios et al., 2022) |
| Licencia | No declarada: uso académico, sin redistribuir las imágenes |
| Observaciones | 945 imágenes RGB de 1280×720, con 4.398 tornillos anotados como polígonos COCO |
| Tarea | Segmentación de instancias de una clase |
| Partición | Agrupada por escena: 744 / 93 / 108 imágenes, con 14 grupos en test |
| Sesgos | Pocos tipos de aparato; los tornillos pequeños se concentran en pocas escenas (el grupo 6 tiene 59 de los 155 pequeños de test) |

## Lo que muestran los datos y su implicación

1. **Tamaño.** El 98,8 % de los tornillos mide menos de 32² px (COCO small), y el lado mediano es de 10,6 px.
   - Los cortes de tercil (de las áreas de val) son 75 y 157 px.
   - **Implicación:** con un error de 1–2 px, el IoU de una propuesta con un tornillo diminuto cae por debajo de 0,5. Es la limitación que ataca SGW (capítulo 5).
2. **Densidad.** Todas las imágenes tienen al menos un tornillo (mediana 4, máximo 17).
   - **Implicación:** el tope de 100 detecciones nunca recorta en WDSD. Sí lo hace en dos de los conjuntos adicionales (capítulo 8).
3. **Forma.** La relación de aspecto mediana de las cajas es 1,13 (del 10 % al 90 %: de 0,89 a 1,50).
   - **Implicación:** las anclas de aspecto 0,5, 1 y 2 la cubren.
4. **Cambio de distribución.** El brillo medio es casi igual en las tres particiones (mediana 112), y la validación adversarial no separa train de test (AUC 0,33, nulo p95 0,55).
   - **Implicación:** el cambio entre escenas está en el contenido, no en la iluminación global.

## Auditoría de fuga (E2 §10.1)

| Verificación | Resultado | ¿Aprueba? |
|---|---|---|
| Grupos de escena compartidos entre particiones | Ninguno | Sí |
| Duplicados exactos (SHA-256) | 0 | Sí |
| Casi gemelos (pHash, Hamming ≤ 10) | 12 % de test en la agrupada (58 % en la oficial) | No (límite: 5 %) |
| Validación adversarial train/test | AUC 0,33 (p de permutación 1,0) | Sí |

El criterio de casi gemelos no se cumple. Se reporta y se mantiene la partición, que es la que congeló el E1.
