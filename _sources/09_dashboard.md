# Dashboard y despliegue (Bloques E y F)

## Arquitectura

La app es multipágina, hecha con Dash, `use_pages=True` y dash-bootstrap-components con el tema FLATLY.

```
app.py            crea la app, la barra fija, el pie con la política de privacidad y expone server
pages/            contexto.py · eda.py · modelos.py · original.py · ficha.py (un archivo y sus callbacks por página)
src/data.py       carga y caché de data/ y results/ (la tabla maestra es la misma que usa el libro)
src/plots.py      figuras reutilizables con una plantilla y una paleta de 5 colores
src/inference.py  ONNX Runtime en CPU, validación de entradas, aviso fuera de dominio e intervalo conforme
src/explain.py    explicación local: máscaras y mapa de confianza por píxel
artifacts/        modelos ONNX (pesos en FP16) y metadata.json
data/             datos livianos precalculados (experiments/datos_dashboard.py)
```

La app nunca importa `training/` ni torch: en despliegue solo usa numpy, Pillow, pandas, plotly y onnxruntime.

## Decisiones de diseño

- **Inferencia en vivo:** para SGW y Mask R-CNN, los dos modelos exportados a ONNX. En el despliegue gratuito corre en el navegador de quien abre el dashboard (sección «Memoria e inferencia en vivo»); en local o con Docker, en el servidor, con un solo modelo en memoria a la vez.
  - Los otros 12 modelos se muestran con resultados precalculados (capítulo 3 §4).
- **Explicación local:** en segmentación es la máscara misma. Se muestra cada instancia con su score y un mapa de confianza (probabilidad de máscara × score).
  - Grad-CAM no es posible sobre el grafo ONNX sin gradientes. Por eso se muestran precalculados los mapas de oclusión del E2.
- **Incertidumbre:** intervalo conforme del 90 % para el número de tornillos (± 2, calibrado en val). Se muestra con su cobertura empírica en test (82 %): las escenas no son intercambiables.
- **Fuera de dominio:** distancia de Mahalanobis de 8 rasgos simples (color medio, contraste, gradiente y aspecto) a los de train, con umbral en el percentil 99. También avisa si la imagen no es 16:9.
- **Robustez y seguridad:**
  - límite de 8 MB y de 64 a 6000 px por imagen;
  - validación del formato con `PIL.verify`;
  - los errores se muestran con `dbc.Alert`;
  - las imágenes subidas no se guardan ni se registran;
  - no se exponen rutas ni credenciales.
- **Identidad visual:** tema FLATLY con cinco colores (tinta, naranja solo para SGW, azul para Mask R-CNN, verde para la anotación real y gris), una retícula de 1280 px para barra, contenido y pie, tarjetas con la interpretación junto a cada gráfico y una plantilla Plotly única con coma decimal y tamaños legibles en proyector (`assets/style.css`, `src/ui.py`, `src/plots.py`). Las particiones usan tonos de la tinta para no confundirse con los modelos.
- **Accesibilidad:** el naranja de SGW y el azul de Mask R-CNN también se distinguen por la forma del marcador y por el trazo; las imágenes tienen texto alternativo.
- **Consistencia con el informe:** `tests/test_dashboard.py` verifica que las cifras de la tabla maestra coinciden con `results/confirmatorio` y con el capítulo 7.

## Despliegue en Render

1. El repositorio en GitHub contiene `Dockerfile`, `render.yaml`, `requirements.txt` y `artifacts/`.
2. En Render: *New → Blueprint*, elegir el repositorio; `render.yaml` crea el servicio web gratuito con *runtime* Docker y despliegue automático en cada *push* a `main`.
3. Prueba de humo: `python experiments/prueba_humo_despliegue.py https://<servicio>.onrender.com --salida results/despliegue/prueba_humo.md`.
   - Carga las cinco páginas, dispara los callbacks, comprueba la entrada inválida, las cabeceras COOP/COEP, onnxruntime-web y los dos `.onnx`.
   - La predicción en el navegador se comprueba a mano: *Modelos → Sube tu imagen*, con una imagen de test.
   - En CI corren los mismos controles sobre la app (`tests/test_dashboard.py`).
   - En local: `docker build -t wdsd-sgw . && docker run -p 8050:8050 wdsd-sgw`.

**URL pública:** https://wdsd-segmentacion-tornillos-entregable-3.onrender.com (plan gratuito; desplegado el 27-nov-2026, prueba de humo APROBADA en `results/despliegue/prueba_humo.md`).

```{admonition} Pendiente
:class: warning
- Capturas de pantalla de las cinco páginas, una vez desplegado.
```

## Memoria e inferencia en vivo

Una predicción con el ONNX de SGW sobre una imagen de 1280×720 sube el proceso del servidor a unos 0,9 GB (894 MB de pico, medido con la prueba de humo):
- unos 176 MB son los pesos: se guardan en FP16 pero ONNX Runtime los convierte a FP32 al cargar (43,9 M parámetros × 4 bytes);
- el resto son activaciones: cada mapa del FPN a 1280×720 ocupa unos 60 MB, y las ~1000 propuestas del RPN pasan por el *RoI pooling*.

Se probó a reducir la memoria sin cambiar el modelo:
- desactivar el *arena* y el patrón de memoria de ONNX Runtime;
- usar un hilo y optimización de grafo básica;
- liberar el modelo anterior antes de cargar otro (de 989 a 894 MB).

Ninguna opción baja de unos 450 MB solo en ONNX Runtime, sin contar la app. Recortar las propuestas del RPN de 1000 a 150 bajaría el pico a 640 MB, pero cambia el modelo evaluado y aun así no cabe.

El plan gratuito de Render tiene 512 MB. Por eso, en el despliegue (`WDSD_INFERENCIA_VIVO=0`), la predicción en vivo corre **en el navegador**:
- el servidor solo entrega el mismo archivo ONNX (`/modelo/<nombre>.onnx`, 84 MB, que queda en la caché del navegador) y onnxruntime-web 1.20.1 (WebAssembly) lo ejecuta en el equipo de quien abre la página (`assets/prediccion_navegador.js`);
- el servidor sigue validando la imagen, dando el aviso fuera de dominio y redimensionándola a 1280×720 con Pillow; el navegador recibe esa imagen en PNG (sin pérdida), así que el modelo ve los mismos píxeles en los dos modos;
- las cabeceras COOP/COEP habilitan 4 hilos de WebAssembly: una predicción tarda ~4,3 s (Chromium, 4 hilos) frente a ~14 s con uno;
- el navegador necesita ~1,3 GB de memoria libre, así que el modo está pensado para un computador, no para un teléfono.

Equivalencia: con las imágenes 14 y 49 de la galería, el navegador y el servidor dan las mismas detecciones y los mismos scores con ambos modelos (imagen 14: SGW 8 tornillos, Mask R-CNN 9; imagen 49: 1 y 0). Una primera versión redimensionaba en el *canvas* del navegador y daba un tornillo de más con SGW en la imagen 14 (9 frente a 8): por eso el redimensionado quedó en Python.

Memoria del servidor en modo navegador: 138 MB de pico durante la prueba de humo más dos predicciones (SGW y Mask R-CNN), muy por debajo de los 512 MB.

Reducir la resolución no es alternativa: al 50 % el F1 cae a 0,14 (capítulo 8).
