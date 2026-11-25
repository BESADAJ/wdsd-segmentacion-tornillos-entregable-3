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

- **Inferencia en vivo:** solo para SGW y Mask R-CNN, que caben en memoria. Se cargan bajo demanda, con una caché de un modelo (`lru_cache(maxsize=1)`), porque el plan gratuito de Render tiene 512 MB.
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
- **Accesibilidad:** el naranja de SGW y el azul de Mask R-CNN también se distinguen por la forma del marcador y por el trazo; las imágenes tienen texto alternativo.
- **Consistencia con el informe:** `tests/test_dashboard.py` verifica que las cifras de la tabla maestra coinciden con `results/confirmatorio` y con el capítulo 7.

## Despliegue en Render

1. El repositorio en GitHub contiene `Dockerfile`, `render.yaml`, `requirements.txt` y `artifacts/`.
2. En Render: *New → Blueprint*, elegir el repositorio; `render.yaml` crea el servicio web gratuito con *runtime* Docker y despliegue automático en cada *push* a `main`.
3. Prueba de humo: `GET /` responde 200 y las cinco páginas cargan.
   - El mismo control corre en CI (`tests/test_dashboard.py::test_rutas_responden`).
   - En local: `docker build -t wdsd-sgw . && docker run -p 8050:8050 wdsd-sgw`.

**URL pública:** pendiente. El despliegue requiere la cuenta de Render del equipo.

```{admonition} Pendiente
:class: warning
- Capturas de pantalla de las cinco páginas, una vez desplegado.
- Medir la memoria real en Render. Si la inferencia en vivo supera los 512 MB, se desactiva con `artifacts/` vacío y la app muestra solo la galería precalculada, como ya está previsto en el código.
```
