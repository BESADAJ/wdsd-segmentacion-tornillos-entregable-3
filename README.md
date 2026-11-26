# WDSD · Entregable 3: modelo original SGW, evaluación y dashboard

Segmentación de instancias de tornillos en el WEEE Disassembly Screw Dataset (WDSD). El modelo original, **SGW**,
reemplaza el IoU de la asignación de Mask R-CNN (RPN y RoI) por una similitud de Wasserstein gaussiana normalizada
por la escala de cada tornillo, con un piso de 4 px.

**Resultados**, protocolo preregistrado (tag `e3-protocolo`), 10 semillas:
- F1 de instancia por grupo: 0,738 frente a 0,723 de Mask R-CNN. No es significativo, y R3 no se cumple.
- Recall de tornillos pequeños: 0,385 frente a 0,302 (todos los tornillos juntos). La prueba preregistrada por escena no lo confirma.
- En tres conjuntos adicionales (BBBC039, NuInsSeg, MinneApple), SGW empata con Mask R-CNN; el criterio de consistencia no se cumple.
- Detalle en `book/07_protocolo_validacion.md` y `book/08_evaluacion_ampliada.md`.

**[Libro en línea](https://besadaj.github.io/wdsd-segmentacion-tornillos-entregable-3/)** (Jupyter Book; GitHub Actions lo publica en la rama `gh-pages` en cada cambio de `book/`).

## Estructura

```
app.py, pages/, assets/       dashboard (Dash, multipágina)
src/original_model.py         mecanismo SGW (PyTorch)
src/{data,plots,inference,explain}.py   lógica de la app (sin torch)
training/                     entrenamiento, colas, exportación ONNX
experiments/                  protocolo, bitácora, procedencia, análisis, robustez, conjuntos adicionales
artifacts/                    modelos ONNX (pesos FP16) y metadata.json
data/                         datos livianos del dashboard (tabla maestra, EDA, galería)
results/                      resultados versionados (confirmatorio, teoría, piloto, exportación)
book/                         Jupyter Book
tests/                        pytest (modelo, exportación, app y consistencia)
```

## Uso

```
pip install -r requirements.txt && python app.py                  # dashboard local (http://127.0.0.1:8050)
docker build -t wdsd-sgw . && docker run -p 8050:8050 wdsd-sgw    # en contenedor
pip install -r requirements-train.txt && pytest -q                # pruebas
```

El entrenamiento reutiliza el código del E2 (`WDSD_E2_SRC`) y los datos de la partición congelada.
El orden completo está en `book/11_reproducibilidad.md`.

## Datos

El WDSD no se redistribuye (licencia no declarada): se descarga de <https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/>.
Los conjuntos adicionales (BBBC039, NuInsSeg, MinneApple) se preparan con `experiments/preparar_extra.py`.

Autores: Ivan Berrio, Jose David Castillo y Jaime Besada. Licencia del código: MIT.
