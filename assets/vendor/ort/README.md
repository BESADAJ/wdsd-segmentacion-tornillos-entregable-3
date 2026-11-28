# onnxruntime-web 1.20.1 (copia local)

Motor de inferencia de la predicción en el navegador (`assets/prediccion_navegador.js`). Se copia aquí en vez de
cargarlo de un CDN para que la demostración no dependa de un servicio externo, y porque con las cabeceras COOP/COEP
de `app.py` el navegador exige que el `.wasm` y el `.mjs` vengan del mismo origen.

- Origen: paquete npm `onnxruntime-web@1.20.1`, carpeta `dist/` (`npm pack onnxruntime-web@1.20.1`).
- Licencia: MIT, Copyright (c) Microsoft Corporation (cabecera de `ort.wasm.min.js`).
- Misma versión que `onnxruntime` 1.20.1 de Python con la que se exportaron y validaron los artefactos.
- `app.py` excluye estos archivos de la carga automática de Dash (`assets_ignore`): solo se descargan al predecir.

| Archivo | SHA-256 |
|---|---|
| ort.wasm.min.js | e66568724f8848e57cc2a56e4bea3a5b86ce3ff81b2da11eac8ed7ab02b27bd4 |
| ort-wasm-simd-threaded.mjs | 745eb7c0ce6f18a6aa521971b2877babc7ffb27eecb58ab3bc6e5ef4692672e8 |
| ort-wasm-simd-threaded.wasm | 207d02be4591c156b0a98f024f3d58005b5b04c92274d759fb390338c63559ea |
