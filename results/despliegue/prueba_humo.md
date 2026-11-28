# Prueba de humo del despliegue

- URL: https://wdsd-segmentacion-tornillos-entregable-3.onrender.com/
- Fecha: 2026-11-27 21:44:04 SA Pacific Standard Time
- Inferencia en el servidor probada (--vivo): no
- Resultado: **APROBADA**

| Verificación | Estado | Tiempo (s) | Detalle |
|---|---|---|---|
| GET / | OK | 0.53 | HTTP 200, 6 KB |
| GET /eda | OK | 0.25 | HTTP 200, 6 KB |
| GET /modelos | OK | 0.26 | HTTP 200, 6 KB |
| GET /original | OK | 0.30 | HTTP 200, 6 KB |
| GET /ficha | OK | 0.24 | HTTP 200, 6 KB |
| GET /assets/style.css | OK | 0.42 | HTTP 200, 1 KB |
| GET /assets/e2_mapas_oclusion.jpg | OK | 0.33 | HTTP 200, 109 KB |
| GET /assets/prediccion_navegador.js | OK | 0.30 | HTTP 200, 11 KB |
| GET /_dash-layout | OK | 0.40 | HTTP 200 |
| Cabeceras COOP/COEP (hilos de WebAssembly) | OK | 0.30 | COOP=same-origin, COEP=credentialless |
| HEAD /assets/vendor/ort/ort.wasm.min.js | OK | 0.31 | HTTP 200, application/javascript; charset=utf-8 |
| HEAD /assets/vendor/ort/ort-wasm-simd-threaded.mjs | OK | 0.28 | HTTP 200, text/javascript; charset=utf-8 |
| HEAD /assets/vendor/ort/ort-wasm-simd-threaded.wasm | OK | 0.24 | HTTP 200, application/wasm |
| HEAD /modelo/sgw_k0.45.onnx | OK | 0.30 | HTTP 200, 84 MB |
| HEAD /modelo/maskrcnn.onnx | OK | 0.22 | HTTP 200, 84 MB |
| Modelo desconocido → 404 | OK | 0.27 | HTTP 404 |
| Callback EDA: filtros globales | OK | 2.34 | HTTP 200, 186 KB |
| Callback Modelos: galería precalculada | OK | 0.89 | HTTP 200, 48 KB |
| Callback Modelos: diagnóstico SGW | OK | 0.66 | HTTP 200, 6 KB |
| Callback Modelos: diagnóstico Mask R-CNN | OK | 0.27 | HTTP 200, 6 KB |
| Callback Original: curvas de entrenamiento | OK | 0.40 | HTTP 200, 7 KB |
| Callback Original: robustez (escala) | OK | 0.29 | HTTP 200, 1 KB |
| Callback Original: robustez (ruido) | OK | 0.30 | HTTP 200, 4 KB |
| Callback Original: robustez (desenfoque) | OK | 0.29 | HTTP 200, 4 KB |
| Entrada inválida → aviso (no se cae) | OK | 0.24 | HTTP 200 |
