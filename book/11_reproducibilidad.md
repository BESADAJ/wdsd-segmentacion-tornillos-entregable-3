# Reproducibilidad y bibliografía

## Entornos

| Uso | Entorno | Definición |
|---|---|---|
| Entrenamiento y análisis | Python 3.11, torch 2.11 + CUDA 12.8, torchvision 0.26 (`D:\wdsd_envs\wdsd_dl`) | `requirements-train.txt` (versiones exactas) |
| Dashboard (inferencia) | Python 3.11, onnxruntime 1.20.1, Dash 2.18.2 | `requirements.txt` y `Dockerfile` (contenedor) |
| CI | GitHub Actions, Ubuntu, torch en CPU | `.github/workflows/ci.yml`: pytest en cada *push* |

**Hardware:**
- Las 50 corridas finales del modelo original, la búsqueda de κ, la robustez y los conjuntos adicionales: RTX 3050 Laptop de 6 GB.
- Los rivales del E2: RTX 3060 Laptop.

**Determinismo:** semillas fijas (`comun.fijar_semilla`). La GPU no es determinista bit a bit; la variación entre semillas se reporta.

## Orden de ejecución

```
python experiments/preparar_datos.py <WDSD> D:/wdsd_datos/datos_inst        # COCO de la partición congelada
python training/entrenar_sgw.py buscar --datos D:/wdsd_datos/datos_inst       # κ (solo val)
python training/cola_e3.py --datos D:/wdsd_datos/datos_inst --semillas 42-51  # 50 corridas finales
python experiments/terciles_e3.py D:/wdsd_datos/datos_inst
python experiments/analisis_confirmatorio.py                                  # capítulo 7
python experiments/robustez_e3.py --datos D:/wdsd_datos/datos_inst            # capítulo 8
python experiments/preparar_extra.py <crudo> D:/wdsd_datos/extra              # conjuntos adicionales
python training/cola_extra.py                                                 # 18 corridas
python training/exportar_onnx.py --datos D:/wdsd_datos/datos_inst            # ONNX, ida y vuelta, costo
python training/referencia_ida_vuelta.py                                      # referencia para CI
python experiments/datos_dashboard.py --datos D:/wdsd_datos/datos_inst        # data/ del dashboard
pytest -q
```

## Tabla maestra de experimentos

| Experimento | Corridas | Datos usados | Salida |
|---|---|---|---|
| Piloto de asignación | 0 (sin entrenar) | train | `results/piloto/` |
| Búsqueda de κ | 3 (κ ∈ {0,225; 0,45; 0,9}, semilla 42, 2 épocas) | train / val | `D:/wdsd_e3/results/busqueda/` |
| Finales y ablación | 50 (5 variantes × semillas 42–51) | train / val / test (un acceso) | `D:/wdsd_e3/results/corridas/`, `results/confirmatorio/` |
| Robustez | 20 modelos × 9 perturbaciones (inferencia) | test | `D:/wdsd_e3/results/robustez/` |
| Conjuntos adicionales | 18 (3 conjuntos × 2 modelos × 3 semillas) | BBBC039, NuInsSeg, MinneApple | `D:/wdsd_e3/results_extra/` |
| Exportación | 2 modelos | test | `artifacts/`, `results/exportacion/` |

**Fechas en git:**
- `e3-protocolo`: protocolo congelado el 2026-11-23, antes de las corridas finales.
- `e3-conjuntos`: conjuntos adicionales registrados el 2026-11-25, antes de descargarlos.

## Lista de verificación

| Ítem | Estado |
|---|---|
| Preregistro fechado antes de las corridas confirmatorias | ✅ tag `e3-protocolo` |
| Conjuntos adicionales elegidos antes de ver resultados | ✅ tag `e3-conjuntos`; desviaciones declaradas antes de entrenar |
| Hashes de las particiones | ✅ protocolo §3 y `resumen_preparacion.json` de cada conjunto |
| Semillas y resultados por corrida | ✅ `resumen.json` por corrida; `f1_grupo_por_semilla.csv` |
| Entorno en contenedor | ✅ para el dashboard; 🟡 el entrenamiento usa un entorno conda con versiones exactas |
| Pruebas automáticas en CI | ✅ modelo, exportación, app y consistencia |
| Análisis de potencia a priori | 🟡 no se registró antes; se reporta como exploratorio |
| Repositorio archivado con DOI | ❌ pendiente (Zenodo al publicar) |

## Bibliografía

Las referencias en BibTeX están en `book/referencias.bib`.
