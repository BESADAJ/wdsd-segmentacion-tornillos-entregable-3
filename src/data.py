"""Carga y caché de los datos livianos del dashboard (data/ y results/). Nunca entrena ni lee datos crudos."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(os.environ.get("WDSD_APP_RAIZ", Path(__file__).resolve().parents[1]))
DATA, RESULTS, ARTIFACTS = RAIZ / "data", RAIZ / "results", RAIZ / "artifacts"
VERSION = "1.0.0"
ACTUALIZADO = "2026-11-25"


@lru_cache(maxsize=None)
def csv(nombre: str, carpeta: str = "data") -> pd.DataFrame:
    return pd.read_csv((DATA if carpeta == "data" else RESULTS / carpeta) / nombre)


@lru_cache(maxsize=None)
def jsn(nombre: str, carpeta: str = "data"):
    return json.loads(((DATA if carpeta == "data" else RESULTS / carpeta) / nombre).read_text(encoding="utf8"))


def tabla_maestra() -> pd.DataFrame:
    return csv("tabla_maestra.csv")


def eda_anotaciones() -> pd.DataFrame:
    return csv("eda_anotaciones.csv")


def eda_imagenes() -> pd.DataFrame:
    return csv("eda_imagenes.csv")


def confirmatorio(nombre: str):
    return jsn(nombre, "confirmatorio") if nombre.endswith(".json") else csv(nombre, "confirmatorio")


def texto(nombre: str) -> str:
    return (RAIZ / nombre).read_text(encoding="utf8")


# Inferencia en vivo. Una predicción ONNX a 1280x720 en CPU sube el proceso a ~0,9 GB (medido con
# experiments/prueba_humo_despliegue.py: 894 MB de pico), que no caben en los 512 MB del plan gratuito de Render. Hay
# dos modos:
#   - servidor (WDSD_INFERENCIA_VIVO=1, por defecto en local y en Docker): el servidor ejecuta ONNX Runtime.
#   - navegador (WDSD_INFERENCIA_VIVO=0, en Render): el servidor solo entrega el .onnx y onnxruntime-web lo ejecuta en
#     el navegador de quien abre el dashboard (assets/prediccion_navegador.js). Mismo archivo, mismas salidas.
INFERENCIA_VIVO = os.environ.get("WDSD_INFERENCIA_VIVO", "1") != "0"
PICO_MEMORIA_MB = 900


def modelos_exportados() -> dict[str, dict]:
    """Modelos con artefacto ONNX presente: nombre -> metadata.json (independiente del modo de inferencia)."""
    salida = {}
    for m in sorted(ARTIFACTS.glob("*.metadata.json")):
        meta = json.loads(m.read_text(encoding="utf8"))
        if (ARTIFACTS / meta["archivo"]).exists():
            salida[meta["modelo"]] = meta
    return salida


def modelos_onnx() -> dict[str, dict]:
    """Modelos que el servidor ejecuta en vivo (vacío en modo navegador)."""
    return modelos_exportados() if INFERENCIA_VIVO else {}
