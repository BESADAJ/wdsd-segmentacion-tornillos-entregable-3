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


# Pico de memoria medido de una inferencia ONNX a 1280x720 en CPU: ~660 MB (240 MB de pesos + activaciones). No cabe en
# los 512 MB del plan gratuito de Render: allí WDSD_INFERENCIA_VIVO=0 y la app muestra resultados precalculados.
INFERENCIA_VIVO = os.environ.get("WDSD_INFERENCIA_VIVO", "1") != "0"
PICO_MEMORIA_MB = 660


def modelos_onnx() -> dict[str, dict]:
    """Modelos exportados disponibles para inferencia en vivo: nombre -> metadata.json."""
    salida = {}
    if not INFERENCIA_VIVO:
        return salida
    for m in sorted(ARTIFACTS.glob("*.metadata.json")):
        meta = json.loads(m.read_text(encoding="utf8"))
        if (ARTIFACTS / meta["archivo"]).exists():
            salida[meta["modelo"]] = meta
    return salida
