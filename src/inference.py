"""Inferencia en CPU con ONNX Runtime sobre los artefactos exportados (artifacts/*.onnx).

- Modo servidor (WDSD_INFERENCIA_VIVO=1, local/Docker): carga bajo demanda de un solo modelo a la vez. El pico del
  proceso es de ~0,9 GB (medido) y no cabe en el plan gratuito de Render (512 MB); allí WDSD_INFERENCIA_VIVO=0 y el
  mismo .onnx se ejecuta en el navegador (assets/prediccion_navegador.js). La validación y el aviso fuera de dominio
  de este módulo se usan en los dos modos.
- Validación de la entrada: tipo de archivo, tamaño y dimensiones; el contenido subido nunca se ejecuta.
- Aviso fuera de dominio: distancia de Mahalanobis de rasgos simples de la imagen a los de train
  (data/ood_referencia.json; los rasgos coinciden con experiments/datos_dashboard.py:rasgos).
- Incertidumbre: intervalo conforme del número de tornillos (data/conforme.json).
No importa torch: en despliegue solo hacen falta numpy, Pillow y onnxruntime.
"""
from __future__ import annotations

import base64
import gc
import io
import time

import numpy as np
from PIL import Image

from . import data

MAX_BYTES = 8 * 2**20
FORMATOS = {"PNG", "JPEG", "BMP", "TIFF"}
H, W = 720, 1280


class EntradaInvalida(ValueError):
    pass


def decodificar(contenido: str) -> Image.Image:
    """contenido = 'data:<tipo>;base64,<datos>' de dcc.Upload."""
    try:
        cabecera, b64 = contenido.split(",", 1)
        crudo = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise EntradaInvalida("El archivo no se pudo leer.") from e
    if len(crudo) > MAX_BYTES:
        raise EntradaInvalida(f"El archivo pesa {len(crudo) / 2**20:.1f} MB; el máximo es 8 MB.")
    try:
        im = Image.open(io.BytesIO(crudo))
        im.verify()
        im = Image.open(io.BytesIO(crudo))
    except Exception as e:
        raise EntradaInvalida("No es una imagen válida (se aceptan PNG, JPEG, BMP o TIFF).") from e
    if im.format not in FORMATOS:
        raise EntradaInvalida(f"Formato {im.format} no admitido.")
    if min(im.size) < 64 or max(im.size) > 6000:
        raise EntradaInvalida(f"Dimensiones {im.size[0]}x{im.size[1]} fuera de rango (64 a 6000 px).")
    return im.convert("RGB")


def rasgos(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    g = x.mean(-1)
    gy, gx = np.gradient(g)
    return np.array([*x.reshape(-1, 3).mean(0), *x.reshape(-1, 3).std(0), np.hypot(gx, gy).mean(),
                     rgb.shape[1] / rgb.shape[0]])


def fuera_de_dominio(im: Image.Image) -> dict:
    ref = data.jsn("ood_referencia.json")
    r = ref["reduccion"]
    x = rasgos(np.asarray(im.reduce(r) if min(im.size) >= 64 * r else im))
    d = x - np.array(ref["media"])
    dist = float(np.sqrt(d @ np.array(ref["cov_inv"]) @ d))
    return {"distancia": dist, "umbral": ref["umbral_p99"], "fuera": dist > ref["umbral_p99"],
            "aspecto_distinto": abs(im.size[0] / im.size[1] - W / H) > 0.05}


_SESION: dict = {}


def sesion(modelo: str):
    """Un solo modelo en memoria. Al cambiar de modelo se libera el anterior ANTES de cargar el nuevo: con
    lru_cache(maxsize=1) el nuevo se crea mientras el viejo sigue vivo y el pico de memoria suma los dos
    (medido con el guion de prueba de humo: 989 MB frente a 894 MB)."""
    if modelo not in _SESION:
        _SESION.clear()
        gc.collect()
        _SESION[modelo] = _crear_sesion(modelo)
    return _SESION[modelo]


def _crear_sesion(modelo: str):
    import onnxruntime as ort
    meta = data.modelos_onnx()[modelo]
    op = ort.SessionOptions()
    op.intra_op_num_threads = 2
    op.enable_cpu_mem_arena = False  # menos memoria residente en el plan gratuito
    op.log_severity_level = 3  # sin avisos de forma de salida (la forma fija del grafo es solo nominal)
    return ort.InferenceSession(str(data.ARTIFACTS / meta["archivo"]), op, providers=["CPUExecutionProvider"])


def predecir(modelo: str, im: Image.Image) -> dict:
    meta = data.modelos_onnx()[modelo]
    original = im.size
    x = np.asarray(im.resize((W, H), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
    t0 = time.perf_counter()
    cajas, _, scores, mascaras = sesion(modelo).run(None, {"imagen": x})
    ms = (time.perf_counter() - t0) * 1000
    u = meta["umbral_score"]
    keep = scores >= u
    conf = data.jsn("conforme.json")
    n = int(keep.sum())
    radio = conf["radio"]["0.9"]
    return {"modelo": modelo, "cajas": cajas[keep], "scores": scores[keep], "mascaras": mascaras[keep, 0],
            "todas_scores": scores, "umbral": u, "latencia_ms": ms, "n": n,
            "intervalo_90": (max(0, n - radio), n + radio), "cobertura_test": conf["cobertura_test"]["0.9"],
            "tamano_original": original}
