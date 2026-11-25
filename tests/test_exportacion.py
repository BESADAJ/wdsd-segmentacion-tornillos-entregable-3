"""Prueba de ida y vuelta de la exportación (E3 §5.2): el ONNX reproduce al modelo en memoria.

Las salidas de PyTorch se guardaron con training/referencia_ida_vuelta.py sobre una imagen fija del repositorio.
Tolerancia declarada: |Δscore| < 1e-3 en las detecciones emparejadas, |Δcaja| < 0,5 px, IoU de máscara >= 0,95 y el
mismo número de detecciones sobre el umbral de val. En máscaras de menos de 50 px se exige una diferencia
de a lo sumo 5 px en lugar del IoU, porque un píxel de borde cambia mucho el IoU de una máscara diminuta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training"))
ort = pytest.importorskip("onnxruntime")
MODELOS = [p.name.replace(".metadata.json", "") for p in (RAIZ / "artifacts").glob("*.metadata.json")]


@pytest.mark.parametrize("nombre", MODELOS)
def test_onnx_reproduce_al_modelo_en_memoria(nombre):
    from referencia_ida_vuelta import entrada_fija
    meta = json.loads((RAIZ / "artifacts" / f"{nombre}.metadata.json").read_text(encoding="utf8"))
    ref = np.load(RAIZ / "results" / "exportacion" / f"referencia_{nombre}.npz")
    s = ort.InferenceSession(str(RAIZ / "artifacts" / meta["archivo"]), providers=["CPUExecutionProvider"])
    cajas, _, scores, masks = s.run(None, {"imagen": entrada_fija()})
    u = meta["umbral_score"]
    assert int((scores >= u).sum()) == int((ref["scores"] >= u).sum())
    k = len(ref["scores"])
    assert len(scores) == k
    if k == 0:
        return
    assert np.abs(scores[:k] - ref["scores"][:k]).max() < 1e-3
    assert np.abs(cajas[:k] - ref["boxes"][:k]).max() < 0.5
    for a, b in zip(masks[:k, 0] > 0.5, ref["masks"][:k]):
        u_, i_ = (a | b).sum(), (a & b).sum()
        if b.sum() >= 50:   # máscaras normales: IoU >= 0,95
            assert i_ / u_ >= 0.95
        else:               # máscaras diminutas: el redondeo de un borde cambia mucho el IoU; se exige <= 5 px
            assert u_ - i_ <= 5


def test_metadatos_completos():
    for nombre in MODELOS:
        meta = json.loads((RAIZ / "artifacts" / f"{nombre}.metadata.json").read_text(encoding="utf8"))
        for clave in ("semilla", "umbral_score", "preprocesamiento", "esquema_entrada", "versiones",
                      "metricas_val", "metricas_test", "costo_cpu"):
            assert clave in meta, (nombre, clave)
        assert meta["costo_cpu"]["p_TOST_delta_020"] < 0.05  # exportado equivalente al de memoria
