"""Guarda las salidas de referencia de PyTorch (modelo en memoria, CPU, FP32) para la prueba de ida y vuelta en CI.

La entrada es una imagen fija que sí está en el repositorio (data/galeria/<id>.jpg, ampliada a 1280x720), porque CI
no tiene ni el dataset ni los pesos .pt. tests/test_exportacion.py corre el ONNX sobre la misma entrada y compara.

Uso: python training/referencia_ida_vuelta.py [--modelos sgw_k0.45,maskrcnn]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SRC_E2 = Path(os.environ.get("WDSD_E2_SRC", RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "src"))
sys.path[:0] = [str(RAIZ / "src"), str(SRC_E2)]


def entrada_fija() -> np.ndarray:
    from PIL import Image
    g = json.loads((RAIZ / "data" / "galeria.json").read_text(encoding="utf8"))[0]
    im = Image.open(RAIZ / "data" / g["archivo"]).convert("RGB").resize((1280, 720), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--modelos", default="sgw_k0.45,maskrcnn")
    a = ap.parse_args()
    import torch
    import original_model as om
    x = entrada_fija()
    for nombre in a.modelos.split(","):
        meta = json.loads((RAIZ / "artifacts" / f"{nombre}.metadata.json").read_text(encoding="utf8"))
        m = om.mask_rcnn_base(True)
        m.load_state_dict(torch.load(a.e3 / "corridas" / f"{nombre}_s{meta['semilla']}" / "mejor.pt",
                                     map_location="cpu"))
        m.eval()
        m.roi_heads.score_thresh = meta["umbral_score"]  # igual que el ONNX exportado
        with torch.no_grad():
            o = m([torch.from_numpy(x)])[0]
        np.savez_compressed(RAIZ / "results" / "exportacion" / f"referencia_{nombre}.npz",
                            boxes=o["boxes"].numpy(), scores=o["scores"].numpy(),
                            masks=(o["masks"][:, 0] > 0.5).numpy())
        print(nombre, "detecciones:", len(o["scores"]), "sobre el umbral:", int((o["scores"] >= meta["umbral_score"]).sum()))


if __name__ == "__main__":
    main()
