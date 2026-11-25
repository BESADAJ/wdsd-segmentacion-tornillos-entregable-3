"""Robustez del modelo original frente a Mask R-CNN (E3 §8; predicción 4 de book/05_modelo_diseno.md §4).

Qué mide, sobre las 108 imágenes de test, con los pesos de las corridas finales (10 semillas por modelo):
- Escala (predicción 4): la red procesa la imagen al 100 %, 75 % y 50 % de su resolución (1280x720). Se cambia
  también el tamaño interno de torchvision (min_size/max_size), porque si no la red volvería a ampliar la imagen a
  720 px y los tornillos no serían más pequeños. Las máscaras se llevan de vuelta al tamaño original con
  interpolación bilineal de la máscara blanda y umbral 0,5, y se evalúan contra la anotación original.
  Predicción: la ventaja de SGW sobre Mask R-CNN crece al bajar la escala.
- Ruido gaussiano (sigma 0,02; 0,05; 0,10) y desenfoque (sigma 1, 2, 4 px): las mismas perturbaciones y la misma
  semilla de ruido que robustez_latencia_e2.py del E2, para que las cifras sean comparables con los 13 modelos.
El umbral de score es el elegido en val con la imagen limpia (no se reajusta). F1 de instancia igual que en el E2.

Uso: python experiments/robustez_e3.py --datos D:/wdsd_datos/datos_inst [--modelos sgw_k0.45,maskrcnn]
                                       [--semillas 42-51] [--prueba]
Salida: D:/wdsd_e3/results/robustez/<modelo>_s<semilla>.csv (por imagen) y se salta lo ya hecho.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SRC_E2 = Path(os.environ.get("WDSD_E2_SRC", RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "src"))
sys.path[:0] = [str(RAIZ / "src"), str(SRC_E2)]

ESCALAS = (1.0, 0.75, 0.5)
RUIDO = (0.02, 0.05, 0.10)
DESENFOQUE = (1.0, 2.0, 4.0)
PERTURBACIONES = ([("escala", s) for s in ESCALAS] + [("ruido", s) for s in RUIDO]
                  + [("desenfoque", s) for s in DESENFOQUE])


def perturbar(rgb: np.ndarray, tipo: str, nivel: float, image_id: int) -> np.ndarray:
    """Copia literal de robustez_latencia_e2.py (E2): mismo ruido por imagen y nivel para todos los modelos."""
    import cv2
    if tipo == "ruido":
        rng = np.random.default_rng([int(image_id), int(round(nivel * 1000))])
        x = rgb.astype(np.float32) / 255.0 + rng.normal(0.0, nivel, rgb.shape).astype(np.float32)
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    if tipo == "desenfoque":
        return cv2.GaussianBlur(rgb, (0, 0), sigmaX=nivel)
    return rgb


def cargar_modelo(carpeta: Path, disp):
    import torch
    from wdsd_inst import maskrcnn
    m = maskrcnn.crear_modelo(True)  # SGW solo cambia la asignación de entrenamiento: en inferencia es la misma red
    m.load_state_dict(torch.load(carpeta / "mejor.pt", map_location="cpu"))
    return m.to(disp).eval()


def predecir(m, rgb: np.ndarray, escala: float, disp):
    import torch
    import torch.nn.functional as Fn
    h, w = rgb.shape[:2]
    m.transform.min_size, m.transform.max_size = (int(round(720 * escala)),), int(round(1280 * escala))
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(disp)
    if escala != 1.0:
        img = Fn.interpolate(img[None], scale_factor=escala, mode="bilinear", antialias=True,
                             align_corners=False)[0]
    with torch.no_grad(), torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
        out = m([img])[0]
    blandas = out["masks"].float()
    if escala != 1.0 and len(blandas):
        blandas = Fn.interpolate(blandas, size=(h, w), mode="bilinear", align_corners=False)
    return out["scores"].float().cpu().numpy(), (blandas[:, 0] > 0.5).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--modelos", default="sgw_k0.45,maskrcnn")
    ap.add_argument("--semillas", default="42-51")
    ap.add_argument("--prueba", action="store_true", help="5 imágenes, sin guardar")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from wdsd_e2.comun import filtrar
    from wdsd_inst.metricas import a_rle, cargar_gt, pq_por_imagen

    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i, j = (int(x) for x in a.semillas.split("-")) if "-" in a.semillas else (int(a.semillas),) * 2
    gt = cargar_gt(a.datos / "coco" / "test.json")
    imgs = sorted(gt.imgs.values(), key=lambda x: x["id"])[: 5 if a.prueba else None]
    rgbs = {im["id"]: np.asarray(Image.open(im["ruta"]).convert("RGB")) for im in imgs}
    out = a.e3 / "robustez"
    out.mkdir(parents=True, exist_ok=True)
    for s in range(i, j + 1):  # por semilla: si se corta, quedan pares completos
        for modelo in a.modelos.split(","):
            destino = out / f"{modelo}_s{s}.csv"
            if destino.exists() and not a.prueba:
                continue
            carpeta = a.e3 / "corridas" / f"{modelo}_s{s}"
            umbral = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))["metricas"]["umbral_score"]
            m = cargar_modelo(carpeta, disp)
            filas = []
            for tipo, nivel in PERTURBACIONES:
                preds = []
                for im in imgs:
                    sc, ms = predecir(m, perturbar(rgbs[im["id"]], tipo, nivel, im["id"]),
                                      nivel if tipo == "escala" else 1.0, disp)
                    preds += [{"image_id": im["id"], "category_id": 1, "segmentation": a_rle(mk), "score": float(x)}
                              for x, mk in zip(sc, ms) if x >= 0.05 and mk.any()]
                t = pq_por_imagen(gt, filtrar(preds), umbral)
                t = t[t.image_id.isin(rgbs)]
                t.insert(0, "nivel", nivel); t.insert(0, "perturbacion", tipo)
                t.insert(0, "semilla", s); t.insert(0, "modelo", modelo)
                filas.append(t)
                print(f"{modelo} s{s} {tipo:10s} {nivel:<5}: F1 grupo {t.groupby('grupo').f1.mean().mean():.3f}",
                      flush=True)
            df = pd.concat(filas, ignore_index=True)
            limpio = df[(df.perturbacion == "escala") & (df.nivel == 1.0)].groupby("grupo").f1.mean().mean()
            ref = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))["metricas"]["test"]["F1_grupo"]
            print(f"  control escala 1.0: {limpio:.4f} vs resumen.json {ref:.4f}", flush=True)
            if not a.prueba:
                df.to_csv(destino, index=False)
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
