"""Recall y F1 por tercil de área en test (hipótesis H-M), con la misma lógica que la RQ3 del E2
(analisis_extra_e2.py de Ivan): cortes por percentiles 33/67 de las áreas de VAL, predicciones sobre el umbral
elegido en val, emparejamiento húngaro por IoU de máscara >= 0.5. Cada TP/FN va al tercil de su tornillo real.

Lee todas las corridas finales del E3 (<salida>/corridas/*/resumen.json) y, opcionalmente, las de Mask R-CNN del E2.

Uso: python experiments/terciles_e3.py <datos_inst> [--salida D:/wdsd_e3/results] [--e2 <results/e2/corridas>]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

TERCILES = ["pequeño", "mediano", "grande"]


def gt_rles(coco):
    imgs = {i["id"]: i for i in coco["images"]}
    por_img = defaultdict(list)
    for a in coco["annotations"]:
        i = imgs[a["image_id"]]
        r = mu.frPyObjects(a["segmentation"], i["height"], i["width"])
        por_img[a["image_id"]].append(mu.merge(r) if isinstance(r, list) else r)
    return imgs, por_img


def conteos(carpeta: Path, umbral: float, imgs, gts, area_gt, cortes):
    preds = json.loads((carpeta / "preds_test.json").read_text())
    por_img = defaultdict(list)
    for p in preds:
        if p["score"] >= umbral:
            por_img[int(p["image_id"])].append(p)
    c = defaultdict(lambda: np.zeros(3))  # (grupo, tercil) -> tp, fp, fn
    for iid, info in imgs.items():
        g, ag = gts.get(iid, []), area_gt.get(iid, [])
        dr = [p["segmentation"] for p in sorted(por_img[iid], key=lambda p: -p["score"])]
        eg, ed = set(), set()
        if g and dr:
            iou = mu.iou(dr, g, [0] * len(g))
            for a, b in zip(*linear_sum_assignment(-iou)):
                if iou[a, b] >= 0.5:
                    ed.add(a); eg.add(b)
        for j, a in enumerate(ag):
            c[(info["grupo"], int(np.searchsorted(cortes, a, side="right")))][0 if j in eg else 2] += 1
        for j, r in enumerate(dr):
            if j not in ed:
                c[(info["grupo"], int(np.searchsorted(cortes, float(mu.area(r)), side="right")))][1] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--e2", type=Path, default=Path(__file__).resolve().parents[3]
                    / "Entregable_2_Deep_Learning" / "results" / "e2" / "corridas")
    a = ap.parse_args()
    test = json.loads((a.datos / "coco" / "test.json").read_text(encoding="utf8"))
    val = json.loads((a.datos / "coco" / "val.json").read_text(encoding="utf8"))
    cortes = np.percentile([x["area"] for x in val["annotations"]], [100 / 3, 200 / 3])
    imgs, gts = gt_rles(test)
    area_gt = {i: [float(mu.area(r)) for r in rs] for i, rs in gts.items()}

    corridas = []
    for r in sorted((a.salida / "corridas").glob("*/resumen.json")):
        d = json.loads(r.read_text(encoding="utf8"))
        corridas.append((d["modelo"], d["semilla"], d.get("maquina"), r.parent, d["metricas"]["umbral_score"]))
    if a.e2.exists():
        for r in sorted(a.e2.glob("maskrcnn_s*/resumen.json")):
            d = json.loads(r.read_text(encoding="utf8"))
            corridas.append(("maskrcnn_E2", d["semilla"], d.get("maquina"), r.parent, d["metricas"]["umbral_score"]))

    filas = []
    for modelo, s, maq, carpeta, u in corridas:
        for (grp, k), (tp, fp, fn) in conteos(carpeta, u, imgs, gts, area_gt, cortes).items():
            filas.append((modelo, s, maq, grp, TERCILES[k], tp, fp, fn))
    T = pd.DataFrame(filas, columns=["modelo", "semilla", "maquina", "grupo", "tercil", "tp", "fp", "fn"])
    out = a.salida / "analisis"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "terciles_conteos.csv", index=False)
    R = T.groupby(["modelo", "semilla", "tercil"])[["tp", "fn"]].sum()
    R = (R.tp / (R.tp + R.fn)).unstack()[TERCILES].round(3)
    R.to_csv(out / "terciles_recall_por_corrida.csv")
    print(f"cortes de área (val): {cortes.round(1).tolist()}")
    print(R.to_string())


if __name__ == "__main__":
    main()
