"""Piloto de asignación (predicción 1 de H-M, book/05_modelo_diseno.md §4). No entrena ni toca test.

Para cada tornillo de la partición de desarrollo (train de la partición agrupada) cuenta:
  - anclas positivas en RPN (umbral 0.7, con baja calidad, igual que torchvision), con IoU y con SGW;
  - fracción de propuestas sintéticas positivas en la cabeza RoI (umbral 0.5): 200 copias de la caja con un
    error gaussiano de sigma px en cada esquina, que imita el error de localización de RPN.
Resume por tercil de área (cortes del E2: 75 y 156.67 px, calculados en val).

Uso: python experiments/piloto_asignacion.py <datos_inst/coco/train.json> [carpeta_salida] [--sigma 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original_model import PISO_C_PX, REJILLA_KAPPA, SimilitudSGW  # noqa: E402

CORTES = (75.0, 156.66666666666669)  # results/e2/analisis_extra/rq3_cortes.json del E2


def anclas(alto: int, ancho: int) -> torch.Tensor:
    """Las mismas anclas que el Mask R-CNN del E1/E2: tamaños 8-128 px, razones 0.5/1/2, sin reescalar."""
    gen = AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 5)
    alto_p, ancho_p = int(np.ceil(alto / 32) * 32), int(np.ceil(ancho / 32) * 32)  # relleno de GeneralizedRCNNTransform
    img = ImageList(torch.zeros(1, 3, alto_p, ancho_p), [(alto, ancho)])
    mapas = [torch.zeros(1, 1, alto_p // s, ancho_p // s) for s in (4, 8, 16, 32, 64)]
    return gen(img, mapas)[0]


def tercil(area: float) -> str:
    return "pequeño" if area < CORTES[0] else ("mediano" if area < CORTES[1] else "grande")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_train")
    ap.add_argument("salida", nargs="?", default=str(Path(__file__).resolve().parents[1] / "results" / "piloto"))
    ap.add_argument("--sigma", type=float, default=2.0)
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    d = json.loads(Path(a.coco_train).read_text(encoding="utf8"))
    por_img = {i["id"]: [] for i in d["images"]}
    for an in d["annotations"]:
        por_img[an["image_id"]].append(an)
    tam = {i["id"]: (i["height"], i["width"]) for i in d["images"]}
    cache_anclas = {}
    medidas = {"IoU": box_iou, **{f"SGW k={k}": SimilitudSGW(k, PISO_C_PX) for k in REJILLA_KAPPA}}
    matcher_rpn = det_utils.Matcher(0.7, 0.3, allow_low_quality_matches=True)
    g = torch.Generator().manual_seed(0)
    filas = []
    for img_id, anns in por_img.items():
        if not anns:
            continue
        hw = tam[img_id]
        if hw not in cache_anclas:
            cache_anclas[hw] = anclas(*hw)
        A = cache_anclas[hw]
        gt = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (an["bbox"] for an in anns)], dtype=torch.float32)
        ruido = torch.randn(len(gt), 200, 4, generator=g) * a.sigma
        props = (gt[:, None, :] + ruido)
        for nombre, f in medidas.items():
            idx = matcher_rpn(f(gt, A))
            n_pos = torch.bincount(idx[idx >= 0], minlength=len(gt))
            frac_roi = [float((f(gt[i:i + 1], props[i])[0] >= 0.5).float().mean()) for i in range(len(gt))]
            for i, an in enumerate(anns):
                filas.append({"image_id": img_id, "ann_id": an["id"], "area": an["area"], "tercil": tercil(an["area"]),
                              "medida": nombre, "anclas_pos_rpn": int(n_pos[i]), "frac_props_pos_roi": frac_roi[i]})
    t = pd.DataFrame(filas)
    t.to_csv(salida / "piloto_asignacion.csv", index=False)
    res = (t.groupby(["medida", "tercil"])
             .agg(n=("ann_id", "size"), anclas_pos_media=("anclas_pos_rpn", "mean"),
                  anclas_pos_mediana=("anclas_pos_rpn", "median"),
                  frac_solo_1_ancla=("anclas_pos_rpn", lambda x: float((x <= 1).mean())),
                  frac_props_pos_roi=("frac_props_pos_roi", "mean"))
             .reset_index())
    res.to_csv(salida / "piloto_asignacion_resumen.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
