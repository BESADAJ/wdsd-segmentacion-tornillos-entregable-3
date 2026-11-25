"""Análisis de los 3 conjuntos adicionales (experiments/protocolo.md §6, tag e3-conjuntos).

Unidad estadística en test (declarada antes de entrenar): la imagen (BBBC039, NuInsSeg, MinneApple). F1 de instancia por
imagen promediado sobre las semillas de cada modelo. Por conjunto: Wilcoxon pareado por imagen, bootstrap pareado
por imagen y t pareado sobre semillas (F1 medio por corrida), bilaterales; Holm sobre los 3 conjuntos; decide el
máximo de los p ajustados. Consistencia (E3 §8): SGW con mayor F1 medio en >= 2 de 3 y no significativamente peor en
ninguno. Secundario: recall del tercil pequeño (cortes por percentiles 33/67 de las áreas de val de cada conjunto).

Uso: python experiments/analisis_extra.py [--raiz D:/wdsd_datos/extra] [--salida D:/wdsd_e3/results_extra]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import stats
from scipy.optimize import linear_sum_assignment

RAIZ = Path(__file__).resolve().parents[1]
SGW, CTRL = "sgw_k0.45", "maskrcnn"
CONJUNTOS = ("bbbc039", "nuinsseg", "minneapple")
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def recall_pequeno(carpeta: Path, test: dict, cortes, umbral: float):
    imgs = {i["id"]: i for i in test["images"]}
    gts = defaultdict(list)
    for a in test["annotations"]:
        i = imgs[a["image_id"]]
        gts[a["image_id"]].append(mu.merge(mu.frPyObjects(a["segmentation"], i["height"], i["width"])))
    preds = defaultdict(list)
    for p in json.loads((carpeta / "preds_test.json").read_text()):
        if p["score"] >= umbral:
            preds[int(p["image_id"])].append(p)
    tp = fn = 0
    for iid, g in gts.items():
        areas = [float(mu.area(r)) for r in g]
        d = [p["segmentation"] for p in sorted(preds[iid], key=lambda p: -p["score"])]
        hit = set()
        if d:
            iou = mu.iou(d, g, [0] * len(g))
            for a, b in zip(*linear_sum_assignment(-iou)):
                if iou[a, b] >= 0.5:
                    hit.add(b)
        for j, ar in enumerate(areas):
            if ar < cortes[0]:
                tp, fn = tp + (j in hit), fn + (j not in hit)
    return tp / (tp + fn) if tp + fn else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", type=Path, default=Path("D:/wdsd_datos/extra"))
    ap.add_argument("--salida", type=Path, default=Path("D:/wdsd_e3/results_extra"))
    a = ap.parse_args()
    out = RAIZ / "results" / "conjuntos_extra"
    out.mkdir(parents=True, exist_ok=True)
    filas, por_corrida = [], []
    for c in CONJUNTOS:
        test = json.loads((a.raiz / c / "coco" / "test.json").read_text(encoding="utf8"))
        val = json.loads((a.raiz / c / "coco" / "val.json").read_text(encoding="utf8"))
        cortes = np.percentile([x["area"] for x in val["annotations"]], [100 / 3, 200 / 3])
        img = {}
        for m in (SGW, CTRL):
            tabs = []
            for r in sorted((a.salida / c / "corridas").glob(f"{m}_s*/resumen.json")):
                d = json.loads(r.read_text(encoding="utf8"))
                t = pd.read_csv(r.parent / "test_por_imagen.csv")
                t["semilla"] = d["semilla"]
                tabs.append(t)
                por_corrida.append(dict(conjunto=c, modelo=m, semilla=d["semilla"],
                                        F1_img_test=t.f1.mean(), F1_grupo_test=d["metricas"]["test"]["F1_grupo"],
                                        F1_grupo_val=d["metricas"]["val"]["F1_grupo"],
                                        AP_test=d["metricas"]["test"]["AP"],
                                        recall_pequeno=recall_pequeno(r.parent, test, cortes,
                                                                      d["metricas"]["umbral_score"])))
            if tabs:
                img[m] = pd.concat(tabs)
        if len(img) < 2:
            print(f"{c}: faltan corridas"); continue
        A = img[SGW].groupby("image_id").f1.mean()
        B = img[CTRL].groupby("image_id").f1.mean()
        dimg = (A - B).dropna().to_numpy()
        R = pd.DataFrame(por_corrida)
        R = R[R.conjunto == c].pivot_table(index="semilla", columns="modelo", values="F1_img_test")
        pares = R.dropna()
        bs = dimg[rng.integers(0, len(dimg), (10_000, len(dimg)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pb = min(1.0, 2 * (min((bs <= 0).sum(), (bs >= 0).sum()) + 1) / 10_001)
        pt = float(stats.ttest_rel(pares[SGW], pares[CTRL]).pvalue) if len(pares) >= 2 else np.nan
        filas.append(dict(conjunto=c, n_imagenes=len(dimg), semillas=len(pares), F1_SGW=A.mean(), F1_MaskRCNN=B.mean(),
                          dif=dimg.mean(), ic95_lo=lo, ic95_hi=hi, imagenes_SGW_mejor=int((dimg > 0).sum()),
                          p_wilcoxon=float(stats.wilcoxon(dimg).pvalue) if np.any(dimg) else 1.0,
                          p_bootstrap=pb, p_t_semillas=pt))
        if c == "nuinsseg":  # descriptivo: F1 por órgano
            gi = {i["id"]: i["grupo_nombre"] for i in test["images"]}
            org = pd.DataFrame({"SGW": A, "MaskRCNN": B})
            org["organo"] = [gi[i] for i in org.index]
            org.groupby("organo").mean().round(4).to_csv(out / "nuinsseg_por_organo.csv")
    F = pd.DataFrame(filas)
    if F.empty:
        return
    for k in ("p_wilcoxon", "p_bootstrap", "p_t_semillas"):
        F[k.replace("p_", "p_holm_")] = holm(F[k].fillna(1.0))
    F["p_decision"] = F[["p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t_semillas"]].max(axis=1)
    F["SGW_significativamente_peor"] = (F.p_decision < 0.05) & (F.dif < 0)
    F["SGW_significativamente_mejor"] = (F.p_decision < 0.05) & (F.dif > 0)
    F.to_csv(out / "contrastes.csv", index=False)
    pd.DataFrame(por_corrida).to_csv(out / "por_corrida.csv", index=False)
    consistente = bool((F.dif > 0).sum() >= 2 and not F.SGW_significativamente_peor.any() and len(F) == 3)
    (out / "consistencia.json").write_text(json.dumps(
        {"conjuntos_con_SGW_mejor_en_media": int((F.dif > 0).sum()), "conjuntos_evaluados": len(F),
         "alguno_significativamente_peor": bool(F.SGW_significativamente_peor.any()),
         "criterio_consistencia_cumplido": consistente}, indent=1), encoding="utf8")
    pd.set_option("display.width", 220)
    print(F.round(4).to_string(index=False))
    print(pd.DataFrame(por_corrida).groupby(["conjunto", "modelo"])[["F1_img_test", "AP_test", "recall_pequeno"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("consistencia:", consistente)


if __name__ == "__main__":
    main()
