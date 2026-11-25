"""Genera los datos livianos del dashboard y del informe (data/), desde las corridas del E2 y del E3.

Todo lo que muestra el dashboard sale de aquí o de results/: así las cifras del dashboard y del libro vienen de la
misma tabla maestra (data/tabla_maestra.csv), y tests/test_consistencia.py lo verifica.

Escribe:
- tabla_maestra.csv     una fila por modelo: familia, F1 test media ± DE e IC 95 %, F1 val, semillas, parámetros,
                        GFLOPs, latencia GPU y CPU, ECE, rango de Friedman y decisión frente a SGW.
- eda_anotaciones.csv   cada tornillo del WDSD: partición, grupo, área, ancho, alto, centro.
- eda_imagenes.csv      cada imagen: partición, grupo, número de tornillos, brillo y contraste.
- galeria/*.jpg + galeria.json   8 imágenes de test reducidas con contornos reales y predichos (SGW y Mask R-CNN).
- curvas.csv            pérdida y AP de val por época de SGW y Mask R-CNN (10 semillas).
- terciles.csv          recall por tercil de área de cada corrida.
- ood_referencia.json   media y covarianza de rasgos simples de las imágenes de train, con su umbral (p99).
- conforme.json         intervalo conforme del número de tornillos por imagen (split conformal en val, 90 %).
- robustez.csv          F1 por grupo bajo perturbaciones (E2: 13 modelos; E3: SGW y control).

Uso: python experiments/datos_dashboard.py --datos D:/wdsd_datos/datos_inst
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
E2 = RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "results" / "e2"
SGW, CTRL = "sgw_k0.45", "maskrcnn"
NOMBRES = {"sgw_k0.45": "SGW (original)", "maskrcnn": "Mask R-CNN", "cascade": "Cascade Mask R-CNN", "htc": "HTC",
           "msrcnn": "MS R-CNN", "pointrend": "PointRend", "mask2former": "Mask2Former", "condinst": "CondInst",
           "solov2": "SOLOv2", "yolov7": "YOLOv7-seg", "yolov8": "YOLOv8-seg", "yolov9": "YOLOv9-seg",
           "yolo11": "YOLO11-seg", "yolo26": "YOLO26-seg", "clasico": "Clásico (umbral + morfología)"}
FAMILIA = {"sgw_k0.45": "Dos etapas (asignación SGW)", "maskrcnn": "Dos etapas", "cascade": "Dos etapas",
           "htc": "Dos etapas", "msrcnn": "Dos etapas", "pointrend": "Dos etapas", "mask2former": "Transformer",
           "condinst": "Una etapa (kernels)", "solov2": "Una etapa (kernels)", "yolov7": "Una etapa (YOLO)",
           "yolov8": "Una etapa (YOLO)", "yolov9": "Una etapa (YOLO)", "yolo11": "Una etapa (YOLO)",
           "yolo26": "Una etapa (YOLO)", "clasico": "Clásico"}
ROL = {"sgw_k0.45": "Modelo original", "maskrcnn": "Baseline fuerte", "yolov8": "Baseline fuerte",
       "solov2": "Baseline fuerte", "clasico": "Baseline clásico"}


def tabla_maestra(e3: Path):
    from scipy import stats
    conf = RAIZ / "results" / "confirmatorio"
    filas = []
    c2 = pd.read_csv(E2.parent.parent / "results" / "e2" / "corridas.csv")
    c2 = c2[c2.semilla.between(42, 46)]
    for m in NOMBRES:
        if m in (SGW, CTRL):
            rs = [json.loads(p.read_text(encoding="utf8")) for p in sorted((e3 / "corridas").glob(f"{m}_s*/resumen.json"))]
            test = np.array([r["metricas"]["test"]["F1_grupo"] for r in rs])
            val = np.array([r["metricas"]["val"]["F1_grupo"] for r in rs])
            ap = np.array([r["metricas"]["test"]["AP"] for r in rs])
            maq = "RTX 3050"
        else:
            d = c2[c2.modelo == m]
            test, val, ap = d.test_F1_grupo.to_numpy(), d.val_F1_grupo.to_numpy(), d.test_AP.to_numpy()
            maq = "RTX 3060"
        n = len(test)
        h = stats.t.ppf(0.975, n - 1) * test.std(ddof=1) / np.sqrt(n)
        filas.append(dict(modelo=m, nombre=NOMBRES[m], rol=ROL.get(m, "Top 10"), familia=FAMILIA[m],
                          F1_test=test.mean(), F1_test_DE=test.std(ddof=1), F1_test_ic_lo=test.mean() - h,
                          F1_test_ic_hi=test.mean() + h, F1_val=val.mean(), AP_mascara_test=ap.mean(),
                          semillas=n, maquina=maq))
    T = pd.DataFrame(filas).set_index("modelo")
    lat = pd.read_csv(E2 / "analisis_extra" / "latencia_tabla.csv").set_index("modelo")
    for m in T.index:
        src = "maskrcnn" if m == SGW else m  # misma red en inferencia
        if src in lat.index:
            T.loc[m, "parametros"] = lat.loc[src, "parametros"]
            T.loc[m, "GFLOPs"] = lat.loc[src, "gflops_una_imagen"]
            T.loc[m, "latencia_GPU_ms"] = lat.loc[src, "mediana_ms"]
    cal = pd.read_csv(E2 / "analisis_extra2" / "calibracion_completa.csv").set_index("modelo")
    for m in T.index:
        if m in cal.index:
            T.loc[m, "ECE"] = cal.loc[m, "ECE"]
    cpu = RAIZ / "results" / "exportacion" / "costo_cpu.csv"
    if cpu.exists():
        cc = pd.read_csv(cpu).set_index("modelo")
        for m in cc.index:
            T.loc[m, "latencia_CPU_ms"] = cc.loc[m, "lat_onnx_fp16_mediana_ms"]
            T.loc[m, "latencia_CPU_p95_ms"] = cc.loc[m, "lat_onnx_fp16_p95_ms"]
            T.loc[m, "MB_onnx_fp16"] = cc.loc[m, "mb_onnx_fp16"]
    rangos = pd.read_csv(conf / "friedman_rangos.csv", index_col=0).iloc[:, 0]
    T["rango_Friedman"] = rangos.reindex(T.index)
    fam = pd.read_csv(conf / "familia_R3.csv").set_index("rival")
    T["dif_SGW_menos_modelo"] = fam.dif_media.reindex(T.index)
    T["p_holm_decision"] = fam.p_decision.reindex(T.index)
    T["decision_vs_SGW"] = fam.decision.reindex(T.index)
    T = T.sort_values("F1_test", ascending=False)
    T.round(6).to_csv(RAIZ / "data" / "tabla_maestra.csv")
    print(T[["nombre", "F1_test", "F1_test_DE", "semillas", "rango_Friedman"]].round(3).to_string())


def eda(datos: Path):
    ann, img = [], []
    for split in ("train", "val", "test"):
        d = json.loads((datos / "coco" / f"{split}.json").read_text(encoding="utf8"))
        n = defaultdict(int)
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            n[a["image_id"]] += 1
            ann.append(dict(particion=split, image_id=a["image_id"], area_px=a["area"], ancho_px=w, alto_px=h,
                            cx=x + w / 2, cy=y + h / 2))
        for i in d["images"]:
            g = np.asarray(Image.open(i["ruta"]).convert("L").reduce(4), dtype=np.float32)
            img.append(dict(particion=split, image_id=i["id"], grupo=i["grupo"], n_tornillos=n[i["id"]],
                            brillo=g.mean(), contraste=g.std(), ancho=i["width"], alto=i["height"]))
    A, I = pd.DataFrame(ann), pd.DataFrame(img)
    A = A.merge(I[["particion", "image_id", "grupo"]], on=["particion", "image_id"])
    A.round(2).to_csv(RAIZ / "data" / "eda_anotaciones.csv", index=False)
    I.round(3).to_csv(RAIZ / "data" / "eda_imagenes.csv", index=False)
    print(f"EDA: {len(A)} tornillos, {len(I)} imágenes")


def rasgos(rgb: np.ndarray) -> np.ndarray:
    """Rasgos simples para el aviso fuera de dominio (deben coincidir con src/inference.py)."""
    x = rgb.astype(np.float32) / 255.0
    g = x.mean(-1)
    gy, gx = np.gradient(g)
    return np.array([*x.reshape(-1, 3).mean(0), *x.reshape(-1, 3).std(0), np.hypot(gx, gy).mean(),
                     rgb.shape[1] / rgb.shape[0]])


def ood(datos: Path):
    d = json.loads((datos / "coco" / "train.json").read_text(encoding="utf8"))
    X = np.stack([rasgos(np.asarray(Image.open(i["ruta"]).convert("RGB").reduce(4))) for i in d["images"]])
    mu, cov = X.mean(0), np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    inv = np.linalg.inv(cov)
    dm = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, inv, X - mu))
    ref = dict(rasgos=["media_R", "media_G", "media_B", "de_R", "de_G", "de_B", "gradiente_medio", "aspecto"],
               media=mu.tolist(), cov_inv=inv.tolist(), umbral_p99=float(np.percentile(dm, 99)),
               umbral_max=float(dm.max()), n_train=len(X), reduccion=4)
    (RAIZ / "data" / "ood_referencia.json").write_text(json.dumps(ref, indent=1), encoding="utf8")
    print(f"OOD: umbral p99 = {ref['umbral_p99']:.2f}")


def mejor_semilla(e3: Path, m: str):
    rs = [json.loads(p.read_text(encoding="utf8")) for p in (e3 / "corridas").glob(f"{m}_s*/resumen.json")]
    r = max(rs, key=lambda r: r["metricas"]["val"]["F1_grupo"])
    return r["semilla"], r["metricas"]["umbral_score"]


def conforme(datos: Path, e3: Path):
    """Split conformal del conteo: |n_pred - n_real| en val -> cuantil (1 - a)(1 + 1/n)."""
    from pycocotools import mask as mu  # noqa: F401
    s, u = mejor_semilla(e3, SGW)
    val = json.loads((datos / "coco" / "val.json").read_text(encoding="utf8"))
    n_gt = defaultdict(int)
    for a in val["annotations"]:
        n_gt[a["image_id"]] += 1
    pv = json.loads((e3 / "corridas" / f"{SGW}_s{s}" / "preds_val.json").read_text())
    n_pr = defaultdict(int)
    for p in pv:
        if p["score"] >= u:
            n_pr[p["image_id"]] += 1
    r = np.array([abs(n_pr[i["id"]] - n_gt[i["id"]]) for i in val["images"]])
    out = {}
    for nivel in (0.8, 0.9, 0.95):
        k = int(np.ceil((len(r) + 1) * nivel))
        out[str(nivel)] = float(np.sort(r)[min(k, len(r)) - 1])
    test = json.loads((datos / "coco" / "test.json").read_text(encoding="utf8"))
    tg = defaultdict(int)
    for a in test["annotations"]:
        tg[a["image_id"]] += 1
    pt = json.loads((e3 / "corridas" / f"{SGW}_s{s}" / "preds_test.json").read_text())
    tp = defaultdict(int)
    for p in pt:
        if p["score"] >= u:
            tp[p["image_id"]] += 1
    cob = {k: float(np.mean([abs(tp[i["id"]] - tg[i["id"]]) <= q for i in test["images"]])) for k, q in out.items()}
    (RAIZ / "data" / "conforme.json").write_text(json.dumps(
        {"modelo": SGW, "semilla": s, "metodo": "split conformal sobre |n_pred - n_real| en val (93 imágenes)",
         "radio": out, "cobertura_test": cob}, indent=1), encoding="utf8")
    print("conforme:", out, "cobertura test:", cob)


def galeria(datos: Path, e3: Path, n=8, escala=0.5):
    import cv2
    from pycocotools import mask as mu
    test = json.loads((datos / "coco" / "test.json").read_text(encoding="utf8"))
    por_grupo = defaultdict(list)
    for i in sorted(test["images"], key=lambda i: i["id"]):
        por_grupo[i["grupo"]].append(i)
    elegidas = [v[len(v) // 2] for g, v in sorted(por_grupo.items())][:n]
    gts = defaultdict(list)
    for a in test["annotations"]:
        gts[a["image_id"]].append(a)
    preds = {}
    for m in (SGW, CTRL):
        s, u = mejor_semilla(e3, m)
        pp = defaultdict(list)
        for p in json.loads((e3 / "corridas" / f"{m}_s{s}" / "preds_test.json").read_text()):
            if p["score"] >= u:
                pp[p["image_id"]].append(p)
        f1 = pd.read_csv(e3 / "corridas" / f"{m}_s{s}" / "test_por_imagen.csv").set_index("image_id")
        preds[m] = (pp, f1, s)

    def contornos(m):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(c[:, 0, :] * escala).round(1).tolist() for c in cs if len(c) >= 3]

    salida = []
    carpeta = RAIZ / "data" / "galeria"
    carpeta.mkdir(parents=True, exist_ok=True)
    for i in elegidas:
        h, w = i["height"], i["width"]
        im = Image.open(i["ruta"]).convert("RGB").resize((int(w * escala), int(h * escala)))
        im.save(carpeta / f"{i['id']}.jpg", quality=85)
        reg = dict(image_id=i["id"], archivo=f"galeria/{i['id']}.jpg", grupo=i["grupo"], ancho=im.width,
                   alto=im.height, real=[], predicciones={})
        for a in gts[i["id"]]:
            reg["real"] += contornos(mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], h, w))))
        for m, (pp, f1, s) in preds.items():
            dets = []
            for p in pp[i["id"]]:
                dets.append(dict(score=round(p["score"], 3), contornos=contornos(mu.decode(p["segmentation"]))))
            fila = f1.loc[i["id"]]
            reg["predicciones"][m] = dict(semilla=int(s), detecciones=dets, f1=float(fila.f1), tp=int(fila.tp),
                                          fp=int(fila.fp), fn=int(fila.fn))
        salida.append(reg)
    (RAIZ / "data" / "galeria.json").write_text(json.dumps(salida), encoding="utf8")
    print(f"galería: {len(salida)} imágenes")


def curvas_y_terciles(e3: Path):
    filas = []
    for m in (SGW, CTRL):
        for p in sorted((e3 / "corridas").glob(f"{m}_s*/historial.csv")):
            d = pd.read_csv(p)
            d.insert(0, "semilla", int(p.parent.name.split("_s")[-1]))
            d.insert(0, "modelo", m)
            filas.append(d)
    pd.concat(filas).round(5).to_csv(RAIZ / "data" / "curvas.csv", index=False)
    t = e3 / "analisis" / "terciles_recall_por_corrida.csv"
    pd.read_csv(t).to_csv(RAIZ / "data" / "terciles.csv", index=False)


def por_imagen(e3: Path):
    """F1, Dice semántico, TP, FP y FN por imagen de test, promediados sobre las semillas de cada modelo."""
    filas = []
    for m in NOMBRES:
        raiz, sem = (e3 / "corridas", range(42, 52)) if m in (SGW, CTRL) else (E2 / "corridas", range(42, 47))
        for s in sem:
            d = pd.read_csv(raiz / f"{m}_s{s}" / "test_por_imagen.csv")
            d.insert(0, "semilla", s); d.insert(0, "modelo", m)
            filas.append(d[["modelo", "semilla", "image_id", "grupo", "n_gt", "tp", "fp", "fn", "f1",
                            "dice_semantico"]])
    P = pd.concat(filas).groupby(["modelo", "image_id", "grupo", "n_gt"])[["tp", "fp", "fn", "f1",
                                                                           "dice_semantico"]].mean()
    P.round(4).reset_index().to_csv(RAIZ / "data" / "por_imagen.csv", index=False)


def robustez(e3: Path):
    r2 = pd.read_csv(E2 / "analisis_extra" / "robustez_tabla.csv")[["modelo", "perturbacion", "nivel", "F1_grupo"]]
    r2["fuente"] = "E2 (3060)"
    filas = []
    for p in sorted((e3 / "robustez").glob("*.csv")):
        d = pd.read_csv(p)
        g = d.groupby(["modelo", "semilla", "perturbacion", "nivel", "grupo"]).f1.mean()
        filas.append(g.groupby(["modelo", "semilla", "perturbacion", "nivel"]).mean())
    if filas:
        r3 = pd.concat(filas).groupby(["modelo", "perturbacion", "nivel"]).agg(["mean", "std", "count"]).reset_index()
        r3.columns = ["modelo", "perturbacion", "nivel", "F1_grupo", "DE_semillas", "semillas"]
        r3["fuente"] = "E3 (3050)"
        r2 = pd.concat([r2, r3], ignore_index=True)
    r2.round(5).to_csv(RAIZ / "data" / "robustez.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--solo", default="tabla,eda,ood,conforme,galeria,curvas,por_imagen,robustez")
    a = ap.parse_args()
    (RAIZ / "data").mkdir(exist_ok=True)
    pasos = {"tabla": lambda: tabla_maestra(a.e3), "eda": lambda: eda(a.datos), "ood": lambda: ood(a.datos),
             "conforme": lambda: conforme(a.datos, a.e3), "galeria": lambda: galeria(a.datos, a.e3),
             "curvas": lambda: curvas_y_terciles(a.e3), "por_imagen": lambda: por_imagen(a.e3),
             "robustez": lambda: robustez(a.e3)}
    for p in a.solo.split(","):
        pasos[p]()


if __name__ == "__main__":
    main()
