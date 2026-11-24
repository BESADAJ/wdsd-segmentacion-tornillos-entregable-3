"""Entrenador del modelo original (SGW) y de sus variantes, con el mismo flujo que el E2.

Reutiliza sin cambios el trabajador de Mask R-CNN del E2 (wdsd_e2.m_torchvision), su evaluación (wdsd_e2.evaluar)
y sus constantes (lote 2, coseno hasta el 1 %, tasa = 0.0025 * 2 * factor, factor 2 elegido en la búsqueda del E2).
Lo único que cambia es el constructor del modelo: wdsd_inst.maskrcnn.crear_modelo se reemplaza por el de
src/original_model.py, así que entrenamiento y predicción usan la asignación SGW.

Variantes (ablación, book/05_modelo_diseno.md §4):
  maskrcnn  referencia del E2 (IoU), corrida en esta misma máquina como control de hardware
  sgw       SGW en RPN y RoI (modelo completo)
  sgw_rpn   SGW solo en RPN
  sgw_roi   SGW solo en la cabeza RoI
  sgw_c0    SGW sin piso (c = 1e-3 px)

Uso:
  python training/entrenar_sgw.py buscar --datos D:/wdsd_datos/datos_inst            # kappa en la rejilla, semilla 42, 2 épocas, solo val
  python training/entrenar_sgw.py final  --datos ... --variante sgw --kappa 0.45 --semillas 42,43,...
Una corrida terminada tiene resumen.json y se salta al relanzar.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SRC_E2 = Path(os.environ.get("WDSD_E2_SRC", RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "src"))
sys.path[:0] = [str(RAIZ / "src"), str(SRC_E2)]

FACTOR_LR_MASKRCNN = 2.0   # elegido en la búsqueda del E2 (results/e2/busqueda/maskrcnn_elegida.json)
EPOCAS_FINAL = 6
EPOCAS_BUSQUEDA = 2        # ceil(6 / 3), igual que el E2
SPLITS = {"buscar": ("val",), "final": ("val", "test", "train93")}
VARIANTES = {
    "maskrcnn": None,
    "sgw": dict(en_rpn=True, en_roi=True),
    "sgw_rpn": dict(en_rpn=True, en_roi=False),
    "sgw_roi": dict(en_rpn=False, en_roi=True),
    "sgw_c0": dict(en_rpn=True, en_roi=True, piso=1e-3),
}


def asegurar_train93(datos: Path):
    """Misma submuestra fija que el E2 (entrenar_e2.py cmd_datos: random.Random(0), 93 imágenes)."""
    p = datos / "coco" / "train93.json"
    if p.exists():
        return
    tr = json.loads((datos / "coco" / "train.json").read_text(encoding="utf8"))
    ids = sorted(random.Random(0).sample(sorted(i["id"] for i in tr["images"]), 93))
    sub = {"images": [i for i in tr["images"] if i["id"] in ids],
           "annotations": [x for x in tr["annotations"] if x["image_id"] in ids], "categories": tr["categories"]}
    p.write_text(json.dumps(sub), encoding="utf8")


def una(variante: str, kappa: float, semilla: int, modo: str, datos: Path, salida: Path, maquina: str) -> dict | None:
    import torch
    from wdsd_e2 import comun, evaluar, m_torchvision
    from wdsd_inst import maskrcnn
    import original_model as om

    nombre = variante if variante == "maskrcnn" else f"{variante}_k{kappa:g}"
    carpeta = salida / ("busqueda" if modo == "buscar" else "corridas") / f"{nombre}_s{semilla}"
    if (carpeta / "resumen.json").exists():
        print(f"-- {carpeta.name}: ya está, se salta")
        return None
    carpeta.mkdir(parents=True, exist_ok=True)
    comun.DATOS = datos         # ruta_json() y evaluar.* leen comun.DATOS en cada llamada
    m_torchvision.DATOS = datos  # m_torchvision lo importó por nombre: hay que cambiarlo también ahí
    if not hasattr(maskrcnn, "_crear_original"):
        maskrcnn._crear_original = maskrcnn.crear_modelo
    cfg = VARIANTES[variante]
    if cfg is None:
        maskrcnn.crear_modelo = maskrcnn._crear_original
    else:
        kw = {"kappa": kappa, **cfg}
        maskrcnn.crear_modelo = lambda preentrenado=True, _kw=kw: om.aplicar_sgw(maskrcnn._crear_original(preentrenado), **_kw)

    disp = comun.preparar_gpu(False)
    comun.fijar_semilla(semilla)
    epocas = EPOCAS_BUSQUEDA if modo == "buscar" else EPOCAS_FINAL
    t0 = time.time()
    r = m_torchvision.entrenar("maskrcnn", carpeta, semilla, epocas, FACTOR_LR_MASKRCNN, SPLITS[modo], disp,
                               preentrenado=True, conservar_pesos=modo == "final", log=print)
    hist = r.pop("historial", [])
    if any(not math.isfinite(h["perdida"]) for h in hist):
        raise RuntimeError("pérdida no finita")
    import pandas as pd
    pd.DataFrame(hist).to_csv(carpeta / "historial.csv", index=False)
    metricas = evaluar.busqueda(carpeta) if modo == "buscar" else evaluar.final(carpeta)
    ap = [h["val_AP"] for h in hist]
    resumen = {"modelo": nombre, "variante": variante, "kappa": None if cfg is None else kappa,
               "config_sgw": None if cfg is None else {"kappa": kappa, "piso": om.PISO_C_PX, **cfg},
               "modo": modo, "semilla": semilla, "epocas": epocas, "factor_lr": FACTOR_LR_MASKRCNN,
               "maquina": maquina, "fecha_fin": dt.datetime.now().isoformat(timespec="seconds"),
               "s_total": time.time() - t0, "entorno": comun.entorno(), **r, "metricas": metricas,
               "val_AP_sube_al_final": bool(len(ap) >= 2 and ap[-1] > max(ap[:-1]))}
    comun.escribir_json(carpeta / "resumen.json", resumen)
    v, t = metricas["val"], metricas.get("test") or {}
    print(f"== {nombre} {modo} s{semilla}: val F1 grupo {v['F1_grupo']:.3f}"
          + (f" | test F1 grupo {t['F1_grupo']:.3f}" if t else "") + f" | {resumen['s_total'] / 60:.1f} min", flush=True)
    del r
    torch.cuda.empty_cache()
    return resumen


def main():
    import original_model as om
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=("buscar", "final"))
    ap.add_argument("--datos", required=True, type=Path)
    # en D: por defecto: cada corrida final guarda mejor.pt (~170 MB) y C: tiene poco espacio
    ap.add_argument("--salida", type=Path, default=Path(os.environ.get("WDSD_E3_SALIDA", "D:/wdsd_e3/results")))
    ap.add_argument("--variante", default="sgw", choices=tuple(VARIANTES))
    ap.add_argument("--kappa", type=float, default=om.KAPPA_DEFECTO)
    ap.add_argument("--semillas", default="42")
    ap.add_argument("--maquina", default=os.environ.get("WDSD_MAQUINA", "3050"))
    a = ap.parse_args()
    asegurar_train93(a.datos)
    if a.modo == "buscar":
        trabajos = [("sgw", k, 42) for k in om.REJILLA_KAPPA]
    else:
        trabajos = [(a.variante, a.kappa, int(s)) for s in a.semillas.split(",")]
    for variante, kappa, semilla in trabajos:
        for intento in (1, 2):  # como el E2: un fallo técnico se repite una vez con la misma semilla
            try:
                una(variante, kappa, semilla, a.modo, a.datos, a.salida, a.maquina)
                break
            except Exception:
                traceback.print_exc()
                if intento == 2:
                    nombre = variante if variante == "maskrcnn" else f"{variante}_k{kappa:g}"
                    c = a.salida / ("busqueda" if a.modo == "buscar" else "corridas") / f"{nombre}_s{semilla}"
                    c.mkdir(parents=True, exist_ok=True)
                    (c / "fallo.json").write_text(json.dumps({"error": traceback.format_exc()}), encoding="utf8")


if __name__ == "__main__":
    main()
