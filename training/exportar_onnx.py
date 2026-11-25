"""Exportación a ONNX, prueba de ida y vuelta y costo de inferencia en CPU (E3 Bloque B, §5.2).

Selección (solo con validación, nunca con test): de las 10 semillas finales de cada modelo se toma la de mayor F1 por
grupo en val. SGW solo cambia la asignación de entrenamiento: en inferencia la red es una Mask R-CNN estándar, así que
SGW y el control se exportan con el mismo código.

Pasos por modelo:
1. Carga mejor.pt en CPU (FP32) y exporta el grafo completo de torchvision (preprocesado, RPN, RoI, máscaras) con
   entrada fija 3x720x1280, opset 17.
2. Guarda los pesos en FP16 (comprimir_pesos; el FP32 pesa ~170 MB y no cabe en GitHub). Entradas, salidas y
   cómputo siguen en FP32.
3. Ida y vuelta sobre 10 imágenes fijas de test: compara cajas, scores y máscaras de PyTorch (CPU, FP32) con ONNX
   Runtime (FP32 y FP16). Tolerancia declarada: |score| < 1e-3 en las detecciones emparejadas e IoU de máscara >= 0,95.
4. Métrica primaria del modelo exportado en las 108 imágenes de test, frente a la del modelo en memoria
   (umbral de val de la corrida), y prueba de equivalencia TOST por grupo con margen delta = 0,20.
5. Costo en CPU: latencia mediana y p95 (lote 1, 3 de calentamiento, 20 medidas), memoria pico del proceso,
   parámetros y GFLOPs (FlopCounterMode en PyTorch).
Escribe artifacts/<modelo>.onnx (FP16), artifacts/<modelo>.metadata.json y results/exportacion/*.csv.

Uso: python training/exportar_onnx.py --datos D:/wdsd_datos/datos_inst [--modelos sgw_k0.45,maskrcnn]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SRC_E2 = Path(os.environ.get("WDSD_E2_SRC", RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "src"))
sys.path[:0] = [str(RAIZ / "src"), str(SRC_E2)]
H, W = 720, 1280


def elegir_semilla(e3: Path, modelo: str) -> tuple[int, dict]:
    mejores = []
    for r in sorted((e3 / "corridas").glob(f"{modelo}_s*/resumen.json")):
        d = json.loads(r.read_text(encoding="utf8"))
        mejores.append((d["metricas"]["val"]["F1_grupo"], d["semilla"], d))
    f1, s, d = max(mejores, key=lambda t: t[0])
    return s, d


def comprimir_pesos(src: Path, dst: Path):
    """Guarda en FP16 los pesos float32 de >= 1024 elementos y los vuelve a FP32 con un nodo Cast al cargar.

    La conversión completa a FP16 (onnxruntime.transformers.float16) rompe la inferencia de tipos de los nodos If del
    grafo de Mask R-CNN, y la cuantización dinámica INT8 de las convoluciones (ConvInteger) no tiene kernel en CPU.
    Comprimir solo el almacenamiento deja el cómputo en FP32: el archivo pasa de ~168 a ~84 MB (cabe en GitHub) y la
    única pérdida es el redondeo de los pesos a FP16.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(src))
    g = m.graph
    nodos, inits = [], []
    for init in list(g.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        w = numpy_helper.to_array(init)
        if w.size < 1024:
            continue
        g.initializer.remove(init)
        inits.append(numpy_helper.from_array(w.astype(np.float16), init.name + "__fp16"))
        nodos.append(helper.make_node("Cast", [init.name + "__fp16"], [init.name], to=TensorProto.FLOAT))
    g.initializer.extend(inits)
    viejos = list(g.node)
    del g.node[:]
    g.node.extend(nodos + viejos)
    onnx.save(m, str(dst))


def memoria_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--modelos", default="sgw_k0.45,maskrcnn")
    a = ap.parse_args()
    import onnx
    import onnxruntime as ort
    import pandas as pd
    import torch
    from PIL import Image
    from pycocotools import mask as mu
    from scipy import stats
    from wdsd_e2.comun import filtrar
    from wdsd_inst import maskrcnn
    from wdsd_inst.metricas import a_rle, cargar_gt, pq_por_imagen

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    art, out = RAIZ / "artifacts", RAIZ / "results" / "exportacion"
    art.mkdir(exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    gt = cargar_gt(a.datos / "coco" / "test.json")
    imgs = sorted(gt.imgs.values(), key=lambda x: x["id"])
    rgb = {im["id"]: np.asarray(Image.open(im["ruta"]).convert("RGB")) for im in imgs}
    fijas = [im["id"] for im in imgs[::11]][:10]
    filas_rt, filas_costo = [], []

    for nombre in a.modelos.split(","):
        semilla, resumen = elegir_semilla(a.e3, nombre)
        carpeta = a.e3 / "corridas" / f"{nombre}_s{semilla}"
        umbral = resumen["metricas"]["umbral_score"]
        print(f"== {nombre}: semilla {semilla} (val F1 grupo {resumen['metricas']['val']['F1_grupo']:.3f})", flush=True)
        m = maskrcnn.crear_modelo(True)
        m.load_state_dict(torch.load(carpeta / "mejor.pt", map_location="cpu"))
        m.eval()

        def torch_pred(x: np.ndarray):
            with torch.no_grad():
                o = m([torch.from_numpy(x).permute(2, 0, 1).float().div(255)])[0]
            return o["boxes"].numpy(), o["scores"].numpy(), o["masks"][:, 0].numpy()

        f32 = art / f"{nombre}_fp32.onnx"
        f16 = art / f"{nombre}.onnx"
        ejemplo = torch.from_numpy(rgb[fijas[0]]).permute(2, 0, 1).float().div(255)
        torch.onnx.export(m, ([ejemplo],), str(f32), opset_version=17, dynamo=False,
                          input_names=["imagen"], output_names=["boxes", "labels", "scores", "masks"],
                          do_constant_folding=True)
        comprimir_pesos(f32, f16)
        print(f"  ONNX FP32 {f32.stat().st_size / 2**20:.0f} MB, FP16 {f16.stat().st_size / 2**20:.0f} MB", flush=True)

        sesiones = {p.stem.split("_")[-1] if p == f32 else "fp16": ort.InferenceSession(
            str(p), providers=["CPUExecutionProvider"]) for p in (f32, f16)}

        def ort_pred(s, x):
            b, l, sc, mk = s.run(None, {"imagen": x.transpose(2, 0, 1).astype(np.float32)[None][0] / 255.0})
            return b, sc, mk[:, 0]

        # ---- ida y vuelta
        for iid in fijas:
            bt, st, mt = torch_pred(rgb[iid])
            for tipo, s in sesiones.items():
                bo, so, mo = ort_pred(s, rgb[iid])
                k = min(len(st), len(so))
                ious = []
                for i in range(k):
                    a1, a2 = mt[i] > 0.5, mo[i] > 0.5
                    u = (a1 | a2).sum()
                    ious.append((a1 & a2).sum() / u if u else 1.0)
                filas_rt.append(dict(modelo=nombre, formato=tipo, image_id=iid, n_torch=len(st), n_onnx=len(so),
                                     max_abs_score=float(np.abs(st[:k] - so[:k]).max()) if k else 0.0,
                                     max_abs_caja_px=float(np.abs(bt[:k] - bo[:k]).max()) if k else 0.0,
                                     iou_mascara_min=float(min(ious)) if ious else 1.0,
                                     n_sobre_umbral_igual=int((st >= umbral).sum() == (so >= umbral).sum())))

        # ---- métrica primaria: torch en memoria vs ONNX FP16, 108 imágenes
        def evaluar(pred):
            preds = []
            for im in imgs:
                _, sc, mk = pred(rgb[im["id"]])
                preds += [{"image_id": im["id"], "category_id": 1, "segmentation": a_rle(x > 0.5), "score": float(v)}
                          for v, x in zip(sc, mk) if v >= 0.05 and (x > 0.5).any()]
            t = pq_por_imagen(gt, filtrar(preds), umbral)
            return t.groupby("grupo").f1.mean()
        g_t = evaluar(torch_pred)
        g_o = evaluar(lambda x: ort_pred(sesiones["fp16"], x))
        d = (g_o - g_t).to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d)) if d.std(ddof=1) > 0 else 1e-12
        p_tost = float(max(stats.t.sf((d.mean() + 0.20) / se, len(d) - 1), stats.t.cdf((d.mean() - 0.20) / se,
                                                                                        len(d) - 1)))

        # ---- costo en CPU
        lat = {}
        for tipo, fn in (("torch_fp32", torch_pred), ("onnx_fp32", lambda x: ort_pred(sesiones["fp32"], x)),
                         ("onnx_fp16", lambda x: ort_pred(sesiones["fp16"], x))):
            for iid in fijas[:3]:
                fn(rgb[iid])
            ts = []
            for k in range(20):
                t0 = time.perf_counter(); fn(rgb[fijas[k % len(fijas)]]); ts.append((time.perf_counter() - t0) * 1000)
            lat[tipo] = (float(np.median(ts)), float(np.percentile(ts, 95)))
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            torch_pred(rgb[fijas[0]])
        costo = dict(modelo=nombre, semilla=semilla, parametros=int(sum(p.numel() for p in m.parameters())),
                     gflops=fc.get_total_flops() / 1e9, mb_onnx_fp32=f32.stat().st_size / 2**20,
                     mb_onnx_fp16=f16.stat().st_size / 2**20, memoria_pico_proceso_mb=memoria_mb(),
                     F1_grupo_test_torch=float(g_t.mean()), F1_grupo_test_onnx_fp16=float(g_o.mean()),
                     dif_F1_onnx_menos_torch=float(d.mean()), p_TOST_delta_020=p_tost,
                     cpu=platform.processor(), hilos=torch.get_num_threads(),
                     **{f"lat_{k}_mediana_ms": v[0] for k, v in lat.items()},
                     **{f"lat_{k}_p95_ms": v[1] for k, v in lat.items()})
        filas_costo.append(costo)
        print("  ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in costo.items()}), flush=True)

        import torchvision
        meta = dict(modelo=nombre, archivo=f16.name, formato="ONNX opset 17, pesos almacenados en FP16 (Cast a FP32 al cargar), cómputo FP32",
                    semilla=semilla, criterio_seleccion="mayor F1 por grupo en val entre las 10 semillas finales",
                    entrenamiento=dict(epocas=resumen["epocas"], mejor_epoca=resumen.get("mejor_epoca"),
                                       lr_inicial=resumen.get("lr_inicial"), lote=2, optimizador="SGD",
                                       config_sgw=resumen.get("config_sgw")),
                    preprocesamiento=dict(entrada="RGB uint8 HxWx3 -> float32 3xHxW en [0, 1]",
                                          resolucion=[H, W], normalizacion_interna="media/DE de ImageNet (en el grafo)",
                                          redimension_interna="lado corto 720, largo <= 1280 (en el grafo)"),
                    esquema_entrada=dict(nombre="imagen", tipo="float32", forma=[3, H, W], rango=[0.0, 1.0]),
                    esquema_salida=dict(boxes="N x 4 (x1,y1,x2,y2) px", scores="N", labels="N (1 = tornillo)",
                                        masks="N x 1 x H x W probabilidad; umbral 0,5"),
                    umbral_score=umbral, metricas_val=resumen["metricas"]["val"],
                    metricas_test=resumen["metricas"]["test"], costo_cpu=costo,
                    versiones=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                   onnx=onnx.__version__, onnxruntime=ort.__version__,
                                   python=platform.python_version()))
        (art / f"{nombre}.metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf8")
        f32.unlink()  # solo se conserva el FP16

    pd.DataFrame(filas_rt).to_csv(out / "ida_y_vuelta.csv", index=False)
    pd.DataFrame(filas_costo).to_csv(out / "costo_cpu.csv", index=False)
    rt = pd.DataFrame(filas_rt)
    print(rt.groupby(["modelo", "formato"])[["max_abs_score", "max_abs_caja_px", "iou_mascara_min"]]
          .agg(["max", "min"]).round(4).to_string())


if __name__ == "__main__":
    main()
