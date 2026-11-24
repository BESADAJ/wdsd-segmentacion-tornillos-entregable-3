"""Verificación numérica del teorema 1 (b) y de la proposición 2 (book/05_modelo_diseno.md).

Para tornillos cuadrados de lado a = 3..60 px, perturba cada caja con desplazamientos de las esquinas de norma
fija (0.5, 1, 2 px) en direcciones aleatorias y mide el cociente |Δsimilitud| / ||Δcaja||. Compara el máximo
empírico de SGW con su cota uniforme 1 / (sqrt(2) kappa c) y con el del IoU, que crece como 1/a.

Uso: python experiments/verificacion_teorema.py [carpeta_salida]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original_model import KAPPA_DEFECTO, PISO_C_PX, REJILLA_KAPPA, SimilitudSGW  # noqa: E402

salida = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "teoria")
salida.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(0)
lados = np.arange(3, 61)
filas = []
for kappa in REJILLA_KAPPA:
    sim = SimilitudSGW(kappa, PISO_C_PX)
    for a in lados:
        gt = torch.tensor([[100.0, 100.0, 100.0 + a, 100.0 + a]], dtype=torch.float64)
        for norma in (0.5, 1.0, 2.0):
            u = torch.randn(2000, 4, generator=g, dtype=torch.float64)
            u = u / u.norm(dim=1, keepdim=True) * norma
            b = gt.repeat(2000, 1)
            bp = b + u
            ds = (sim(gt, b)[0] - sim(gt, bp)[0]).abs() / norma
            di = (box_iou(gt, b)[0] - box_iou(gt, bp)[0]).abs() / norma
            filas.append({"kappa": kappa, "lado_px": int(a), "norma_px": norma,
                          "sgw_max": float(ds.max()), "iou_max": float(di.max()),
                          "cota_sgw": sim.cota_lipschitz("esquinas")})

with open(salida / "verificacion_teorema.json", "w", encoding="utf8") as f:
    json.dump(filas, f, indent=1)
viol = [r for r in filas if r["sgw_max"] > r["cota_sgw"] * (1 + 1e-9)]
print(f"{len(filas)} combinaciones; violaciones de la cota de SGW: {len(viol)}")
for kappa in REJILLA_KAPPA:
    sel = [r for r in filas if r["kappa"] == kappa and r["norma_px"] == 1.0]
    peq = [r for r in sel if r["lado_px"] <= 8]
    gra = [r for r in sel if r["lado_px"] >= 40]
    print(f"kappa={kappa}: cota={sel[0]['cota_sgw']:.3f} | SGW max (a<=8)={max(r['sgw_max'] for r in peq):.3f} "
          f"(a>=40)={max(r['sgw_max'] for r in gra):.3f} | IoU max (a<=8)={max(r['iou_max'] for r in peq):.3f} "
          f"(a>=40)={max(r['iou_max'] for r in gra):.3f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    sel = [r for r in filas if r["kappa"] == KAPPA_DEFECTO and r["norma_px"] == 1.0]
    ax.plot([r["lado_px"] for r in sel], [r["iou_max"] for r in sel], label="IoU (empírico)", color="#b2182b")
    ax.plot([r["lado_px"] for r in sel], [r["sgw_max"] for r in sel], label=f"SGW κ={KAPPA_DEFECTO} (empírico)",
            color="#2166ac")
    ax.axhline(sel[0]["cota_sgw"], ls="--", color="#2166ac", label="cota del teorema 1 (b)")
    ax.set_xlabel("lado del tornillo (px)")
    ax.set_ylabel("máx |Δ similitud| / ||Δ caja|| (1/px)")
    ax.set_title("Sensibilidad a 1 px de error de localización")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "verificacion_teorema.png", dpi=150)
    print("figura:", salida / "verificacion_teorema.png")
except ImportError:
    print("matplotlib no está; se omite la figura")
