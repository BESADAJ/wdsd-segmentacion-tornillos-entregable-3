"""Análisis de robustez del E3 (capítulo 8) y prueba de la predicción 4.

Predicción 4 (book/05_modelo_diseno.md §4): al reducir la imagen al 75 % y al 50 % (tornillos más pequeños), la
ventaja de SGW sobre Mask R-CNN crece. Contraste: d(nivel) = F1(SGW) - F1(Mask R-CNN), y la interacción
d(0,5) - d(1,0) > 0 (y d(0,75) - d(1,0) > 0). Pruebas como en el protocolo: Wilcoxon por grupo (F1 por grupo
promediado sobre semillas), bootstrap pareado por grupo y t pareado sobre semillas; Holm sobre los 2 contrastes.
Para ruido y desenfoque se reporta d por nivel (exploratorio).

Uso: python experiments/analisis_robustez.py [--e3 D:/wdsd_e3/results]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
SGW, CTRL = "sgw_k0.45", "maskrcnn"
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    a = ap.parse_args()
    out = RAIZ / "results" / "robustez"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([pd.read_csv(p) for p in sorted((a.e3 / "robustez").glob("*.csv"))], ignore_index=True)
    G = d.groupby(["modelo", "semilla", "perturbacion", "nivel", "grupo"]).f1.mean().reset_index()
    S = G.groupby(["modelo", "semilla", "perturbacion", "nivel"]).f1.mean().unstack("modelo")  # F1 por grupo
    resumen = S.groupby(["perturbacion", "nivel"]).agg(["mean", "std"]).round(4)
    resumen.columns = [f"{m}_{s}" for m, s in resumen.columns]
    dif = (S[SGW] - S[CTRL]).groupby(["perturbacion", "nivel"]).agg(["mean", "std",
                                                                     lambda x: int((x > 0).sum())])
    dif.columns = ["dif_media", "dif_DE", "semillas_SGW_mejor"]
    tabla = resumen.join(dif.round(4))
    tabla.to_csv(out / "robustez_sgw_vs_maskrcnn.csv")
    print(tabla.to_string())

    # Predicción 4: interacción escala x modelo
    Gm = G.groupby(["modelo", "perturbacion", "nivel", "grupo"]).f1.mean().unstack("modelo")
    dg = (Gm[SGW] - Gm[CTRL])
    ds = (S[SGW] - S[CTRL])
    filas = []
    for nivel in (0.75, 0.5):
        ig = (dg.loc[("escala", nivel)] - dg.loc[("escala", 1.0)]).to_numpy()
        is_ = (ds.xs(("escala", nivel), level=["perturbacion", "nivel"])
               - ds.xs(("escala", 1.0), level=["perturbacion", "nivel"])).to_numpy()
        bs = ig[rng.integers(0, len(ig), (10_000, len(ig)))].mean(1)
        filas.append(dict(contraste=f"[d(escala {nivel}) - d(escala 1,0)] > 0", interaccion_grupos=ig.mean(),
                          ic95_lo=np.percentile(bs, 2.5), ic95_hi=np.percentile(bs, 97.5),
                          interaccion_semillas=is_.mean(), semillas_a_favor=int((is_ > 0).sum()),
                          p_wilcoxon=float(stats.wilcoxon(ig, alternative="greater").pvalue),
                          p_bootstrap=float((np.sum(bs <= 0) + 1) / 10_001),
                          p_t_semillas=float(stats.ttest_1samp(is_, 0, alternative="greater").pvalue)))
    P = pd.DataFrame(filas)
    for c in ("p_wilcoxon", "p_bootstrap", "p_t_semillas"):
        P[c.replace("p_", "p_holm_")] = holm(P[c])
    P["p_decision"] = P[["p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t_semillas"]].max(axis=1)
    P["se_cumple"] = (P.p_decision < 0.05) & (P.ic95_lo > 0)
    P.to_csv(out / "prediccion4_escala.csv", index=False)
    print(P.round(4).to_string(index=False))
    control = d[(d.perturbacion == "escala") & (d.nivel == 1.0)]
    (out / "control.json").write_text(json.dumps({"n_modelos_semillas": int(control.groupby(
        ["modelo", "semilla"]).ngroups)}, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
