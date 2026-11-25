"""Análisis confirmatorio del E3 según experiments/protocolo.md (congelado con el tag e3-protocolo).

Unidad de independencia: el grupo de escena (14 en test). Para cada modelo, F1 por grupo = media del F1 de sus
imágenes en test, promediada sobre sus semillas (como en el E2).

Familias:
  A. R3 (primaria): SGW frente a cada rival (13 del E2 + clásico; Mask R-CNN = control de 10 semillas en la misma
     GPU). Unilateral (SGW > rival). Tres pruebas: Wilcoxon por grupo, bootstrap pareado por grupo y, solo frente a
     Mask R-CNN, t pareado sobre semillas. Holm sobre la familia en cada prueba; decide el máximo de los p ajustados.
     Relevancia: diferencia >= delta = 0,20 con IC inferior > 0.
  B. H-M (mecanismo): recall del tercil pequeño de SGW frente a Mask R-CNN, margen 0,05 (H1: diferencia > 0,05).
     Predicciones secundarias 2, 3 y 5 de book/05_modelo_diseno.md §4, con Holm entre ellas.
  C. Ablación: sgw_rpn, sgw_roi y sgw_c0 frente a sgw en F1 por grupo, bilateral, Holm sobre las tres.
  D. Friedman + Iman-Davenport + Nemenyi sobre todos los modelos; diagrama de diferencia crítica.
  Exploratorio: potencia y diferencia mínima detectable (no se registró antes; se etiqueta así), sensibilidad
  frente a Mask R-CNN del E2 (3060) y la brecha de hardware.

Uso: python experiments/analisis_confirmatorio.py [--e3 D:/wdsd_e3/results] [--e2 <E2>/results/e2/corridas]
                                                  [--salida results/confirmatorio]
Requiere haber corrido antes experiments/terciles_e3.py (terciles_conteos.csv).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DELTA, ALPHA, MARGEN, B = 0.20, 0.05, 0.05, 10_000
SEM_E3, SEM_E2 = range(42, 52), range(42, 47)
SGW, CTRL = "sgw_k0.45", "maskrcnn"
ABL = {"sgw_rpn_k0.45": "sgw_rpn (sin SGW en RoI)", "sgw_roi_k0.45": "sgw_roi (sin SGW en RPN)",
       "sgw_c0_k0.45": "sgw_c0 (sin piso)"}
RIVALES_E2 = ["mask2former", "yolov7", "yolov9", "yolo26", "yolo11", "cascade", "htc", "pointrend", "msrcnn",
              "condinst", "yolov8", "solov2", "clasico"]
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- utilidades estadísticas
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon(d, alt):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative=alt).pvalue)


def bootstrap(d, alt, nulo=0.0):
    """Bootstrap pareado por grupo: IC 95 % percentil de la media y p (proporción de medias del lado del nulo)."""
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    if alt == "greater":
        p = (np.sum(m <= nulo) + 1) / (B + 1)
    else:
        p = min(1.0, 2 * (min(np.sum(m <= nulo), np.sum(m >= nulo)) + 1) / (B + 1))
    return float(lo), float(hi), float(p)


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sign(a[:, None] - b[None, :])).mean())


def cohen_pareado(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    est = d.mean() / sd if sd > 0 else np.nan
    bs = d[rng.integers(0, len(d), size=(B, len(d)))]
    v = bs.mean(1) / bs.std(1, ddof=1)
    lo, hi = np.nanpercentile(v[np.isfinite(v)], [2.5, 97.5])
    return float(est), float(lo), float(hi)


# ---------------------------------------------------------------- carga de datos
def f1_por_grupo(carpeta: Path):
    d = pd.read_csv(carpeta / "test_por_imagen.csv")
    return d.groupby("grupo")["f1"].mean()


def cargar(e3: Path, e2: Path):
    filas = []
    for m in [SGW, CTRL, *ABL]:
        for s in SEM_E3:
            for g, v in f1_por_grupo(e3 / "corridas" / f"{m}_s{s}").items():
                filas.append((m, s, g, v, "3050"))
    for m in RIVALES_E2 + ["maskrcnn"]:
        for s in SEM_E2:
            nombre = "maskrcnn_E2" if m == "maskrcnn" else m
            for g, v in f1_por_grupo(e2 / f"{m}_s{s}").items():
                filas.append((nombre, s, g, v, "3060"))
    return pd.DataFrame(filas, columns=["modelo", "semilla", "grupo", "f1_grupo", "maquina"])


def recall_tercil(conteos: pd.DataFrame, tercil: str):
    """Recall del tercil por (modelo, semilla, grupo); grupos sin tornillos del tercil quedan fuera."""
    c = conteos[conteos.tercil == tercil].groupby(["modelo", "semilla", "grupo"])[["tp", "fn"]].sum()
    c = c[(c.tp + c.fn) > 0]
    return (c.tp / (c.tp + c.fn)).rename("recall").reset_index()


def recall_micro(conteos, tercil):
    c = conteos[conteos.tercil == tercil].groupby(["modelo", "semilla"])[["tp", "fn"]].sum()
    return (c.tp / (c.tp + c.fn)).unstack(0)


# ---------------------------------------------------------------- análisis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", type=Path, default=Path("D:/wdsd_e3/results"))
    ap.add_argument("--e2", type=Path, default=RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "results" / "e2"
                    / "corridas")
    ap.add_argument("--salida", type=Path, default=RAIZ / "results" / "confirmatorio")
    a = ap.parse_args()
    out = a.salida
    out.mkdir(parents=True, exist_ok=True)

    P = cargar(a.e3, a.e2)
    P.to_csv(out / "f1_grupo_por_semilla.csv", index=False)
    F = P.pivot_table(index="grupo", columns="modelo", values="f1_grupo", aggfunc="mean")
    assert F.shape[0] == 14 and not F.isna().any().any(), F.shape
    F.to_csv(out / "f1_grupo_promedio_semillas.csv")
    S = P.groupby(["modelo", "semilla"]).f1_grupo.mean().unstack(0)  # F1_grupo por corrida

    # verificación contra resumen.json
    verif = []
    for m in [SGW, CTRL, *ABL]:
        for s in SEM_E3:
            r = json.loads((a.e3 / "corridas" / f"{m}_s{s}" / "resumen.json").read_text(encoding="utf8"))
            verif.append(abs(r["metricas"]["test"]["F1_grupo"] - S.loc[s, m]))
    max_dif = float(max(verif))
    print(f"max |dif| contra resumen.json: {max_dif:.2e}")

    tabla = pd.DataFrame({"F1_grupo_media": S.mean(), "DE_semillas": S.std(ddof=1), "n_semillas": S.count()})
    tabla = tabla.sort_values("F1_grupo_media", ascending=False)
    tabla.to_csv(out / "tabla_principal.csv")

    # ---------------- A. familia R3
    rivales = [CTRL] + RIVALES_E2
    filas = []
    for r in rivales:
        d = (F[SGW] - F[r]).to_numpy()
        lo, hi, pb = bootstrap(d, "greater")
        pt = np.nan
        if r == CTRL:
            pt = float(stats.ttest_rel(S[SGW], S[CTRL], alternative="greater").pvalue)
        dc, dclo, dchi = cohen_pareado(d)
        filas.append(dict(rival=r, dif_media=d.mean(), ic95_lo=lo, ic95_hi=hi, grupos_sgw_mejor=int((d > 0).sum()),
                          p_wilcoxon=wilcoxon(d, "greater"), p_bootstrap=pb, p_t_semillas=pt,
                          cliff_delta=cliff(F[SGW], F[r]), d_cohen=dc, d_cohen_lo=dclo, d_cohen_hi=dchi))
    A = pd.DataFrame(filas)
    A["p_holm_wilcoxon"] = holm(A.p_wilcoxon)
    A["p_holm_bootstrap"] = holm(A.p_bootstrap)
    A["p_holm_t"] = np.nan
    A.loc[A.rival == CTRL, "p_holm_t"] = A.loc[A.rival == CTRL, "p_t_semillas"]  # única prueba t de la familia
    A["p_decision"] = A[["p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t"]].max(axis=1)

    def decide(r):
        if r.p_decision < ALPHA and r.dif_media >= DELTA and r.ic95_lo > 0:
            return "superior y relevante (cumple R3)"
        if r.p_decision < ALPHA:
            return "superior, sin relevancia (< delta)"
        if r.ic95_lo > -DELTA and r.ic95_hi < DELTA:
            return "equivalente dentro de delta"
        return "no concluyente"
    A["decision"] = A.apply(decide, axis=1)
    A.to_csv(out / "familia_R3.csv", index=False)
    r3 = bool((A.decision == "superior y relevante (cumple R3)").all())

    # ---------------- B. H-M y predicciones
    conteos = pd.read_csv(a.e3 / "analisis" / "terciles_conteos.csv")
    conteos = conteos[conteos.modelo != "maskrcnn_E2"]
    RP = recall_tercil(conteos, "pequeño").pivot_table(index="grupo", columns="modelo", values="recall",
                                                        aggfunc="mean")
    RG = recall_tercil(conteos, "grande").pivot_table(index="grupo", columns="modelo", values="recall",
                                                       aggfunc="mean")
    MP, MG = recall_micro(conteos, "pequeño"), recall_micro(conteos, "grande")
    RP.to_csv(out / "recall_pequeno_por_grupo.csv")

    def contraste(nombre, d_grupo, d_sem, alt, nulo):
        d_grupo = pd.Series(d_grupo).dropna().to_numpy()
        lo, hi, pb = bootstrap(d_grupo, alt, nulo)
        pt = float(stats.ttest_1samp(d_sem, nulo, alternative=alt).pvalue)
        return dict(contraste=nombre, n_grupos=len(d_grupo), dif_media_grupos=d_grupo.mean(), ic95_lo=lo,
                    ic95_hi=hi, dif_media_semillas=float(np.mean(d_sem)),
                    semillas_a_favor=int((np.asarray(d_sem) > nulo).sum()), n_semillas=len(d_sem), nulo=nulo,
                    p_wilcoxon=wilcoxon(d_grupo - nulo, alt), p_bootstrap=pb, p_t_semillas=pt)

    hm = contraste("H-M: recall pequeño SGW - Mask R-CNN > 0,05", RP[SGW] - RP[CTRL], MP[SGW] - MP[CTRL],
                   "greater", MARGEN)
    hm["p_decision"] = max(hm["p_wilcoxon"], hm["p_bootstrap"], hm["p_t_semillas"])
    hm["confirmada"] = bool(hm["p_decision"] < ALPHA and hm["ic95_lo"] > MARGEN)
    hm["n_grupos_sin_tornillos_pequenos"] = int(14 - hm["n_grupos"])

    preds = [
        contraste("P2: ganancia pequeño - ganancia grande > 0",
                  (RP[SGW] - RP[CTRL]) - (RG[SGW] - RG[CTRL]),
                  (MP[SGW] - MP[CTRL]) - (MG[SGW] - MG[CTRL]), "greater", 0.0),
        contraste("P3: recall pequeño sgw_roi - sgw_rpn > 0 (quitar RoI pesa más)",
                  RP["sgw_roi_k0.45"] - RP["sgw_rpn_k0.45"], MP["sgw_roi_k0.45"] - MP["sgw_rpn_k0.45"],
                  "greater", 0.0),
        contraste("P5: recall pequeño sgw - sgw_c0 > 0 (sin piso se pierde)",
                  RP[SGW] - RP["sgw_c0_k0.45"], MP[SGW] - MP["sgw_c0_k0.45"], "greater", 0.0),
    ]
    Pr = pd.DataFrame(preds)
    for c in ["p_wilcoxon", "p_bootstrap", "p_t_semillas"]:
        Pr[c.replace("p_", "p_holm_")] = holm(Pr[c])
    Pr["p_decision"] = Pr[["p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t_semillas"]].max(axis=1)
    Pr["se_cumple"] = (Pr.p_decision < ALPHA) & (Pr.ic95_lo > 0)
    p5_de = {m: float(MP[m].std(ddof=1)) for m in [SGW, "sgw_c0_k0.45", CTRL]}
    Pr.to_csv(out / "predicciones.csv", index=False)
    pd.DataFrame([hm]).to_csv(out / "hipotesis_mecanismo.csv", index=False)
    terc = pd.DataFrame({t: recall_micro(conteos, t).agg(["mean", "std"]).stack() for t in
                         ["pequeño", "mediano", "grande"]})
    terc.to_csv(out / "recall_terciles_media_de.csv")

    # ---------------- C. ablación
    filas = []
    for m, etiqueta in ABL.items():
        d = (F[m] - F[SGW]).to_numpy()
        lo, hi, pb = bootstrap(d, "two-sided")
        filas.append(dict(variante=etiqueta, dif_media_vs_sgw=d.mean(), ic95_lo=lo, ic95_hi=hi,
                          p_wilcoxon=wilcoxon(d, "two-sided"), p_bootstrap=pb,
                          p_t_semillas=float(stats.ttest_rel(S[m], S[SGW]).pvalue),
                          recall_pequeno_media=float(MP[m].mean())))
    C = pd.DataFrame(filas)
    for c in ["p_wilcoxon", "p_bootstrap", "p_t_semillas"]:
        C[c.replace("p_", "p_holm_")] = holm(C[c])
    C["p_decision"] = C[["p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t_semillas"]].max(axis=1)
    C["significativa"] = C.p_decision < ALPHA
    C.to_csv(out / "ablacion.csv", index=False)

    # ---------------- supuestos y sobreajuste (SGW frente a Mask R-CNN)
    d_g = (F[SGW] - F[CTRL]).to_numpy()
    d_s = (S[SGW] - S[CTRL]).to_numpy()
    sup = dict(
        shapiro_dif_grupos_p=float(stats.shapiro(d_g).pvalue),
        shapiro_dif_semillas_p=float(stats.shapiro(d_s).pvalue),
        asimetria_dif_grupos=float(stats.skew(d_g)),
        levene_semillas_p=float(stats.levene(S[SGW], S[CTRL]).pvalue),
        nota="Wilcoxon y bootstrap no suponen normalidad; el t pareado sobre semillas sí (Shapiro sobre sus "
             "diferencias). Si Shapiro rechaza, decide el máximo de los p, que ya es conservador.")
    brecha = []
    for m in [SGW, CTRL, *ABL]:
        for s in SEM_E3:
            r = json.loads((a.e3 / "corridas" / f"{m}_s{s}" / "resumen.json").read_text(encoding="utf8"))
            mt = r["metricas"]
            brecha.append(dict(modelo=m, semilla=s, train93_F1_grupo=mt["train93"]["F1_grupo"],
                               val_F1_grupo=mt["val"]["F1_grupo"], test_F1_grupo=mt["test"]["F1_grupo"],
                               brecha_train93_val=mt["train93"]["F1_grupo"] - mt["val"]["F1_grupo"],
                               brecha_val_test=mt["val"]["F1_grupo"] - mt["test"]["F1_grupo"],
                               mejor_epoca=r.get("mejor_epoca"), val_AP_sube_al_final=r.get("val_AP_sube_al_final")))
    Bg = pd.DataFrame(brecha)
    Bg.to_csv(out / "sobreajuste_por_corrida.csv", index=False)
    bs, bc = Bg[Bg.modelo == SGW].set_index("semilla"), Bg[Bg.modelo == CTRL].set_index("semilla")
    sobre = dict(
        brecha_train93_val_sgw=float(bs.brecha_train93_val.mean()),
        brecha_train93_val_maskrcnn=float(bc.brecha_train93_val.mean()),
        p_t_brecha_train93_val=float(stats.ttest_rel(bs.brecha_train93_val, bc.brecha_train93_val).pvalue),
        brecha_val_test_sgw=float(bs.brecha_val_test.mean()),
        brecha_val_test_maskrcnn=float(bc.brecha_val_test.mean()),
        p_t_brecha_val_test=float(stats.ttest_rel(bs.brecha_val_test, bc.brecha_val_test).pvalue),
        corridas_con_val_AP_subiendo_al_final={m: int(Bg[Bg.modelo == m].val_AP_sube_al_final.sum())
                                              for m in [SGW, CTRL, *ABL]})

    # ---------------- D. Friedman / Iman-Davenport / Nemenyi
    modelos = [SGW] + rivales
    X = F[modelos].to_numpy()
    N, k = X.shape
    rangos = stats.rankdata(-X, axis=1).mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(rangos ** 2) - k * (k + 1) ** 2 / 4)
    p_chi2 = float(stats.chi2.sf(chi2, k - 1))
    Fid = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = float(stats.f.sf(Fid, k - 1, (k - 1) * (N - 1)))
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    CD = float(q * np.sqrt(k * (k + 1) / (6 * N)))
    R = pd.Series(rangos, index=modelos).sort_values()
    R.rename("rango_medio").to_csv(out / "friedman_rangos.csv")
    fried = dict(N_grupos=N, k_modelos=k, chi2=float(chi2), p_chi2=p_chi2, iman_davenport_F=float(Fid),
                 p_iman_davenport=p_id, nemenyi_q=float(q), CD=CD,
                 rango_sgw=float(R[SGW]), mejor_rango=R.index[0],
                 empatados_con_sgw_por_nemenyi=[m for m in R.index if m != SGW and abs(R[m] - R[SGW]) < CD])
    (out / "friedman.json").write_text(json.dumps(fried, indent=1, ensure_ascii=False), encoding="utf8")
    diagrama_cd(R, CD, out / "diferencia_critica.png")

    # ---------------- exploratorio: potencia, sensibilidad y hardware
    d_ctrl = (F[SGW] - F[CTRL]).to_numpy()
    n = len(d_ctrl)
    a_h = ALPHA / len(rivales)  # el peor caso de Holm
    mde = lambda sd, n, al: float((stats.t.ppf(1 - al, n - 1) + stats.t.ppf(0.90, n - 1)) * sd / np.sqrt(n))
    d_sem = (S[SGW] - S[CTRL]).to_numpy()
    poder = dict(
        etiqueta="exploratorio (el análisis de potencia no se registró antes de las corridas)",
        sd_dif_grupos=float(d_ctrl.std(ddof=1)), n_grupos=n,
        mde_grupos_alfa_holm_peor=mde(d_ctrl.std(ddof=1), n, a_h), mde_grupos_alfa_005=mde(d_ctrl.std(ddof=1), n,
                                                                                            ALPHA),
        sd_dif_semillas=float(d_sem.std(ddof=1)), n_semillas=len(d_sem),
        mde_semillas_alfa_005=mde(d_sem.std(ddof=1), len(d_sem), ALPHA),
        potencia_t_grupos_para_delta=float(stats.nct.sf(stats.t.ppf(1 - a_h, n - 1), n - 1,
                                                        DELTA / (d_ctrl.std(ddof=1) / np.sqrt(n)))),
        nota="potencia aproximada por t pareado como sustituto de Wilcoxon")
    d_e2 = (F[SGW] - F["maskrcnn_E2"]).to_numpy()
    lo, hi, pb = bootstrap(d_e2, "greater")
    sens = dict(contraste="SGW (3050) - Mask R-CNN del E2 (3060)", dif_media=float(d_e2.mean()), ic95_lo=lo,
                ic95_hi=hi, p_wilcoxon=wilcoxon(d_e2, "greater"), p_bootstrap=pb)
    hw = (S[CTRL].loc[list(SEM_E2)] - S["maskrcnn_E2"].loc[list(SEM_E2)])
    hard = dict(contraste="Mask R-CNN 3050 - 3060, semillas 42-46", dif_media=float(hw.mean()),
                p_t_pareado=float(stats.ttest_rel(S[CTRL].loc[list(SEM_E2)],
                                                  S["maskrcnn_E2"].loc[list(SEM_E2)]).pvalue))
    resumen = dict(verificacion_max_dif=max_dif, R3_cumple=r3, H_M=hm, P5_DE_recall_pequeno=p5_de,
                   friedman=fried, potencia=poder, sensibilidad_maskrcnn_E2=sens, hardware=hard,
                   supuestos=sup, sobreajuste=sobre)
    (out / "resumen_confirmatorio.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=float),
                                                    encoding="utf8")

    # ---------------- informe en consola
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n== Tabla principal (F1 por grupo en test)\n", tabla.round(4).to_string())
    print("\n== Familia R3 (SGW > rival)\n", A[["rival", "dif_media", "ic95_lo", "ic95_hi", "grupos_sgw_mejor",
                                              "p_holm_wilcoxon", "p_holm_bootstrap", "p_holm_t", "cliff_delta",
                                              "decision"]].round(4).to_string(index=False))
    print(f"R3 cumple: {r3}")
    print("\n== H-M\n", pd.Series(hm).to_string())
    print("\n== Predicciones\n", Pr[["contraste", "dif_media_grupos", "ic95_lo", "ic95_hi", "semillas_a_favor",
                                     "p_decision", "se_cumple"]].round(4).to_string(index=False))
    print("DE entre semillas del recall pequeño:", {k: round(v, 4) for k, v in p5_de.items()})
    print("\n== Recall por tercil (micro, media y DE en semillas)\n", terc.round(3).to_string())
    print("\n== Ablación (vs sgw)\n", C[["variante", "dif_media_vs_sgw", "ic95_lo", "ic95_hi", "p_decision",
                                       "significativa", "recall_pequeno_media"]].round(4).to_string(index=False))
    print("\n== Friedman\n", json.dumps(fried, indent=1, ensure_ascii=False, default=float))
    print("\n== Potencia (exploratorio)\n", json.dumps(poder, indent=1, ensure_ascii=False))
    print("\n== Sensibilidad\n", sens, "\n", hard)
    print("\n== Supuestos\n", json.dumps(sup, indent=1, ensure_ascii=False))
    print("\n== Sobreajuste\n", json.dumps(sobre, indent=1, ensure_ascii=False))


def diagrama_cd(R: pd.Series, CD: float, ruta: Path):
    k = len(R)
    fig, ax = plt.subplots(figsize=(10, 0.35 * k + 1.5))
    for i, (m, r) in enumerate(R.items()):
        ax.plot([r, r], [0, -(i + 1)], color="0.6", lw=0.8)
        ax.plot(r, 0, "o", color="C3" if m == SGW else "0.3", ms=4)
        ax.text(r, -(i + 1), f" {m} ({r:.2f})", va="center", fontsize=8,
                color="C3" if m == SGW else "0.1", fontweight="bold" if m == SGW else None)
    ax.plot([1, 1 + CD], [1, 1], color="k", lw=2)
    ax.text(1 + CD / 2, 1.3, f"CD = {CD:.2f}", ha="center", fontsize=8)
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(-(k + 1), 2)
    ax.set_yticks([]); ax.set_xlabel("Rango medio de Friedman (1 = mejor) sobre 14 grupos de test")
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
