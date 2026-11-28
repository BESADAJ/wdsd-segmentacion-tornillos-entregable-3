"""Funciones que construyen las figuras del dashboard, con una plantilla y una paleta comunes (5 colores)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Paleta limitada (5 colores, la misma de assets/style.css): tinta (texto y modelos en general), naranja (SGW),
# azul (Mask R-CNN), verde (anotación real) y gris (resto). El naranja/azul se distinguen también por forma o patrón.
OSCURO, SGW, MRCNN, REAL, GRIS = "#2C3E50", "#E67E22", "#2980B9", "#18BC9C", "#95A5A6"
COLOR_MODELO = {"sgw_k0.45": SGW, "maskrcnn": MRCNN}
# Particiones: tres tonos de la tinta, para no confundirlas con los modelos (el naranja queda para SGW) y que se
# distingan también por luminosidad (daltonismo, proyector).
PARTICION = {"train": "#AEB6BF", "val": "#5D6D7E", "test": OSCURO}
ORDEN_PARTICION = ["train", "val", "test"]
TEXTO_2, REJILLA = "#5D6D7E", "#E9EDF1"

# Plantilla única: tipografía de la app, tamaños legibles desde un proyector, coma decimal y leyenda arriba.
pio.templates["wdsd"] = go.layout.Template(layout=dict(
    font=dict(family="Lato, Segoe UI, sans-serif", size=14, color=OSCURO),
    title=dict(x=0, xref="container", xanchor="left", y=0.98, yanchor="top", pad=dict(l=4), font=dict(size=16, color=OSCURO, weight=700)),
    colorway=[OSCURO, SGW, MRCNN, REAL, GRIS], paper_bgcolor="white", plot_bgcolor="white", separators=",.",
    xaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="#C9D1D9", ticks="outside", tickcolor="#C9D1D9",
               tickfont=dict(size=13, color=TEXTO_2), title=dict(font=dict(size=14), standoff=10), automargin=True),
    yaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="#C9D1D9", tickfont=dict(size=13, color=TEXTO_2),
               title=dict(font=dict(size=14), standoff=10), automargin=True),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", font=dict(size=13),
                bgcolor="rgba(0,0,0,0)", title=dict(text="")),
    margin=dict(l=8, r=16, t=72, b=8), hoverlabel=dict(font_size=13, bgcolor="white", bordercolor="#C9D1D9",
                                                      font=dict(color=OSCURO)),
    colorscale=dict(sequential=[[0, "#F6F8FA"], [1, OSCURO]])))
pio.templates.default = "wdsd"


def vacia(mensaje: str) -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=GRIS))
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    return f


# ----------------------------------------------------------------------------------------------- EDA
def histograma_area(a: pd.DataFrame, cortes=(75.0, 156.7)) -> go.Figure:
    f = go.Figure()
    for split in [p for p in ORDEN_PARTICION if p in set(a.particion)]:
        d = a[a.particion == split]
        f.add_histogram(x=np.log10(d.area_px.clip(lower=1)), name=split, opacity=0.8, nbinsx=50,
                        marker_color=PARTICION[split],
                        hovertemplate="log10(área)=%{x:.2f}<br>tornillos=%{y}<extra>" + split + "</extra>")
    for c, t, pos in zip(cortes, ("pequeño | mediano", "mediano | grande"), ("top left", "top right")):
        f.add_vline(x=np.log10(c), line_dash="dash", line_color=OSCURO, annotation_text=t, annotation_position=pos)
    f.add_vline(x=np.log10(32 ** 2), line_dash="dot", line_color=GRIS, annotation_text="COCO small (32²)")
    f.update_layout(barmode="overlay", title="Área de cada tornillo (escala log)",
                    xaxis_title="log10(área en píxeles)", yaxis_title="Número de tornillos")
    return f


def histograma_conteo(i: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for split in [p for p in ORDEN_PARTICION if p in set(i.particion)]:
        d = i[i.particion == split]
        f.add_histogram(x=d.n_tornillos, name=split, opacity=0.8, xbins=dict(size=1), marker_color=PARTICION[split])
    f.update_layout(barmode="overlay", title="Tornillos por imagen", xaxis_title="Tornillos en la imagen",
                    yaxis_title="Número de imágenes")
    return f


def mapa_centros(a: pd.DataFrame) -> go.Figure:
    f = go.Figure(go.Histogram2d(x=a.cx, y=a.cy, nbinsx=32, nbinsy=18, colorscale=[[0, "#FFFFFF"], [0.15, "#D5DBE1"], [1, OSCURO]],
                                 colorbar=dict(title="tornillos"),
                                 hovertemplate="x=%{x}<br>y=%{y}<br>tornillos=%{z}<extra></extra>"))
    f.update_yaxes(autorange="reversed", scaleanchor="x", title="y (px)")
    f.update_layout(title="Dónde aparecen los tornillos en la imagen", xaxis_title="x (px)")
    return f


def brillo_por_particion(i: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for split in [p for p in ORDEN_PARTICION if p in set(i.particion)]:
        d = i[i.particion == split]
        f.add_box(y=d.brillo, name=split, boxpoints="outliers", marker_color=PARTICION[split],
                  line=dict(color=OSCURO, width=1.5), fillcolor=PARTICION[split])
    f.update_layout(title="Brillo medio de la imagen por partición (cambio de distribución)",
                    yaxis_title="Brillo medio (0–255)", showlegend=False)
    return f


def dispersion(df: pd.DataFrame, x: str, y: str, color: str | None, etiquetas: dict | None = None) -> go.Figure:
    et = etiquetas or {}
    f = go.Figure()
    if color == "particion":
        grupos = [(k, df[df.particion == k]) for k in ORDEN_PARTICION if k in set(df.particion)]
    else:
        grupos = df.groupby(color) if color else [("todas", df)]
    for k, d in grupos:
        f.add_scattergl(x=d[x], y=d[y], mode="markers", name=str(k),
                        marker=dict(size=6, opacity=0.65, color=PARTICION.get(k) if color == "particion" else None))
    f.update_layout(title=f"{et.get(y, y)} frente a {et.get(x, x)}", xaxis_title=et.get(x, x),
                    yaxis_title=et.get(y, y))
    return f


# ----------------------------------------------------------------------------------------------- modelos
def diferencia_critica(rangos: pd.Series, cd: float, destacar="sgw_k0.45", nombres=None) -> go.Figure:
    r = rangos.sort_values()
    nombres = nombres or {}
    f = go.Figure()
    y = np.arange(len(r))[::-1]
    colores = [SGW if m == destacar else OSCURO for m in r.index]
    f.add_scatter(x=r.values, y=y, mode="markers+text", marker=dict(color=colores, size=11,
                  symbol=["diamond" if m == destacar else "circle" for m in r.index]),
                  text=[f"  {nombres.get(m, m)} ({v:.2f})" for m, v in r.items()], textposition="middle right",
                  hovertemplate="%{text}<extra></extra>")
    base = r[destacar]
    f.add_shape(type="rect", x0=base - cd, x1=base + cd, y0=-0.5, y1=len(r) - 0.5, fillcolor=SGW, opacity=0.08,
                line_width=0)
    f.add_annotation(x=base + cd, y=len(r) - 0.5, text=f"± CD = {cd:.2f} alrededor de SGW", showarrow=False,
                     xanchor="right", yanchor="bottom", font=dict(color=SGW))
    f.update_yaxes(visible=False)
    f.update_layout(title="Diagrama de diferencia crítica (Friedman + Nemenyi, 14 grupos de test)",
                    xaxis_title="Rango medio (1 = mejor)", xaxis_range=[0.5, len(r) + 4], height=480)
    return f


def pareto(t: pd.DataFrame, x="latencia_GPU_ms") -> go.Figure:
    d = t.dropna(subset=[x])
    f = go.Figure()
    etiquetar = d.rol.isin(["Modelo original", "Baseline fuerte", "Baseline clásico"])
    posicion = {"sgw_k0.45": "top center", "maskrcnn": "bottom left", "yolov8": "bottom center", "solov2": "middle right"}
    f.add_scatter(x=d[x], y=d.F1_test, mode="markers+text", text=d.nombre.where(etiquetar, ""),
                  customdata=d.nombre, textposition=[posicion.get(m, "middle right") for m in d.modelo],
                  error_y=dict(type="data", array=d.F1_test_DE, color=GRIS),
                  marker=dict(size=12, color=[COLOR_MODELO.get(m, OSCURO) for m in d.modelo],
                              symbol=["diamond" if m == "sgw_k0.45" else "circle" for m in d.modelo]),
                  hovertemplate="%{customdata}<br>F1 = %{y:.3f}<br>%{x:.1f}<extra></extra>")
    f.update_xaxes(type="log", range=[np.log10(18), np.log10(330)], tickvals=[20, 30, 50, 100, 200, 300], ticktext=["20", "30", "50", "100", "200", "300"])
    f.update_layout(title="Desempeño frente a costo (frontera de Pareto)", yaxis_title="F1 por grupo en test",
                    xaxis_title="Latencia mediana en GPU, ms por imagen (escala log)")
    return f


def f1_por_semilla(t: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for m, d in t.groupby("modelo"):
        f.add_box(y=d.f1, name=m, boxpoints="all", jitter=0.3, marker_color=COLOR_MODELO.get(m, OSCURO))
    f.update_layout(title="F1 por grupo en test de cada semilla", yaxis_title="F1 por grupo", showlegend=False)
    return f


def curvas(c: pd.DataFrame, y="val_AP") -> go.Figure:
    f = go.Figure()
    for (m, s), d in c.groupby(["modelo", "semilla"]):
        f.add_scatter(x=d.epoca, y=d[y], mode="lines", line=dict(color=COLOR_MODELO.get(m, OSCURO), width=1),
                      opacity=0.5, name=m, legendgroup=m, showlegend=bool(s == c.semilla.min()),
                      hovertemplate=f"{m} s{s}<br>época %{{x}}<br>%{{y:.3f}}<extra></extra>")
    f.update_layout(title=f"Curvas de entrenamiento: {y} por época (10 semillas)", xaxis_title="Época",
                    yaxis_title=y)
    return f


def terciles(t: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    d = t[t.modelo.isin(["sgw_k0.45", "maskrcnn"])]
    for m, g in d.groupby("modelo"):
        largo = g.melt(id_vars=["modelo", "semilla"], value_vars=["pequeño", "mediano", "grande"])
        f.add_box(x=largo.variable, y=largo.value, name=m, marker_color=COLOR_MODELO[m], boxpoints="all")
    f.update_layout(boxmode="group", title="Recall por tercil de área (10 semillas; todos los tornillos)",
                    yaxis_title="Recall", xaxis_title="Tercil de área")
    return f


def robustez(r: pd.DataFrame, perturbacion: str) -> go.Figure:
    d = r[r.perturbacion == perturbacion]
    f = go.Figure()
    for (m, fuente), g in d.groupby(["modelo", "fuente"]):
        g = g.sort_values("nivel")
        destacado = m in COLOR_MODELO and fuente.startswith("E3")
        f.add_scatter(x=g.nivel, y=g.F1_grupo, mode="lines+markers", name=f"{m} ({fuente})",
                      line=dict(color=COLOR_MODELO.get(m, GRIS) if destacado else GRIS,
                                width=3 if destacado else 1, dash="solid" if destacado else "dot"),
                      error_y=dict(type="data", array=g.get("DE_semillas"), visible=destacado))
    f.update_layout(title=f"F1 por grupo bajo {perturbacion}", xaxis_title="Nivel", yaxis_title="F1 por grupo")
    if perturbacion == "escala":
        f.update_xaxes(autorange="reversed", title="Escala de la imagen (1 = original)")
    return f
