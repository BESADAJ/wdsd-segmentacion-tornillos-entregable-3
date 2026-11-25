"""Sección 2: EDA con filtros globales (partición, grupo, área) que actualizan todos los gráficos."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from src import data, plots

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · WDSD")

VARIABLES = {"area_px": "Área (px)", "ancho_px": "Ancho de la caja (px)", "alto_px": "Alto de la caja (px)",
             "cx": "Centro x (px)", "cy": "Centro y (px)"}


def insight(texto):
    return html.P(["💡 ", texto], className="small text-muted mb-4")


def layout():
    a = data.eda_anotaciones()
    grupos = sorted(a.grupo.unique())
    lmax = float(np.ceil(np.log10(a.area_px.max())))
    return html.Div([
        html.H2("Exploración de los datos"),
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([dbc.Label("Partición"), dcc.Dropdown(id="eda-part", options=["train", "val", "test"],
                                                          value=["train", "val", "test"], multi=True)], md=4),
            dbc.Col([dbc.Label("Grupos de escena (vacío = todos)"),
                     dcc.Dropdown(id="eda-grupo", options=[int(g) for g in grupos], multi=True,
                                  placeholder="Todos")], md=4),
            dbc.Col([dbc.Label("Área del tornillo (log10 px)"),
                     dcc.RangeSlider(id="eda-area", min=0, max=lmax, step=0.1, value=[0, lmax],
                                     marks={i: f"{10 ** i:g}" for i in range(int(lmax) + 1)})], md=4),
        ])), className="mb-3 sticky-filtros"),
        html.Div(id="eda-resumen", className="mb-2"),
        dbc.Row([
            dbc.Col([dcc.Graph(id="eda-area-hist"), insight(
                "Casi todos los tornillos miden menos de 32² px (COCO small) y un tercio menos de 75 px: las anclas "
                "por defecto (32–512 px) y la asignación por IoU no están pensadas para este tamaño. Motiva SGW.")],
                lg=6),
            dbc.Col([dcc.Graph(id="eda-conteo"), insight(
                "Clase única: todas las imágenes tienen al menos un tornillo (mediana 4, máximo 17), pero ocupan "
                "una fracción ínfima de los píxeles. Por eso la métrica primaria es F1 de instancia y no exactitud "
                "de píxel, y el tope de 100 detecciones por imagen nunca recorta en WDSD.")], lg=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(id="eda-mapa"), insight(
                "Panel del tipo de problema (densidad espacial): los tornillos no están uniformes en la imagen, "
                "así que el volteo horizontal es la única aumentación geométrica segura.")], lg=6),
            dbc.Col([dcc.Graph(id="eda-brillo"), insight(
                "Cambio de distribución entre particiones: el brillo medio es casi igual en train, val y test "
                "(mediana 112 en las tres), y la validación adversarial no separa train de test (AUC 0,33). "
                "El cambio entre escenas está en el contenido (aparatos y tamaños), no en la iluminación global.")],
                lg=6),
        ]),
        html.H4("Explorador"),
        dbc.Row([
            dbc.Col([dbc.Label("Eje x"), dcc.Dropdown(id="eda-x", options=VARIABLES, value="ancho_px")], md=4),
            dbc.Col([dbc.Label("Eje y"), dcc.Dropdown(id="eda-y", options=VARIABLES, value="alto_px")], md=4),
            dbc.Col([dbc.Label("Color"), dcc.Dropdown(id="eda-color", options={"particion": "Partición",
                                                                                "grupo": "Grupo"},
                                                      value="particion", clearable=True)], md=4),
        ]),
        dcc.Graph(id="eda-explorador"),
        insight("Elige dos variables. Ancho frente a alto: la relación de aspecto mediana es 1,13 (10 % a 90 %: "
                "0,89–1,50), así que las anclas de aspecto 0,5, 1 y 2 la cubren. Colorea por grupo para ver que "
                "los tornillos más pequeños se concentran en pocas escenas."),
        html.H4("Calidad del conjunto"),
        dbc.Table([html.Tbody([html.Tr([html.Th(k), html.Td(v)]) for k, v in [
            ("Duplicados exactos entre particiones", "0"),
            ("Grupos de escena compartidos entre particiones", "0 (partición agrupada)"),
            ("Casi gemelos en test (Hamming ≤ 10 sobre hash perceptual)", "12 % agrupada · 58 % oficial"),
            ("Validación adversarial train frente a test (AUC)", "0,33 (nulo p95 = 0,55): sin señal de fuga"),
            ("Ruido de etiquetas", "Polígonos COCO; la máscara PNG oficial es un 16 % mayor: se usa el polígono"),
        ]])], bordered=True, size="sm", responsive=True),
    ])


@callback(Output("eda-area-hist", "figure"), Output("eda-conteo", "figure"), Output("eda-mapa", "figure"),
          Output("eda-brillo", "figure"), Output("eda-explorador", "figure"), Output("eda-resumen", "children"),
          Input("eda-part", "value"), Input("eda-grupo", "value"), Input("eda-area", "value"),
          Input("eda-x", "value"), Input("eda-y", "value"), Input("eda-color", "value"))
def actualizar(part, grupos, rango, x, y, color):
    a, i = data.eda_anotaciones(), data.eda_imagenes()
    part = part or ["train", "val", "test"]
    a, i = a[a.particion.isin(part)], i[i.particion.isin(part)]
    if grupos:
        a, i = a[a.grupo.isin(grupos)], i[i.grupo.isin(grupos)]
    lo, hi = 10 ** rango[0], 10 ** rango[1]
    a = a[(a.area_px >= lo) & (a.area_px <= hi)]
    if a.empty:
        v = plots.vacia("No hay tornillos con estos filtros")
        return v, v, v, v, v, dbc.Alert("Los filtros no dejan ningún tornillo.", color="warning")
    resumen = dbc.Badge(f"{len(a):,} tornillos · {len(i):,} imágenes · {a.grupo.nunique()} grupos · "
                        f"área mediana {a.area_px.median():.0f} px".replace(",", "."), color="secondary")
    return (plots.histograma_area(a), plots.histograma_conteo(i), plots.mapa_centros(a),
            plots.brillo_por_particion(i), plots.dispersion(a, x or "ancho_px", y or "alto_px", color), resumen)
