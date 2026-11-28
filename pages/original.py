"""Subsección del modelo original: procedencia, mecanismo, pseudocódigo, curvas, pruebas, robustez y veredicto."""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import data, plots, ui

dash.register_page(__name__, path="/original", name="Modelo original", title="SGW · modelo original")

PSEUDO = """Entrada: cajas reales G, anclas o propuestas B, kappa = 0,45, piso c = 4 px
para cada g en G y b en B:
    p(g) = (cx, cy, w/2, h/2);  p(b) igual            # gaussiana 2D de cada caja
    W(g, b) = || p(g) - p(b) ||_2                       # Wasserstein-2 entre gaussianas
    D(g)    = kappa * (sqrt(w_g * h_g) + c)             # escala propia del tornillo, con piso
    s(g, b) = exp(-W(g, b) / D(g))                       # similitud en (0, 1]
RPN:  positivas s >= 0,7 ; negativas s < 0,3   (mismos umbrales que con IoU)
RoI:  positivas s >= 0,5 ; el resto, fondo      (box_similarity y assign_targets reemplazados)
El resto de Mask R-CNN (pérdidas, muestreo, cabezas, inferencia) no cambia."""


def tabla(df: pd.DataFrame, cols: dict):
    d = df[list(cols)].rename(columns=cols)
    return dbc.Table.from_dataframe(d.round(3), responsive=True, hover=True, className="tabla-wdsd")


def layout():
    fam = data.confirmatorio("familia_R3.csv")
    hm = data.confirmatorio("hipotesis_mecanismo.csv").iloc[0]
    abl = data.confirmatorio("ablacion.csv")
    pre = data.confirmatorio("predicciones.csv")
    res = data.confirmatorio("resumen_confirmatorio.json")
    nombres = dict(zip(data.tabla_maestra().modelo, data.tabla_maestra().nombre))
    fam = fam.assign(rival=fam.rival.map(nombres))
    return html.Div([
        ui.encabezado("Asignación SGW para tornillos diminutos", etiqueta="Modelo original"),
        html.Div([
            html.Div("Veredicto", className="etiqueta"),
            html.H3("No superioridad"),
            html.P(["SGW tiene el mayor F1 medio (0,738) y el mejor rango de Friedman, pero no supera de forma "
                    "significativa a Mask R-CNN, HTC, Cascade ni MS R-CNN (diferencias de +0,014 a +0,022, IC que "
                    "incluyen 0). Supera a 10 de 14 rivales, y solo a 4 por más de δ = 0,20: R3 no se cumple."]),
            html.P([html.B("Mecanismo: "), "el recall de tornillos pequeños sube de 0,30 a 0,39 contando todos los "
                    "tornillos (9 de 10 semillas), pero la ganancia se concentra en una escena y la prueba "
                    "preregistrada por grupo no la confirma (+0,012)."], className="mb-0"),
        ], className="panel-tinta mt-3"),
        ui.seccion("Procedencia y mecanismo", dbc.Row([
            dbc.Col(ui.tarjeta(html.Ul([
                html.Li(["Fuente: Normalized Wasserstein Distance (NWD) para detección de objetos diminutos en "
                         "imágenes aéreas (Wang, Xu, Yang y Yu, 2021; ISPRS J. 2022). NWD usa una constante "
                         "global C ajustada al conjunto."]),
                html.Li(["Lo propio: la escala es la de cada tornillo, D(g) = κ(√(w·h) + c), con piso c = 4 px "
                         "(el paso de P2) y κ calibrado analíticamente con el IoU en el límite de objetos "
                         "grandes. Se implementa desde cero (src/original_model.py); no se usa código de NWD."]),
                html.Li(["Teorema 1: s ∈ (0, 1]; Lipschitz con cota 1/(√2 κ c) uniforme en el tamaño; "
                         "gradiente no nulo donde IoU = 0; para objetos diminutos se reduce a NWD con C = κc. "
                         "Proposición 2: el IoU tiene sensibilidad 2/a, no acotada cuando a → 0."]),
                html.Li(["Paper guía (Kalitsios et al., 2022, ICRCV): Mask R-CNN ResNet-101 en WDSD con la "
                         "asignación estándar por IoU. SGW conserva esa arquitectura y cambia solo la asignación. "
                         "La limitación verificada del paper (fuga por la partición aleatoria) la corrige la "
                         "partición agrupada; SGW ataca otra, medida en el E2: el recall de los tornillos del "
                         "tercil pequeño no pasa de 0,37 en ningún modelo."]),
            ], className="mb-0 ps-3"), titulo="Procedencia y diferencia"), lg=5),
            dbc.Col(ui.tarjeta(html.Pre(PSEUDO, className="pseudocodigo"), titulo="Pseudocódigo"), lg=7),
        ], className="g-tarjetas")),
        ui.seccion("Curvas de entrenamiento", ui.tarjeta(
            dcc.Graph(id="cur-fig", config={"displaylogo": False}),
            ui.insight(f"💡 Brecha train93 − val: SGW {res['sobreajuste']['brecha_train93_val_sgw']:.3f} frente a "
                       f"{res['sobreajuste']['brecha_train93_val_maskrcnn']:.3f} de Mask R-CNN (p = "
                       f"{res['sobreajuste']['p_t_brecha_train93_val']:.3f}): SGW ajusta más el entrenamiento, pero la "
                       f"brecha val − test es la misma ({res['sobreajuste']['brecha_val_test_sgw']:.3f} y "
                       f"{res['sobreajuste']['brecha_val_test_maskrcnn']:.3f})."),
            controles=html.Div([dbc.Label("Curva"),
                                dcc.Dropdown(id="cur-y", value="val_AP", clearable=False,
                                             options={"val_AP": "AP de máscara en val", "perdida": "Pérdida de "
                                                      "entrenamiento", "val_AP_small": "AP small en val"})]))),
        ui.seccion("Pruebas de hipótesis (familia R3, Holm, 14 contrastes)", ui.tarjeta(
            tabla(fam, {"rival": "Rival", "dif_media": "SGW − rival", "ic95_lo": "IC inf", "ic95_hi": "IC sup",
                        "p_holm_wilcoxon": "p Holm Wilcoxon", "p_holm_bootstrap": "p Holm bootstrap",
                        "p_holm_t": "p t semillas", "cliff_delta": "δ de Cliff", "d_cohen": "d pareado",
                        "decision": "Decisión"}))),
        ui.seccion("Mecanismo, ablación y predicciones", dbc.Row([
            dbc.Col(ui.tarjeta(
                dbc.Table([html.Tbody([
                    html.Tr([html.Th("Recall pequeño, por grupo (preregistrado)"),
                             html.Td(f"+{hm.dif_media_grupos:.3f} [{hm.ic95_lo:.3f}; {hm.ic95_hi:.3f}], "
                                     f"p = {hm.p_decision:.2f}: no se confirma")]),
                    html.Tr([html.Th("Todos los tornillos, por semilla (exploratorio)"),
                             html.Td(f"+{hm.dif_media_semillas:.3f}, {hm.semillas_a_favor}/10 semillas, "
                                     f"p = {hm.p_t_semillas:.4f}")]),
                ])], className="tabla-wdsd tabla-ficha ancha"),
                dcc.Graph(figure=plots.terciles(data.csv("terciles.csv")), config={"displaylogo": False}),
                titulo="Hipótesis del mecanismo"), lg=6),
            dbc.Col(ui.tarjeta(
                tabla(abl, {"variante": "Variante", "dif_media_vs_sgw": "Δ F1", "ic95_lo": "IC inf",
                            "ic95_hi": "IC sup", "p_decision": "p Holm", "recall_pequeno_media": "Recall pequeño"}),
                html.H3("Predicciones secundarias", className="titulo-tarjeta mt-4"),
                tabla(pre, {"contraste": "Predicción", "dif_media_grupos": "Por grupo", "ic95_lo": "IC inf",
                            "ic95_hi": "IC sup", "dif_media_semillas": "Todos los tornillos"}),
                titulo="Ablación (frente a SGW completo)"), lg=6),
        ], className="g-tarjetas")),
        ui.seccion("Conjuntos adicionales: ¿en qué escenarios no supera?", ui.tarjeta(
            tabla(data.csv("conjuntos_extra.csv").assign(conjunto=lambda d: d.conjunto.str.upper()),
                  {"conjunto": "Conjunto", "n_imagenes": "Imágenes", "F1_SGW": "F1 SGW",
                   "F1_MaskRCNN": "F1 Mask R-CNN", "dif": "SGW − Mask R-CNN", "ic95_lo": "IC inf",
                   "ic95_hi": "IC sup", "p_decision": "p Holm"}),
            ui.insight("💡 Tres conjuntos de otros dominios (3 semillas, solo frente a Mask R-CNN). SGW empata: gana "
                       "en media solo en NuInsSeg, y ninguna diferencia es significativa tras Holm. El criterio de "
                       "consistencia (mejor en al menos 2 de 3) no se cumple. Son objetos más grandes que los "
                       "tornillos, que es donde SGW ganaba recall."))),
        ui.seccion("Robustez", ui.tarjeta(
            dcc.Graph(id="rob-fig", config={"displaylogo": False}),
            ui.insight("💡 Líneas gruesas: SGW (naranja) y Mask R-CNN (azul) con 10 semillas en la 3050; punteadas: "
                       "los 13 modelos del E2 (semilla 42 o 5 semillas)."),
            controles=html.Div([dbc.Label("Perturbación"),
                                dcc.Dropdown(id="rob-tipo", value="escala", clearable=False,
                                             options={"escala": "Escala de la imagen (predicción 4)", "ruido": "Ruido "
                                                      "gaussiano", "desenfoque": "Desenfoque gaussiano"})]))),
        html.P(["Ficha completa del modelo: ", dcc.Link("Ficha del modelo", href="/ficha")], className="mt-4"),
    ])


@callback(Output("cur-fig", "figure"), Input("cur-y", "value"))
def curvas(y):
    return plots.curvas(data.csv("curvas.csv"), y)


@callback(Output("rob-fig", "figure"), Input("rob-tipo", "value"))
def robustez(t):
    r = data.csv("robustez.csv")
    if r[r.perturbacion == t].empty:
        return plots.vacia("Resultados de robustez pendientes")
    return plots.robustez(r, t)
