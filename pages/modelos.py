"""Sección 3: tabla de líderes, predicción en vivo, diagnósticos y comparación estadística."""
import base64

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import (ClientsideFunction, Input, Output, State, callback, clientside_callback, dash_table, dcc, html,
                  no_update)

from src import data, explain, inference, plots, ui

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · WDSD")

COLUMNAS = [("nombre", "Modelo"), ("rol", "Rol"), ("familia", "Familia"), ("F1_test", "F1 test"),
            ("F1_test_DE", "DE"), ("F1_val", "F1 val"), ("semillas", "Semillas"), ("ECE", "ECE"),
            ("parametros_M", "Parámetros (M)"), ("GFLOPs", "GFLOPs"), ("latencia_GPU_ms", "Latencia GPU (ms)"),
            ("latencia_CPU_ms", "Latencia CPU (ms)"), ("rango_Friedman", "Rango Friedman"),
            ("decision_vs_SGW", "Frente a SGW")]


def lider():
    t = data.tabla_maestra().copy()
    t["parametros_M"] = t.parametros / 1e6
    if "latencia_CPU_ms" not in t:
        t["latencia_CPU_ms"] = None
    cols = []
    una_cifra = {"parametros_M", "GFLOPs", "latencia_GPU_ms", "latencia_CPU_ms"}  # magnitudes: 1 decimal basta
    for c, n in COLUMNAS:
        col = {"name": n, "id": c}
        if t[c].dtype.kind == "f":
            col.update(type="numeric", format=dash_table.Format.Format(
                precision=1 if c in una_cifra else 3, scheme=dash_table.Format.Scheme.fixed))
        cols.append(col)
    return dash_table.DataTable(
        data=t[[c for c, _ in COLUMNAS]].to_dict("records"), columns=cols, sort_action="native",
        filter_action="native", page_size=16, style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Lato, Segoe UI, sans-serif", "fontSize": 14, "padding": "8px 7px",
                    "whiteSpace": "normal", "color": plots.OSCURO, "border": "none",
                    "borderBottom": "1px solid #E3E8ED", "textAlign": "right", "minWidth": "52px"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                for c in ("nombre", "rol", "familia", "decision_vs_SGW")] +
                               [{"if": {"column_id": "nombre"}, "fontWeight": "bold", "minWidth": "120px"},
                                {"if": {"column_id": "decision_vs_SGW"}, "minWidth": "150px", "maxWidth": "170px"},
                                {"if": {"column_id": "familia"}, "maxWidth": "120px"},
                                {"if": {"column_id": "rol"}, "maxWidth": "90px"}],
        style_header={"backgroundColor": "#F6F8FA", "color": plots.OSCURO, "fontWeight": "bold",
                      "borderBottom": "2px solid " + plots.OSCURO, "whiteSpace": "normal"},
        style_filter={"backgroundColor": "#FFFFFF", "borderBottom": "1px solid #E3E8ED"},
        style_data_conditional=[{"if": {"filter_query": '{rol} = "Modelo original"'},
                                 "backgroundColor": "#FDF3EA", "fontWeight": "bold"},
                                {"if": {"filter_query": '{rol} = "Modelo original"', "column_id": "nombre"},
                                 "borderLeft": "4px solid " + plots.SGW}])


def opciones_modelos():
    t = data.tabla_maestra()
    return [{"label": n, "value": m} for m, n in zip(t.modelo, t.nombre)]


def config_navegador() -> dict:
    """Configuración de assets/prediccion_navegador.js. En modo servidor la función de JS no hace nada."""
    if data.INFERENCIA_VIVO:
        return {"modo": "servidor"}
    conf = data.jsn("conforme.json")
    nombres = data.tabla_maestra().set_index("modelo").nombre
    return {
        "modo": "navegador",
        "ort_js": dash.get_asset_url("vendor/ort/ort.wasm.min.js"),
        "ort_dir": dash.get_asset_url("vendor/ort/"),
        "radio": conf["radio"]["0.9"],
        "cobertura": conf["cobertura_test"]["0.9"],
        "modelos": {m: {"url": dash.get_relative_path(f"/modelo/{m}.onnx"), "umbral": meta["umbral_score"],
                        "nombre": nombres.get(m, m),
                        "mb": round((data.ARTIFACTS / meta["archivo"]).stat().st_size / 2**20)}
                    for m, meta in data.modelos_exportados().items()},
    }


def layout():
    onnx = data.modelos_onnx()
    exportados = onnx or data.modelos_exportados()  # en modo navegador se predice con todos los exportados
    mb = round(sum((data.ARTIFACTS / m["archivo"]).stat().st_size for m in exportados.values()) / 2**20 /
               max(1, len(exportados)))
    galeria = data.jsn("galeria.json")
    conf = data.confirmatorio("friedman.json")
    rangos = data.confirmatorio("friedman_rangos.csv").set_index(data.confirmatorio("friedman_rangos.csv").columns[0])
    nombres = dict(zip(data.tabla_maestra().modelo, data.tabla_maestra().nombre))
    nota_navegador = html.P([
        html.B("La predicción corre en tu navegador. "),
        f"En el servidor, una predicción a 1280×720 sube la memoria a unos {data.PICO_MEMORIA_MB} MB y el plan "
        f"gratuito de Render tiene 512 MB. Por eso este despliegue te envía el mismo modelo ONNX ({mb} MB, se "
        "descarga una vez) y lo ejecuta en tu equipo con onnxruntime-web. Necesita un computador con Chrome, Edge o "
        "Firefox y ~1,3 GB de memoria libre; tarda entre 5 y 20 s. La imagen no sale de tu equipo salvo para la "
        "validación y el aviso fuera de dominio, que no la guardan."], className="nota panel mt-3 mb-0 py-2")
    return html.Div([
        ui.encabezado("Modelos", "Tabla de líderes, predicción en vivo con su explicación, diagnóstico del modelo "
                      "elegido y comparación estadística.", etiqueta="Modelos"),
        ui.seccion("Tabla de líderes", ui.tarjeta(
            lider(),
            html.P("Latencia GPU: RTX 3060, lote 1, mediana (E2). SGW es la misma red que Mask R-CNN en inferencia, "
                   "así que comparte parámetros, FLOPs y latencia. Latencia CPU: ONNX Runtime FP16, solo para los "
                   "modelos exportados. ECE: calibración del score de detección (E2).", className="nota mt-3 mb-0")),
            descripcion="Top 10 de la revisión sistemática, los baselines fuertes (Mask R-CNN, YOLOv8, SOLOv2), el "
                        "clásico y el modelo original. F1 de instancia por grupo en test (media y desviación entre "
                        "semillas). Ordena con las flechas y filtra escribiendo bajo cada encabezado (por ejemplo "
                        "«> 0.7»)."),

        ui.seccion("Predicción", ui.tarjeta(dbc.Tabs([
            dbc.Tab(label="Predicción en vivo: sube tu imagen", tab_id="vivo", children=[
                dbc.Row([
                    dbc.Col([html.Div("1 · Modelo", className="paso-pred"),
                             dcc.Dropdown(id="vivo-modelo", options=[{"label": data.tabla_maestra().set_index(
                                 "modelo").nombre.get(m, m), "value": m} for m in exportados],
                                 value="sgw_k0.45" if "sgw_k0.45" in exportados else next(iter(exportados), None),
                                 clearable=False, disabled=not exportados)], md=4),
                    dbc.Col([html.Div("2 · Imagen", className="paso-pred"),
                             dcc.Upload(id="vivo-subir", disabled=not exportados, max_size=inference.MAX_BYTES,
                                        accept="image/png,image/jpeg,image/bmp,image/tiff",
                                        children=html.Div([html.B("Arrastra una imagen aquí"), " o ", html.A("elígela"),
                                                           html.Span(" · PNG o JPEG, máx. 8 MB", className="nota")]),
                                        className="zona-subida")], md=8),
                ], className="g-3 mt-2"),
                nota_navegador if not onnx and exportados else html.Div(),
                dcc.Loading(html.Div(id="vivo-salida", className="mt-3")),
                html.Div(id="nav-resultado"),
                dcc.Store(id="nav-config", data=config_navegador()), dcc.Store(id="nav-imagen"),
            ]),
            dbc.Tab(label="Galería precalculada", tab_id="galeria", children=[
                html.Div([
                    html.Div([dbc.Label("Imagen de test"),
                              dcc.Dropdown(id="gal-img", value=galeria[0]["image_id"], clearable=False,
                                           options=[{"label": f"Imagen {g['image_id']} (grupo {g['grupo']})",
                                                     "value": g["image_id"]} for g in galeria])]),
                    html.Div([dbc.Label("Mostrar"),
                              dbc.Checklist(id="gal-capas", value=["real", "sgw_k0.45"], inline=True,
                                            options=[{"label": "Anotación real", "value": "real"},
                                                     {"label": "SGW", "value": "sgw_k0.45"},
                                                     {"label": "Mask R-CNN", "value": "maskrcnn"}])],
                             style={"maxWidth": "none"}),
                ], className="controles mt-3"),
                dcc.Graph(id="gal-fig", config={"displaylogo": False}), html.Div(id="gal-texto"),
            ]),
        ], active_tab="vivo"), clase="prediccion"),
            descripcion="Elige el modelo y sube una imagen: arriba aparece el conteo, y debajo la máscara predicha "
                        "y su explicación local. La galería muestra predicciones precalculadas sobre imágenes de test."),

        ui.seccion("Diagnóstico del modelo seleccionado", ui.tarjeta(
            dbc.Row([dbc.Col(dcc.Graph(id="diag-f1", config={"displaylogo": False}), lg=6),
                     dbc.Col(dcc.Graph(id="diag-dice", config={"displaylogo": False}), lg=6)]),
            ui.insight("💡 En segmentación de instancias el diagnóstico equivalente a la matriz de confusión es el "
                       "conteo de aciertos (TP), falsas detecciones (FP) y tornillos perdidos (FN) por imagen. Los "
                       "puntos por debajo de la diagonal del Dice semántico frente al F1 de instancia son imágenes "
                       "donde el modelo pinta los píxeles correctos pero fusiona o parte tornillos."),
            controles=html.Div([dbc.Label("Modelo"),
                                dcc.Dropdown(id="diag-modelo", options=opciones_modelos(), value="sgw_k0.45",
                                             clearable=False)]))),

        ui.seccion("Comparación estadística", dbc.Row([
            dbc.Col(ui.tarjeta(
                dcc.Graph(figure=plots.diferencia_critica(rangos.iloc[:, 0], conf["CD"], nombres=nombres),
                          config={"displaylogo": False}),
                html.P(f"Friedman χ² = {conf['chi2']:.1f} (p = {conf['p_chi2']:.1e}). SGW tiene el mejor rango medio "
                       f"({conf['rango_sgw']:.2f}), pero dentro de la diferencia crítica ({conf['CD']:.2f}) de los "
                       "modelos de dos etapas: no hay superioridad significativa.", className="nota mt-2 mb-0")),
                lg=6),
            dbc.Col(ui.tarjeta(
                dcc.Graph(figure=plots.pareto(data.tabla_maestra()), config={"displaylogo": False}),
                ui.insight("💡 Mask R-CNN y SGW están en la frontera: los YOLO son 3 veces más rápidos, pero pierden "
                           "entre 0,1 y 0,3 de F1.")), lg=6),
        ], className="g-tarjetas")),

        ui.seccion("Interpretabilidad global (E2)", ui.tarjeta(
            html.Img(src=dash.get_asset_url("e2_mapas_oclusion.jpg"), className="img-fluid rounded",
                     alt="Mapas de oclusión de Mask R-CNN, HTC y Cascade sobre imágenes de test (E2)")),
            descripcion="Mapas de oclusión de los tres mejores modelos del E2 (capítulo 5.3 del informe del E2), con "
                        "fidelidad (curvas de borrado) y estabilidad entre semillas. SGW comparte la arquitectura de "
                        "Mask R-CNN; su diferencia está en qué propuestas se entrenan como positivas, que se compara "
                        "en la página del modelo original."),
    ])


@callback(Output("gal-fig", "figure"), Output("gal-texto", "children"), Input("gal-img", "value"),
          Input("gal-capas", "value"))
def galeria(iid, capas):
    g = next(x for x in data.jsn("galeria.json") if x["image_id"] == iid)
    ruta = data.DATA / g["archivo"]
    src = "data:image/jpeg;base64," + base64.b64encode(ruta.read_bytes()).decode()
    f = go.Figure()
    f.add_layout_image(source=src, x=0, y=0, sizex=g["ancho"], sizey=g["alto"], xref="x", yref="y",
                       layer="below", sizing="stretch")
    estilos = {"real": (plots.REAL, "solid", "Anotación real"), "sgw_k0.45": (plots.SGW, "solid", "SGW"),
               "maskrcnn": (plots.MRCNN, "dot", "Mask R-CNN")}
    for capa in capas or []:
        color, dash_, nombre = estilos[capa]
        polis = g["real"] if capa == "real" else [c for d in g["predicciones"][capa]["detecciones"]
                                                  for c in d["contornos"]]
        xs, ys = [], []
        for p in polis:
            xs += [q[0] for q in p] + [p[0][0], None]
            ys += [q[1] for q in p] + [p[0][1], None]
        f.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2, dash=dash_), name=nombre)
    f.update_xaxes(visible=False, range=[0, g["ancho"]])
    f.update_yaxes(visible=False, range=[g["alto"], 0], scaleanchor="x")
    f.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"),
                    title=f"Imagen {iid} · grupo {g['grupo']} · {len(g['real'])} tornillos reales")
    filas = [html.Tr([html.Th("Modelo"), html.Th("Semilla"), html.Th("F1"), html.Th("TP"), html.Th("FP"),
                      html.Th("FN")])]
    for m, p in g["predicciones"].items():
        filas.append(html.Tr([html.Td(estilos[m][2]), html.Td(p["semilla"]), html.Td(f"{p['f1']:.2f}"),
                              html.Td(p["tp"]), html.Td(p["fp"]), html.Td(p["fn"])]))
    return f, dbc.Table(filas, size="sm", bordered=True, className="w-auto")


@callback(Output("vivo-salida", "children"), Input("vivo-subir", "contents"), State("vivo-modelo", "value"),
          prevent_initial_call=True)
def vivo(contenido, modelo):
    if not contenido or not modelo:
        return no_update
    try:
        im = inference.decodificar(contenido)
    except inference.EntradaInvalida as e:
        return dbc.Alert(str(e), color="danger")
    avisos = []
    od = inference.fuera_de_dominio(im)
    if od["fuera"] or od["aspecto_distinto"]:
        avisos.append(dbc.Alert([
            html.B("Entrada fuera de dominio. "),
            f"La imagen se aleja de las de entrenamiento (distancia {od['distancia']:.1f}, umbral p99 "
            f"{od['umbral']:.1f})" + ("; además su proporción no es 16:9 y se deforma al llevarla a 1280×720"
                                     if od["aspecto_distinto"] else "") +
            ". No interpretes la predicción como confiable."], color="warning"))
    if not data.INFERENCIA_VIVO:  # modo navegador: el modelo lo ejecuta assets/prediccion_navegador.js
        return html.Div(avisos)
    try:
        r = inference.predecir(modelo, im)
    except Exception:  # la app no debe caerse por una entrada rara
        return dbc.Alert("No se pudo procesar la imagen con el modelo.", color="danger")
    lo, hi = r["intervalo_90"]
    return html.Div(avisos + [
        html.Div([
            html.Div([str(r["n"]), html.Small("tornillos detectados")], className="resultado-n"),
            html.Span(["Intervalo conforme 90 %: ", html.B(f"{lo} a {hi}")], className="resultado-dato"),
            html.Span(["Modelo: ", html.B(data.tabla_maestra().set_index("modelo").nombre.get(modelo, modelo))],
                      className="resultado-dato"),
            html.Span(["Latencia: ", html.B(f"{r['latencia_ms']:.0f} ms")], className="resultado-dato"),
        ], className="resultado-cabeza"),
        dbc.Row([
            dbc.Col([html.Div("Máscaras predichas", className="figura-titulo"),
                     html.Img(src=explain.a_base64(explain.superponer(im, r)), className="figura-pred",
                              alt="Imagen con las máscaras de tornillo predichas en naranja")], lg=8),
            dbc.Col([html.Div("Explicación local: mapa de confianza por píxel", className="figura-titulo"),
                     html.Img(src=explain.a_base64(explain.mapa_confianza(r)), className="figura-pred",
                              alt="Mapa de confianza: probabilidad de máscara por score de cada detección"),
                     dbc.Table([html.Tbody([
                         html.Tr([html.Th("Tornillos detectados"), html.Td(r["n"])]),
                         html.Tr([html.Th("Intervalo conforme 90 % del número de tornillos"),
                                  html.Td(f"{lo} a {hi} (cobertura empírica en test: {r['cobertura_test']:.0%})")]),
                         html.Tr([html.Th("Umbral de score (elegido en val)"), html.Td(f"{r['umbral']:.2f}")]),
                         html.Tr([html.Th("Latencia (CPU, ONNX Runtime)"), html.Td(f"{r['latencia_ms']:.0f} ms")]),
                         html.Tr([html.Th("Scores de las detecciones"),
                                  html.Td(", ".join(f"{s:.2f}" for s in sorted(r["scores"], reverse=True)[:15])
                                          or "—")]),
                     ])], className="tabla-wdsd tabla-ficha mt-3")], lg=4),
        ], className="g-4"),
        html.P("El score de detección no está calibrado como probabilidad (ECE del E2); el intervalo del "
               "conteo sí tiene garantía de cobertura bajo intercambiabilidad, que entre escenas no se cumple "
               "del todo.", className="nota mt-3 mb-0"),
    ])


@callback(Output("nav-imagen", "data"), Input("vivo-subir", "contents"), prevent_initial_call=True)
def nav_imagen(contenido):
    """Modo navegador: Python decodifica, valida y redimensiona (igual que en modo servidor) y entrega al navegador la
    imagen de 1280x720 sin pérdida. Así el modelo ve los mismos píxeles en los dos modos. None si la entrada es
    inválida (el motivo lo muestra `vivo`)."""
    if data.INFERENCIA_VIVO or not contenido:
        return no_update
    try:
        return inference.png_entrada(inference.decodificar(contenido))
    except inference.EntradaInvalida:
        return None


# Modo navegador: el resultado lo pinta JS en #nav-resultado. Cambiar de modelo repite la predicción sobre la misma
# imagen sin volver a subirla.
clientside_callback(ClientsideFunction("wdsd", "predecir"), Output("nav-resultado", "children"),
                    Input("nav-imagen", "data"), Input("vivo-modelo", "value"), State("nav-config", "data"),
                    prevent_initial_call=True)


@callback(Output("diag-f1", "figure"), Output("diag-dice", "figure"), Input("diag-modelo", "value"))
def diagnostico(m):
    p = data.csv("por_imagen.csv")
    d = p[p.modelo == m]
    f1 = go.Figure()
    for k, c in (("tp", plots.REAL), ("fp", plots.GRIS), ("fn", plots.OSCURO)):  # el naranja queda para SGW
        f1.add_bar(x=d.image_id.astype(str), y=d[k], name=k.upper(), marker_color=c)
    f1.update_layout(barmode="stack", title="Aciertos, falsas detecciones y pérdidas por imagen (media en semillas)",
                     xaxis_title="Imagen de test", yaxis_title="Tornillos", xaxis=dict(showticklabels=False))
    f2 = go.Figure(go.Scatter(x=d.f1, y=d.dice_semantico, mode="markers", text=d.grupo,
                              marker=dict(color=plots.COLOR_MODELO.get(m, plots.OSCURO), size=7, opacity=0.7),
                              hovertemplate="F1 %{x:.2f}<br>Dice %{y:.2f}<br>grupo %{text}<extra></extra>"))
    f2.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=plots.GRIS), showlegend=False)
    f2.update_layout(title="Dice semántico frente a F1 de instancia, por imagen", xaxis_title="F1 de instancia",
                     yaxis_title="Dice semántico", showlegend=False)
    return f1, f2
