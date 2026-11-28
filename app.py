"""Dashboard del proyecto WDSD (Entregable 3): punto de entrada. Crea la app multipágina y expone `server` para gunicorn.

Local:      python app.py               (http://127.0.0.1:8050)
Producción: gunicorn app:server --workers 1 --threads 2 --timeout 180
La app solo lee data/, results/ y artifacts/; nunca entrena ni importa training/.
"""
import mimetypes

import dash
import dash_bootstrap_components as dbc
from dash import html
from flask import abort, send_from_directory

from src import data
from src.data import ACTUALIZADO, VERSION

# python:3.11-slim no trae /etc/mime.types: sin esto el navegador rechaza el módulo .mjs y el .wasm de onnxruntime-web
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Tornillos diminutos: SGW",
    suppress_callback_exceptions=True,
    assets_ignore=r"^ort[.-]",  # onnxruntime-web (assets/vendor/ort/) se sirve, pero solo se carga al predecir
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 10 * 2**20  # límite de las cargas


@server.after_request
def aislamiento(resp):
    """Aislamiento de origen cruzado: habilita los hilos de WebAssembly (SharedArrayBuffer) para la predicción en el
    navegador; con 4 hilos tarda ~3 veces menos. `credentialless` deja cargar el CSS y las fuentes de los CDN."""
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return resp


@server.route("/modelo/<nombre>.onnx")
def modelo_onnx(nombre):
    """Entrega el artefacto ONNX para la predicción en el navegador. Solo nombres de artifacts/*.metadata.json."""
    meta = data.modelos_exportados().get(nombre)
    if meta is None:
        abort(404)
    return send_from_directory(data.ARTIFACTS, meta["archivo"], mimetype="application/octet-stream", max_age=86400)


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
        dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
        dbc.NavItem(dbc.NavLink("Modelo original", href="/original", active="exact")),
        dbc.NavItem(dbc.NavLink("Ficha del modelo", href="/ficha", active="exact")),
    ],
    brand="WDSD · Segmentación de tornillos",
    brand_href="/",
    color="white",
    dark=False,
    fluid=True,
    expand="lg",  # en tablet, menú plegable: los cinco enlaces no caben en una línea
    sticky="top",
    className="wdsd-nav",
)

pie = html.Footer(
    html.Div([
        f"Versión {VERSION} · actualizado {ACTUALIZADO} · Ivan Berrio, Jose David Castillo y Jaime Besada · ",
        "Privacidad: las imágenes que subes se procesan en memoria y no se guardan ni se registran. ",
        "Datos: WEEE Disassembly Screw Dataset (VCL, CERTH-ITI), sin licencia declarada: uso académico.",
    ], className="reticula"),
    className="pie",
)

# Una sola retícula (assets/style.css: .reticula) para la barra, el contenido y el pie.
app.layout = html.Div([navbar, html.Main(dash.page_container, className="reticula"), pie])

if __name__ == "__main__":
    app.run(debug=False)
