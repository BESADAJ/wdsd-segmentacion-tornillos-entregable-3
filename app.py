"""Dashboard del proyecto WDSD (Entregable 3): punto de entrada. Crea la app multipágina y expone `server` para gunicorn.

Local:      python app.py               (http://127.0.0.1:8050)
Producción: gunicorn app:server --workers 1 --threads 2 --timeout 120
La app solo lee data/, results/ y artifacts/; nunca entrena ni importa training/.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src.data import ACTUALIZADO, VERSION

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Tornillos diminutos: SGW",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 10 * 2**20  # límite de las cargas

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
        dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
        dbc.NavItem(dbc.NavLink("Modelo original", href="/original", active="exact")),
        dbc.NavItem(dbc.NavLink("Ficha del modelo", href="/ficha", active="exact")),
    ],
    brand="WDSD · segmentación de tornillos",
    brand_href="/",
    color="primary",
    dark=True,
    sticky="top",
)

pie = html.Footer(
    dbc.Container([
        html.Hr(),
        html.Small([
            f"Versión {VERSION} · actualizado {ACTUALIZADO} · Ivan Berrio, Jose David Castillo y Jaime Besada · ",
            "Privacidad: las imágenes que subes se procesan en memoria y no se guardan ni se registran. ",
            "Datos: WEEE Disassembly Screw Dataset (VCL, CERTH-ITI), sin licencia declarada: uso académico.",
        ], className="text-muted"),
    ], fluid=True),
    className="mb-3",
)

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="pt-3"), pie])

if __name__ == "__main__":
    app.run(debug=False)
