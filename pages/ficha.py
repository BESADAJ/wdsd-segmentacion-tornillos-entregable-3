"""Ficha del modelo (model card), leída de MODEL_CARD.md para que el repositorio y la app digan lo mismo."""
import dash
from dash import dcc, html

from src import data

dash.register_page(__name__, path="/ficha", name="Ficha del modelo", title="Ficha del modelo · SGW")


def layout():
    return html.Div([dcc.Markdown(data.texto("MODEL_CARD.md"), className="ficha")], style={"maxWidth": "900px"})
