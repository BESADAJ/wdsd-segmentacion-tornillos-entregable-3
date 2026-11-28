"""Piezas de interfaz comunes a todas las páginas: encabezados, secciones, tarjetas de gráfico e interpretaciones.

Solo componentes de Dash y Dash Bootstrap con las clases de assets/style.css; no cargan datos ni figuras.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def encabezado(titulo: str, bajada=None, etiqueta: str | None = None) -> html.Div:
    """Título de página con bajada opcional (máximo ~70 caracteres por línea)."""
    return html.Div([
        html.Div(etiqueta, className="etiqueta") if etiqueta else None,
        html.H1(titulo, className="titulo-pagina"),
        html.P(bajada, className="bajada") if bajada else None,
    ], className="encabezado")


def seccion(titulo: str, *hijos, descripcion=None, id: str | None = None) -> html.Section:
    """Bloque con título de sección y el mismo espacio vertical en todas las páginas."""
    cabeza = [html.H2(titulo, className="titulo-seccion")]
    if descripcion:
        cabeza.append(html.P(descripcion, className="descripcion-seccion"))
    return html.Section(cabeza + list(hijos), className="seccion", **({"id": id} if id else {}))


def insight(texto) -> html.P:
    """Frase de interpretación de un gráfico (va dentro de su tarjeta, debajo del gráfico)."""
    return html.P(texto if isinstance(texto, list) else [texto], className="insight")


def tarjeta(*hijos, controles=None, titulo: str | None = None, clase: str = "") -> dbc.Card:
    """Tarjeta de contenido: controles arriba (si hay), gráfico o tabla, e interpretación abajo."""
    cuerpo = []
    if titulo:
        cuerpo.append(html.H3(titulo, className="titulo-tarjeta"))
    if controles is not None:
        cuerpo.append(html.Div(controles, className="controles"))
    cuerpo += [h for h in hijos if h is not None]
    return dbc.Card(dbc.CardBody(cuerpo), className=f"tarjeta {clase}".strip())


def cifra(valor: str, texto, clase: str = "") -> html.Div:
    """Cifra destacada con su explicación debajo."""
    return html.Div([html.Div(valor, className="cifra-valor"), html.Div(texto, className="cifra-texto")],
                    className=f"cifra {clase}".strip())
