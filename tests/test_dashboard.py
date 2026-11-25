"""Pruebas de la app: todas las páginas se construyen, los callbacks responden y las entradas inválidas no la tumban;
y las cifras del dashboard coinciden con las del análisis confirmatorio y del informe (misma tabla maestra)."""
import base64
import io
import json
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
pytest.importorskip("dash")
pytest.importorskip("dash_bootstrap_components")


@pytest.fixture(scope="module")
def app():
    import app as modulo
    return modulo


def test_paginas_construyen(app):
    from pages import contexto, eda, ficha, modelos, original
    for p in (contexto, eda, modelos, original, ficha):
        assert p.layout() is not None


def test_rutas_responden(app):
    c = app.server.test_client()
    for ruta in ("/", "/eda", "/modelos", "/original", "/ficha"):
        assert c.get(ruta).status_code == 200


def test_callbacks(app):
    from pages import eda, modelos, original
    figs = eda.actualizar(["train", "test"], [], [0, 4], "ancho_px", "alto_px", "particion")
    assert len(figs) == 6
    vacio = eda.actualizar(["test"], [], [4.5, 5], "cx", "cy", None)
    assert type(vacio[-1]).__name__ == "Alert"
    g = json.loads((RAIZ / "data" / "galeria.json").read_text(encoding="utf8"))[0]
    assert modelos.galeria(g["image_id"], ["real", "sgw_k0.45", "maskrcnn"])[0] is not None
    assert len(modelos.diagnostico("maskrcnn")) == 2
    assert original.curvas("val_AP") is not None
    assert original.robustez("ruido") is not None


@pytest.mark.parametrize("contenido", ["data:image/png;base64,AAAA", "sin-coma", "data:x;base64,@@@"])
def test_entrada_invalida_no_tumba_la_app(app, contenido):
    from pages import modelos
    r = modelos.vivo(contenido, "sgw_k0.45")
    assert type(r).__name__ == "Alert"


def test_imagen_demasiado_pequena(app):
    from PIL import Image
    from src import inference
    b = io.BytesIO()
    Image.new("RGB", (20, 20)).save(b, "PNG")
    with pytest.raises(inference.EntradaInvalida):
        inference.decodificar("data:image/png;base64," + base64.b64encode(b.getvalue()).decode())


def test_fuera_de_dominio_detecta_imagen_rara():
    from PIL import Image
    from src import inference
    negra = Image.new("RGB", (1280, 720))
    assert inference.fuera_de_dominio(negra)["fuera"]


def test_consistencia_tabla_maestra_y_confirmatorio():
    import pandas as pd
    t = pd.read_csv(RAIZ / "data" / "tabla_maestra.csv").set_index("modelo")
    p = pd.read_csv(RAIZ / "results" / "confirmatorio" / "tabla_principal.csv").set_index("modelo")
    for m in t.index:
        assert abs(t.loc[m, "F1_test"] - p.loc[m, "F1_grupo_media"]) < 1e-6, m
    fam = pd.read_csv(RAIZ / "results" / "confirmatorio" / "familia_R3.csv").set_index("rival")
    for m in fam.index:
        assert abs(t.loc[m, "dif_SGW_menos_modelo"] - fam.loc[m, "dif_media"]) < 1e-6


def test_consistencia_con_el_libro():
    """Las cifras clave del capítulo 7 coinciden con la tabla maestra (redondeadas a 3 decimales)."""
    import pandas as pd
    t = pd.read_csv(RAIZ / "data" / "tabla_maestra.csv").set_index("modelo")
    libro = (RAIZ / "book" / "07_protocolo_validacion.md").read_text(encoding="utf8")
    for m in ("sgw_k0.45", "maskrcnn", "htc", "cascade", "yolov8"):
        cifra = f"{t.loc[m, 'F1_test']:.3f}".replace(".", ",")
        assert re.search(re.escape(cifra), libro), (m, cifra)
