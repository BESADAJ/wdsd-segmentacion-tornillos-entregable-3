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


def test_conjuntos_extra_coinciden_con_los_resultados():
    """La tabla de conjuntos adicionales del dashboard es copia exacta de results/conjuntos_extra/contrastes.csv."""
    import pandas as pd
    a = pd.read_csv(RAIZ / "data" / "conjuntos_extra.csv")
    b = pd.read_csv(RAIZ / "results" / "conjuntos_extra" / "contrastes.csv")
    pd.testing.assert_frame_equal(a, b)


def test_modelo_onnx_para_el_navegador(app):
    """Modo navegador: el servidor entrega los .onnx exportados (y solo esos) con las cabeceras de aislamiento."""
    from src import data
    exportados = data.modelos_exportados()
    if not exportados:
        pytest.skip("sin artefactos ONNX")
    c = app.server.test_client()
    m = next(iter(exportados))
    r = c.head(f"/modelo/{m}.onnx")
    assert r.status_code == 200 and int(r.headers["Content-Length"]) > 50 * 2**20
    assert c.head("/modelo/no-existe.onnx").status_code == 404
    assert c.get("/modelo/..%2Fapp.onnx").headers["Content-Type"].startswith("text/html")  # no sale de artifacts/
    h = c.get("/").headers
    assert h["Cross-Origin-Opener-Policy"] == "same-origin" and h["Cross-Origin-Embedder-Policy"] == "credentialless"
    for f, tipo in (("ort.wasm.min.js", "javascript"), ("ort-wasm-simd-threaded.mjs", "javascript"),
                    ("ort-wasm-simd-threaded.wasm", "application/wasm")):
        r = c.head(f"/assets/vendor/ort/{f}")
        assert r.status_code == 200 and tipo in r.headers["Content-Type"]


def test_config_navegador(app, monkeypatch):
    from pages import modelos
    from src import data
    assert modelos.config_navegador() == {"modo": "servidor"}
    monkeypatch.setattr(data, "INFERENCIA_VIVO", False)
    with app.server.test_request_context("/modelos"):
        cfg = modelos.config_navegador()
        assert modelos.layout() is not None
    assert cfg["modo"] == "navegador" and set(cfg["modelos"]) == set(data.modelos_exportados())
    for m, meta in data.modelos_exportados().items():
        assert cfg["modelos"][m]["umbral"] == meta["umbral_score"] and cfg["modelos"][m]["url"] == f"/modelo/{m}.onnx"
    assert cfg["radio"] == data.jsn("conforme.json")["radio"]["0.9"]
    # en modo navegador el callback de Python valida y avisa, pero no ejecuta el modelo
    assert modelos.vivo("data:image/png;base64,xxx", "sgw_k0.45").__class__.__name__ == "Alert"


def test_nav_imagen_entrega_los_pixeles_del_servidor(app, monkeypatch):
    """Modo navegador: la imagen que recibe onnxruntime-web es exactamente la que usa el modo servidor."""
    import numpy as np
    from PIL import Image
    from pages import modelos
    from src import data, inference
    ruta = RAIZ / "data" / "galeria" / "14.jpg"
    contenido = "data:image/jpeg;base64," + base64.b64encode(ruta.read_bytes()).decode()
    assert modelos.nav_imagen(contenido).__class__.__name__ == "NoUpdate"  # modo servidor: no hace nada
    monkeypatch.setattr(data, "INFERENCIA_VIVO", False)
    url = modelos.nav_imagen(contenido)
    recibida = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    esperada = np.asarray(inference.redimensionar(inference.decodificar(contenido)))
    assert recibida.shape == (720, 1280, 3) and np.array_equal(recibida, esperada)
    assert modelos.nav_imagen("data:image/png;base64,AAAA") is None
