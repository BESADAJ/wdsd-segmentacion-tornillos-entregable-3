"""Prueba de humo del despliegue (enunciado del E3, §10.1 paso 6).

Visita cada página de la URL pública, dispara los callbacks de cada página y comprueba lo que necesita la predicción
en el navegador (cabeceras COOP/COEP, onnxruntime-web y los .onnx). Con --vivo (solo modo servidor,
WDSD_INFERENCIA_VIVO=1) lanza además una predicción de ejemplo por modelo en el servidor. Verifica que la respuesta
sea válida y mide los tiempos. Escribe un informe en Markdown para anexarlo al libro (capítulo 9).

La predicción en el navegador en sí se prueba abriendo /modelos > «Sube tu imagen» y subiendo una imagen: el guion
solo usa la biblioteca estándar y no ejecuta JavaScript.

Uso:
    python experiments/prueba_humo_despliegue.py https://TU-SERVICIO.onrender.com
    python experiments/prueba_humo_despliegue.py http://127.0.0.1:8050 --vivo      # modo servidor
    python experiments/prueba_humo_despliegue.py URL --salida results/despliegue/prueba_humo.md

Solo usa la biblioteca estándar (no necesita instalar nada). Sale con código 1 si algo falla.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = ["/", "/eda", "/modelos", "/original", "/ficha"]
ASSETS = ["/assets/style.css", "/assets/e2_mapas_oclusion.jpg", "/assets/prediccion_navegador.js"]
ORT = {"/assets/vendor/ort/ort.wasm.min.js": "javascript", "/assets/vendor/ort/ort-wasm-simd-threaded.mjs": "javascript",
       "/assets/vendor/ort/ort-wasm-simd-threaded.wasm": "application/wasm"}
MODELOS_VIVO = ["sgw_k0.45", "maskrcnn"]


def http(url: str, datos: dict | None = None, timeout: float = 180.0, metodo: str | None = None):
    """Devuelve (estado, cuerpo, segundos, cabeceras)."""
    cuerpo = None if datos is None else json.dumps(datos).encode()
    req = urllib.request.Request(url, data=cuerpo, method=metodo,
                                 headers={"Content-Type": "application/json"} if cuerpo else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            crudo = r.read()
            return r.status, crudo, time.perf_counter() - t0, r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0, e.headers
    except Exception as e:  # sin conexión, tiempo agotado, certificado...
        return 0, str(e).encode(), time.perf_counter() - t0, {}


class App:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.deps = None

    def dependencias(self):
        if self.deps is None:
            s, c, _, _ = http(self.base + "/_dash-dependencies")
            if s != 200:
                raise RuntimeError(f"/_dash-dependencies respondió {s}")
            self.deps = json.loads(c)
        return self.deps

    def callback(self, salida_id: str, entradas: dict, timeout: float = 180.0):
        """Dispara el callback cuya salida incluye `salida_id`. `entradas` = {id_componente: valor}."""
        dep = next(d for d in self.dependencias() if f"{salida_id}." in d["output"])
        salidas = [o for o in dep.get("outputs", [])] if "outputs" in dep else None
        if salidas is None:  # formato antiguo: "..a.b...c.d.."
            partes = [p for p in dep["output"].strip(".").split("...")]
            salidas = [{"id": p.rsplit(".", 1)[0], "property": p.rsplit(".", 1)[1]} for p in partes]
        cuerpo = {
            "output": dep["output"], "outputs": salidas if len(salidas) > 1 else salidas[0],
            "inputs": [{"id": i["id"], "property": i["property"], "value": entradas.get(i["id"])}
                       for i in dep["inputs"]],
            "changedPropIds": [f"{dep['inputs'][0]['id']}.{dep['inputs'][0]['property']}"],
        }
        if dep.get("state"):
            cuerpo["state"] = [{"id": i["id"], "property": i["property"], "value": entradas.get(i["id"])}
                               for i in dep["state"]]
        s, c, dt, _ = http(self.base + "/_dash-update-component", cuerpo, timeout)
        return s, c, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--vivo", action="store_true", help="probar también la inferencia en vivo en el servidor (solo modo servidor, ~0,9 GB)")
    ap.add_argument("--salida", default=None, help="archivo .md donde escribir el informe")
    ap.add_argument("--max-pagina-s", type=float, default=15.0, help="tiempo aceptable por página, en segundos")
    ap.add_argument("--max-vivo-s", type=float, default=30.0, help="tiempo aceptable por predicción, en segundos")
    a = ap.parse_args()

    app = App(a.url)
    filas, fallos = [], 0

    def registrar(nombre, ok, dt, detalle=""):
        nonlocal fallos
        fallos += 0 if ok else 1
        filas.append((nombre, "OK" if ok else "FALLA", f"{dt:.2f}", detalle))
        print(f"{'OK   ' if ok else 'FALLA'} {dt:6.2f} s  {nombre}  {detalle}")

    # 1. páginas y archivos estáticos
    for ruta in PAGINAS + ASSETS:
        s, c, dt, _ = http(a.url.rstrip("/") + ruta)
        lento = ruta in PAGINAS and dt > a.max_pagina_s
        registrar(f"GET {ruta}", s == 200 and not lento, dt, f"HTTP {s}, {len(c) / 1024:.0f} KB" +
                  (" (lento: ¿instancia dormida?)" if lento else ""))
    s, c, dt, _ = http(a.url.rstrip("/") + "/_dash-layout")
    registrar("GET /_dash-layout", s == 200, dt, f"HTTP {s}")

    # 1b. lo que necesita la predicción en el navegador (HEAD: no descarga los 84 MB de cada modelo)
    s, _, dt, h = http(a.url.rstrip("/") + "/", metodo="HEAD")
    coop, coep = h.get("Cross-Origin-Opener-Policy", ""), h.get("Cross-Origin-Embedder-Policy", "")
    registrar("Cabeceras COOP/COEP (hilos de WebAssembly)", coop == "same-origin" and coep in ("credentialless",
              "require-corp"), dt, f"COOP={coop or '—'}, COEP={coep or '—'}")
    for ruta, tipo in ORT.items():
        s, _, dt, h = http(a.url.rstrip("/") + ruta, metodo="HEAD")
        ct = h.get("Content-Type", "")
        registrar(f"HEAD {ruta}", s == 200 and tipo in ct, dt, f"HTTP {s}, {ct}")
    for m in MODELOS_VIVO:
        s, _, dt, h = http(a.url.rstrip("/") + f"/modelo/{m}.onnx", metodo="HEAD")
        mb = int(h.get("Content-Length", 0) or 0) / 2**20
        registrar(f"HEAD /modelo/{m}.onnx", s == 200 and mb > 50, dt, f"HTTP {s}, {mb:.0f} MB")
    s, _, dt, _ = http(a.url.rstrip("/") + "/modelo/no-existe.onnx", metodo="HEAD")
    registrar("Modelo desconocido → 404", s == 404, dt, f"HTTP {s}")

    # 2. callbacks de cada página (valores iniciales de la app)
    try:
        galeria = json.loads((RAIZ / "data" / "galeria.json").read_text(encoding="utf8"))
        casos = [
            ("EDA: filtros globales", "eda-area-hist", {"eda-part": ["train", "val", "test"], "eda-grupo": [],
                                                        "eda-area": [0, 4], "eda-x": "ancho_px", "eda-y": "alto_px",
                                                        "eda-color": "particion"}),
            ("Modelos: galería precalculada", "gal-fig", {"gal-img": galeria[0]["image_id"],
                                                           "gal-capas": ["real", "sgw_k0.45", "maskrcnn"]}),
            ("Modelos: diagnóstico SGW", "diag-f1", {"diag-modelo": "sgw_k0.45"}),
            ("Modelos: diagnóstico Mask R-CNN", "diag-f1", {"diag-modelo": "maskrcnn"}),
            ("Original: curvas de entrenamiento", "cur-fig", {"cur-y": "val_AP"}),
            ("Original: robustez (escala)", "rob-fig", {"rob-tipo": "escala"}),
            ("Original: robustez (ruido)", "rob-fig", {"rob-tipo": "ruido"}),
            ("Original: robustez (desenfoque)", "rob-fig", {"rob-tipo": "desenfoque"}),
        ]
    except Exception as e:
        registrar("Preparar callbacks", False, 0.0, str(e))
        casos = []
    for nombre, salida, entradas in casos:
        try:
            s, c, dt = app.callback(salida, entradas)
            ok = s == 200 and b'"response"' in c
            registrar(f"Callback {nombre}", ok, dt, f"HTTP {s}, {len(c) / 1024:.0f} KB")
        except Exception as e:
            registrar(f"Callback {nombre}", False, 0.0, f"{type(e).__name__}: {e}")

    # 3. entrada inválida: la app debe contestar con un aviso, no caerse
    try:
        s, c, dt = app.callback("vivo-salida", {"vivo-subir": "esto-no-es-una-imagen", "vivo-modelo": "sgw_k0.45"})
        registrar("Entrada inválida → aviso (no se cae)", s == 200 and b"no se pudo leer" in c.lower(), dt,
                  f"HTTP {s}")
    except Exception as e:
        registrar("Entrada inválida → aviso (no se cae)", False, 0.0, f"{type(e).__name__}: {e}")

    # 4. inferencia en vivo (solo si el despliegue la tiene activa)
    if a.vivo:
        img = RAIZ / "data" / "galeria" / Path(galeria[0]["archivo"]).name
        contenido = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
        for m in MODELOS_VIVO:
            try:
                s, c, dt = app.callback("vivo-salida", {"vivo-subir": contenido, "vivo-modelo": m}, 300)
                ok = s == 200 and b"Tornillos detectados" in c and dt <= a.max_vivo_s
                registrar(f"Predicción en vivo: {m}", ok, dt, f"HTTP {s}" + ("" if dt <= a.max_vivo_s else
                          f" (supera {a.max_vivo_s:.0f} s)"))
            except Exception as e:
                registrar(f"Predicción en vivo: {m}", False, 0.0, f"{type(e).__name__}: {e}")
    else:
        print("(inferencia en el servidor no probada: --vivo solo aplica con WDSD_INFERENCIA_VIVO=1)")

    md = [f"# Prueba de humo del despliegue", "", f"- URL: {a.url}", f"- Fecha: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
          f"- Inferencia en el servidor probada (--vivo): {'sí' if a.vivo else 'no'}",
          f"- Resultado: **{'APROBADA' if fallos == 0 else f'{fallos} FALLA(S)'}**", "",
          "| Verificación | Estado | Tiempo (s) | Detalle |", "|---|---|---|---|"]
    md += [f"| {n} | {e} | {t} | {d} |" for n, e, t, d in filas]
    texto = "\n".join(md) + "\n"
    if a.salida:
        out = Path(a.salida)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(texto, encoding="utf8")
        print(f"Informe: {out}")
    print(f"\n{'APROBADA' if fallos == 0 else f'{fallos} FALLA(S)'}")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
