"""Explicación local de una predicción de segmentación de instancias.

En segmentación la explicación natural es la propia máscara: se muestra cada instancia con su score, y un mapa de
confianza por píxel (máximo, sobre las detecciones, de probabilidad de máscara × score). Grad-CAM no es posible
sobre el grafo ONNX sin gradientes; los mapas de oclusión del E2 (capítulo 5.3) se muestran precalculados.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

NARANJA, VERDE = (230, 126, 34), (24, 188, 156)


def a_base64(im: Image.Image, fmt="JPEG") -> str:
    b = io.BytesIO()
    im.save(b, format=fmt, quality=85)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def superponer(im: Image.Image, r: dict, ancho_max=960) -> Image.Image:
    base = im.resize((1280, 720)).convert("RGBA")
    capa = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dib = ImageDraw.Draw(capa)
    for m, s, c in zip(r["mascaras"], r["scores"], r["cajas"]):
        a = Image.fromarray(((m > 0.5) * 140).astype(np.uint8), "L")
        capa.paste(Image.new("RGBA", base.size, NARANJA + (140,)), (0, 0), a)
        x1, y1, x2, y2 = c
        dib.rectangle([x1, y1, x2, y2], outline=NARANJA + (255,), width=2)
        dib.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill=(255, 255, 255, 255))
    out = Image.alpha_composite(base, capa).convert("RGB")
    if out.width > ancho_max:
        out = out.resize((ancho_max, int(out.height * ancho_max / out.width)))
    return out


def mapa_confianza(r: dict, ancho_max=960) -> Image.Image:
    if len(r["mascaras"]) == 0:
        m = np.zeros((720, 1280), np.float32)
    else:
        m = (r["mascaras"] * r["scores"][:, None, None]).max(0)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    rgb[..., 0] = (m * NARANJA[0]).astype(np.uint8)
    rgb[..., 1] = (m * NARANJA[1]).astype(np.uint8)
    rgb[..., 2] = (m * NARANJA[2]).astype(np.uint8)
    im = Image.fromarray(rgb)
    return im.resize((ancho_max, int(720 * ancho_max / 1280)))
