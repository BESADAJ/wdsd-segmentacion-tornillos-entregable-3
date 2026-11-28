"""Genera assets/arquitectura_sgw.svg: la red de Mask R-CNN con la configuración del repo y dónde interviene SGW.

La configuración sale de src/original_model.py:mask_rcnn_base (ResNet-50-FPN preentrenado en COCO, anclas de 8 a
128 px con aspectos 0,5/1/2, min_size 720, max_size 1280) y de la asignación SGW (umbrales 0,7/0,3 en el RPN y 0,5 en
las cabezas, kappa = 0,45, piso c = 4 px). El ejemplo de la fila inferior se calcula aquí con la misma fórmula que
src/original_model.py:SimilitudSGW. La fórmula tipografiada va en la página (pages/original.py, MathJax de dcc).
Solo biblioteca estándar.

Uso:  python experiments/diagrama_arquitectura.py [assets/arquitectura_sgw.svg]
"""
import math
import sys
from xml.sax.saxutils import escape

TINTA, TINTA2, GRIS_F, GRIS_B, NAR, NAR_F, NAR_T = "#2C3E50", "#5D6D7E", "#F6F8FA", "#C9D1D9", "#E67E22", "#FDF3EA", "#B5560B"
AZUL = "#2980B9"
KAPPA, PISO = 0.45, 4.0
W, H = 1400, 740
FUENTE = "Lato, 'Segoe UI', Arial, sans-serif"
out = []


def coma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def t(x, y, s, size=17, color=TINTA, peso=400, anchor="middle"):
    out.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{color}" font-weight="{peso}" '
               f'text-anchor="{anchor}">{escape(s)}</text>')


def rect(x, y, w, h, relleno, borde, grosor=1.5, rx=8, extra=""):
    out.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{relleno}" stroke="{borde}" '
               f'stroke-width="{grosor}"{extra}/>')


def caja(x, y, w, h, titulo, lineas, naranja=False, tam=18):
    rect(x, y, w, h, NAR_F if naranja else GRIS_F, NAR if naranja else GRIS_B, 2.5 if naranja else 1.5)
    cy = y + 30
    t(x + w / 2, cy, titulo, tam, TINTA, 700)
    for l in lineas:
        cy += 23
        t(x + w / 2, cy, l, 15, TINTA2)


def flecha(x1, y1, x2, y2, color=TINTA, punteada=False):
    d = ' stroke-dasharray="6 5"' if punteada else ""
    m = "fn" if color == NAR else "ft"
    out.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="2"{d} '
               f'marker-end="url(#{m})"/>')


def iou_desplazada(lado, dx):
    inter = (lado - dx) * lado
    return inter / (2 * lado * lado - inter)


def sgw_desplazada(lado, dx):
    # p = (cx, cy, w/2, h/2): misma caja trasladada dx en x -> W2 = dx
    return math.exp(-dx / (KAPPA * (math.sqrt(lado * lado) + PISO)))


# --- fila 1: la red ---
cw, gap, y0, h0 = 170, 35, 70, 130
xs = [i * (cw + gap) for i in range(7)]
t(0, 30, "ARQUITECTURA DE MASK R-CNN, QUE SGW CONSERVA (CONFIGURACIÓN DEL REPOSITORIO)", 14, TINTA2, 700, "start")
caja(xs[0], y0, cw, h0, "Imagen", ["RGB 1280×720", "(lado corto 720)"])
caja(xs[1], y0, cw, h0, "Backbone", ["ResNet-50", "preentrenado", "en COCO"])
caja(xs[2], y0, cw, h0, "FPN", ["mapas P2 a P6", "256 canales"])
caja(xs[3], y0, cw, h0, "RPN", ["anclas 8 a 128 px", "aspectos 0,5 / 1 / 2", "→ propuestas"])
caja(xs[4], y0, cw, h0, "RoIAlign", ["recorta cada", "propuesta en", "P2 a P5"])
caja(xs[5], y0 - 20, cw, 78, "Cabeza de caja", ["clase + ajuste"], tam=16)
caja(xs[5], y0 + 72, cw, 78, "Cabeza de máscara", ["28×28 por objeto"], tam=16)
caja(xs[6], y0, cw, h0, "Salida", ["cajas, scores", "y máscaras", "(score ≥ umbral)"])
for i in range(4):
    flecha(xs[i] + cw + 2, y0 + h0 / 2, xs[i + 1] - 4, y0 + h0 / 2)
flecha(xs[4] + cw + 2, y0 + h0 / 2 - 6, xs[5] - 4, y0 + 19)
flecha(xs[4] + cw + 2, y0 + h0 / 2 + 6, xs[5] - 4, y0 + 111)
flecha(xs[5] + cw + 2, y0 + 19, xs[6] - 4, y0 + h0 / 2 - 6)
flecha(xs[5] + cw + 2, y0 + 111, xs[6] - 4, y0 + h0 / 2 + 6)

# --- fila 2: asignación de etiquetas en el entrenamiento (aporte de SGW) ---
y1, h1, w1 = 300, 150, 300
t(0, 268, "APORTE DE SGW, EN EL ENTRENAMIENTO", 14, NAR_T, 700, "start")


def asignacion(xc, titulo, base, sgw):
    x = xc - w1 / 2
    rect(x, y1, w1, h1, NAR_F, NAR, 2.5)
    t(xc, y1 + 30, titulo, 18, TINTA, 700)
    t(x + 18, y1 + 66, "Mask R-CNN (base):", 15, TINTA2, 700, "start")
    t(x + 18, y1 + 88, base, 15, TINTA2, 400, "start")
    t(x + 18, y1 + 116, "SGW (original):", 15, NAR_T, 700, "start")
    t(x + 18, y1 + 138, sgw, 15, TINTA, 400, "start")


cx_rpn, cx_roi = xs[3] + cw / 2, xs[5] + cw / 2
asignacion(cx_rpn, "Asignación en el RPN", "IoU ≥ 0,7 positiva · < 0,3 negativa", "s ≥ 0,7 positiva · < 0,3 negativa")
asignacion(cx_roi, "Asignación en las cabezas", "IoU ≥ 0,5 positiva (caja y máscara)", "s ≥ 0,5 positiva (caja y máscara)")
flecha(cx_rpn, y1 - 2, cx_rpn, y0 + h0 + 6, NAR, True)
flecha(cx_roi, y1 - 2, cx_roi, y0 + 150 + 6, NAR, True)
# leyenda (arriba a la derecha) y explicación (columna izquierda)
rect(1030, 17, 22, 16, GRIS_F, GRIS_B, 1.5, 3)
t(1060, 30, "red de Mask R-CNN", 15, TINTA2, 400, "start")
rect(1220, 17, 22, 16, NAR_F, NAR, 2.5, 3)
t(1250, 30, "aporte de SGW", 15, TINTA2, 400, "start")
t(0, y1 + 24, "Decide qué anclas y propuestas", 15, TINTA, 700, "start")
t(0, y1 + 46, "cuentan como positivas.", 15, TINTA, 700, "start")
t(0, y1 + 80, "s(g, b): similitud de Wasserstein", 15, NAR_T, 700, "start")
t(0, y1 + 102, "normalizada por el tamaño del tornillo.", 15, TINTA2, 400, "start")
t(0, y1 + 124, "Reemplaza al IoU (fórmula debajo).", 15, TINTA2, 400, "start")
t(0, y1 + 158, "Sin costo extra al predecir: no añade", 15, TINTA, 700, "start")
t(0, y1 + 180, "capas ni parámetros (43,9 M en los dos).", 15, TINTA2, 400, "start")

# --- fila 3: ejemplo numérico ---
y2 = 520
t(0, y2, "POR QUÉ: LA MISMA CAJA DESPLAZADA 4 PX (DIBUJADAS AL MISMO TAMAÑO)", 14, NAR_T, 700, "start")


def ejemplo(x0, lado, titulo, conclusion, color_iou):
    dx, d = 4, 96                      # d: tamaño de dibujo de la caja
    esc = d / lado
    iou, s = iou_desplazada(lado, dx), sgw_desplazada(lado, dx)
    ancho = 670
    rect(x0, y2 + 18, ancho, 172, "#FFFFFF", NAR, 1.5, extra=' stroke-dasharray="6 5"')
    bx, by = x0 + 30, y2 + 70
    rect(bx, by, d, d, "none", TINTA, 2.5, 2)                                     # caja real g
    rect(bx + dx * esc, by, d, d, "none", NAR, 2.5, 2, ' stroke-dasharray="5 4"')  # candidata b
    t(bx + d / 2, by - 12, f"{lado}×{lado} px", 14, TINTA2)
    tx = x0 + 200
    t(tx, y2 + 52, titulo, 17, TINTA, 700, "start")
    t(tx, y2 + 84, f"IoU = {coma(iou)}", 17, color_iou, 700, "start")
    t(tx + 150, y2 + 84, f"s = {coma(s)}", 17, NAR_T, 700, "start")
    for i, l in enumerate(conclusion):
        t(tx, y2 + 116 + 24 * i, l, 15, TINTA2, 400, "start")
    return iou, s


ejemplo(0, 10, "Tornillo típico de WDSD (≈ 10 px)",
        ["El desplazamiento es el 40 % del lado. En las cabezas",
         "(umbral 0,5) el IoU la descarta y SGW la toma como",
         "positiva: más propuestas positivas por tornillo."], AZUL)
ejemplo(W - 670, 100, "Objeto grande (100 px)",
        ["El desplazamiento es el 4 % del lado: los dos criterios",
         "casi coinciden. κ = 0,45 se calibró para eso, así los",
         "umbrales de Mask R-CNN se conservan."], AZUL)
# leyenda del ejemplo
ly = y2 + 212
rect(0, ly - 12, 22, 14, "none", TINTA, 2.5, 2)
t(30, ly, "caja real g", 14, TINTA2, 400, "start")
rect(140, ly - 12, 22, 14, "none", NAR, 2.5, 2, ' stroke-dasharray="5 4"')
t(170, ly, "ancla o propuesta b", 14, TINTA2, 400, "start")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10 0 {W + 20} {H}" font-family="{FUENTE}" role="img" '
       f'aria-labelledby="titulo desc">'
       '<title id="titulo">Arquitectura de Mask R-CNN y dónde interviene SGW</title>'
       '<desc id="desc">Arriba, la red de Mask R-CNN: imagen, backbone ResNet-50, FPN, RPN, RoIAlign, cabezas de caja y '
       'de máscara y salida. Abajo, el aporte de SGW durante el entrenamiento: en el RPN y en las cabezas decide qué '
       'anclas y propuestas son positivas con la similitud s de Wasserstein normalizada por el tamaño del tornillo, '
       f'con los umbrales de Mask R-CNN. Ejemplo: una caja de 10 px desplazada 4 px tiene IoU '
       f'{coma(iou_desplazada(10, 4))} y s {coma(sgw_desplazada(10, 4))}; una de 100 px, IoU '
       f'{coma(iou_desplazada(100, 4))} y s {coma(sgw_desplazada(100, 4))}.</desc>'
       f'<defs><marker id="ft" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{TINTA}"/></marker>'
       f'<marker id="fn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{NAR}"/></marker></defs>'
       '<rect x="-10" y="0" width="100%" height="100%" fill="#FFFFFF"/>' + "".join(out) + "</svg>\n")
destino = sys.argv[1] if len(sys.argv) > 1 else "assets/arquitectura_sgw.svg"
open(destino, "w", encoding="utf8").write(svg)
print(len(svg), "bytes", [coma(f(l, 4)) for l in (10, 100) for f in (iou_desplazada, sgw_desplazada)])
