"""Sección 1: contexto, preguntas, ficha de datos, metodología y limitaciones."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import data, ui

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · WDSD")


def pregunta(rq, texto):
    """Separa la pregunta (hasta el «?») de la respuesta para darles jerarquía; el texto no cambia."""
    q, _, r = texto.partition("?")
    return html.Div([html.Div(rq, className="pregunta-id"), html.P(q + "?", className="pregunta-texto"),
                     html.P(r.strip(), className="pregunta-respuesta")], className="pregunta")


def paso(n, titulo, detalle):
    return html.Li([html.Div(n, className="paso-n"), html.Div(titulo, className="paso-titulo"),
                    html.Div(detalle, className="paso-detalle")], className="paso")


def coma(x, d=3, signo=False):
    return (f"{x:+.{d}f}" if signo else f"{x:.{d}f}").replace(".", ",")


def layout():
    t = data.tabla_maestra().set_index("modelo")
    sgw, mr = t.loc["sgw_k0.45"], t.loc["maskrcnn"]
    return html.Div([
        # ---- portada: problema y por qué es difícil
        dbc.Row([
            dbc.Col([
                html.Div("Entregable 3 · Curso de Deep Learning", className="etiqueta"),
                html.H1("Encontrar tornillos diminutos para desensamblar aparatos electrónicos",
                        className="titulo-pagina"),
                html.P([
                    "En 2022 se generaron 62 millones de toneladas de residuos electrónicos y solo el 22,3 % se "
                    "recogió y recicló de forma documentada (", html.Em("Global E-waste Monitor 2024"), ", Baldé et "
                    "al., ITU y UNITAR). Desensamblar con robots exige localizar cada tornillo, y en las imágenes del "
                    "WDSD un tornillo mide en mediana 10,6 px de lado: el 98,8 % es un objeto «pequeño» según COCO."],
                    className="bajada"),
                html.Div([html.B("62 Mt"), html.Span("residuos electrónicos generados en 2022 (GEM 2024)")],
                         className="dato-contexto"),
            ], lg=7),
            dbc.Col(html.Div([
                html.Div("Por qué es difícil", className="etiqueta"),
                html.Div(["10,6", html.Small("px")], className="heroe-valor"),
                html.Div("lado mediano de un tornillo en WDSD, en imágenes de 1280 px de ancho", className="heroe-texto"),
                html.Div([
                    html.Div(html.Div(className="escala-tornillo"), className="escala-barra",
                             role="img", **{"aria-label": "Un tornillo ocupa 10,6 de 1280 píxeles de ancho"}),
                    html.Div([html.Span("10,6 px: un tornillo"), html.Span("1280 px: la imagen")],
                             className="escala-leyenda"),
                ], className="escala"),
            ], className="heroe"), lg=5),
        ], className="portada g-4"),

        # ---- resultado principal
        html.Div([
            html.Div("Resultado principal · F1 de instancia por grupo en test", className="etiqueta"),
            dbc.Row([
                dbc.Col(html.Div([html.Div(coma(sgw.F1_test), className="cifra-valor"),
                                  html.Div("SGW, el modelo original", className="cifra-texto")],
                                 className="cifra cifra-sgw"), lg=3, md=6),
                dbc.Col(html.Div([html.Div(coma(mr.F1_test), className="cifra-valor"),
                                  html.Div("Mask R-CNN, el mejor baseline fuerte", className="cifra-texto")],
                                 className="cifra"), lg=3, md=6),
                dbc.Col(html.Div([
                    html.Div([html.Span(coma(sgw.F1_test - mr.F1_test, signo=True)),
                              html.Span("no significativo", className="veredicto")], className="cifra-valor"),
                    html.Div(f"diferencia de SGW frente a Mask R-CNN (p de Holm = {coma(mr.p_holm_decision, 2)})",
                             className="cifra-texto"),
                ], className="cifra cifra-diferencia"), lg=6, className="col-separada"),
            ], className="g-3 align-items-end"),
        ], className="resultados"),

        # ---- preguntas e hipótesis
        ui.seccion("Preguntas de investigación e hipótesis", dbc.Row([
            dbc.Col(html.Div([
                pregunta("RQ1 · E2", "¿Algún modelo del Top 10 supera al mejor baseline fuerte? "
                         "No: Mask R-CNN no es superado (Holm, p = 1)."),
                pregunta("RQ2 · E2", "¿Las dos etapas superan a la familia de prototipos? Sí, "
                         "+0,18 de F1 (p = 0,0002), sin atribuirlo solo a la arquitectura."),
                pregunta("RQ3 · E2", "¿Cómo cae el recall con el tamaño? En el tercil pequeño "
                         "ningún modelo pasa de 0,37 (YOLOv7): la ceguera ante tornillos diminutos es compartida."),
                pregunta("RQ4 · E2", "¿Hay sobreajuste? La brecha de F1 entre train93 y val "
                         "es de 0,05 a 0,08 en los modelos de dos etapas, con IC que incluyen 0."),
            ], className="preguntas"), lg=7),
            dbc.Col(html.Div([
                html.Div("Hipótesis del mecanismo · modelo original (H-M)", className="etiqueta"),
                html.H3("Asignar las propuestas por tamaño, no por IoU"),
                html.P("Si las propuestas de Mask R-CNN se asignan con una similitud de Wasserstein gaussiana "
                       "normalizada por el tamaño de cada tornillo (SGW), en vez de con IoU, los tornillos diminutos "
                       "reciben tantos positivos como los grandes y el recall del tercil pequeño sube al menos 0,05."),
                html.P(["Mecanismo transferido y adaptado de la Normalized Wasserstein Distance para detección "
                        "aérea de objetos diminutos (Wang, Xu, Yang y Yu, 2021, arXiv:2110.13389; ISPRS 2022). "
                        "La normalización por tamaño con piso y su demostración son propias."],
                       className="nota mb-0"),
            ], className="panel-acento"), lg=5),
        ], className="g-4")),

        ui.seccion("Ficha de datos (Data Card)", dbc.Table([html.Tbody([html.Tr([html.Th(k), html.Td(v)]) for k, v in [
            ("Fuente", "WEEE Disassembly Screw Dataset (WDSD), Visual Computing Lab, CERTH-ITI: "
                       "vcl.iti.gr/dataset/weee-disassembly-screw-dataset"),
            ("Licencia", "No declarada por los autores: uso académico, sin redistribuir las imágenes"),
            ("Observaciones", "945 imágenes RGB de 1280×720 con 4.398 tornillos anotados con polígonos (COCO)"),
            ("Tarea", "Segmentación de instancias de una clase (tornillo)"),
            ("Variable objetivo", "La máscara de cada tornillo; métrica primaria: F1 de instancia por imagen "
                                  "(IoU de máscara ≥ 0,5), promediado por grupo de escena"),
            ("Partición", "Agrupada por escena (sin fuga): 744 / 93 / 108 imágenes; 14 grupos en test"),
            ("Calidad y fuga", "Sin duplicados exactos ni grupos compartidos entre particiones. Casi gemelos "
                               "(hash perceptual): 58 % de test en la partición oficial, 12 % en la agrupada (por "
                               "encima del 5 % tolerado; se reporta)"),
            ("Sesgos", "Pocos tipos de aparato y de iluminación; los tornillos diminutos se concentran en pocas "
                       "escenas (el grupo 6 tiene 59 de los 155 pequeños de test)"),
        ]])], responsive=True, className="tabla-wdsd tabla-ficha")),
        ui.seccion("Metodología", html.Ol([
            paso("1", "EDA", "Tamaños, densidad, escenas y fuga"),
            paso("2", "Revisión sistemática", "PRISMA, 64 estudios, Top 10"),
            paso("3", "Benchmark", "14 modelos × 5 semillas, mismo presupuesto"),
            paso("4", "Protocolo", "Preregistro, partición agrupada, un acceso a test"),
            paso("5", "Ajuste y sobreajuste", "Curvas, brecha train93–val"),
            paso("6", "Comparación estadística", "Wilcoxon, bootstrap, t; Holm; Friedman"),
            paso("7", "Modelo original", "SGW, teorema, 10 semillas, ablación"),
            paso("8", "Evaluación ampliada", "Robustez, escala, 3 conjuntos extra"),
        ], className="pasos")),
        ui.seccion("Limitaciones y ética", html.Div(html.Ul([
            html.Li("El modelo original no supera de forma significativa a los mejores modelos de dos etapas "
                    "(R3 no se cumple): úsese como Mask R-CNN mejorado en recall de tornillos pequeños, no como "
                    "un salto de desempeño."),
            html.Li("La ganancia en tornillos diminutos se concentra en una escena; no se ha confirmado que se "
                    "generalice a escenas nuevas."),
            html.Li("Un falso negativo en desensamblado robótico puede dañar piezas o personas: el sistema debe "
                    "usarse con verificación humana o sensorial."),
            html.Li("El dataset no declara licencia: el dashboard muestra imágenes reducidas solo con fines "
                    "académicos."),
        ], className="mb-0"), className="panel")),
        html.P(["Autores: Ivan Berrio, Jose David Castillo y Jaime Besada. Curso de Deep Learning, Entregable 3. "
                f"Versión {data.VERSION}, {data.ACTUALIZADO}."], className="nota mt-4"),
    ])
