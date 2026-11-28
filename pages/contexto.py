"""Sección 1: contexto, preguntas, ficha de datos, metodología y limitaciones."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import data

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · WDSD")


def tarjeta(cifra, texto, color="primary"):
    return dbc.Card(dbc.CardBody([html.H3(cifra, className=f"text-{color} mb-0"), html.Small(texto)]),
                    className="h-100 shadow-sm")


def paso(n, titulo, detalle):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(n, className="badge bg-primary mb-1"),
                                          html.H6(titulo, className="mb-1"), html.Small(detalle)]),
                            className="h-100"), xs=6, md=3, lg=True, className="mb-2")


def layout():
    t = data.tabla_maestra().set_index("modelo")
    sgw, mr = t.loc["sgw_k0.45"], t.loc["maskrcnn"]
    return html.Div([
        html.H2("Encontrar tornillos diminutos para desensamblar aparatos electrónicos"),
        html.P([
            "En 2022 se generaron 62 millones de toneladas de residuos electrónicos y solo el 22,3 % se recogió y "
            "recicló de forma documentada (", html.Em("Global E-waste Monitor 2024"), ", Baldé et al., ITU y "
            "UNITAR). Desensamblar con robots exige localizar cada tornillo, y en las imágenes del WDSD un tornillo "
            "mide en mediana 10,6 px de lado: el 98,8 % es un objeto «pequeño» según COCO."], className="lead"),
        dbc.Row([
            dbc.Col(tarjeta("62 Mt", "residuos electrónicos generados en 2022 (GEM 2024)"), md=3, className="mb-2"),
            dbc.Col(tarjeta("10,6 px", "lado mediano de un tornillo en WDSD"), md=3, className="mb-2"),
            dbc.Col(tarjeta(f"{sgw.F1_test:.3f}".replace(".", ","), "F1 de instancia por grupo del modelo original (SGW)", "warning"),
                    md=3, className="mb-2"),
            dbc.Col(tarjeta(f"+{sgw.F1_test - mr.F1_test:.3f}".replace(".", ","), "frente a Mask R-CNN: no significativo", "secondary"),
                    md=3, className="mb-2"),
        ]),
        html.H4("Preguntas de investigación e hipótesis", className="mt-4"),
        dbc.Row([
            dbc.Col(dbc.ListGroup([
                dbc.ListGroupItem([html.B("RQ1 (E2). "), "¿Algún modelo del Top 10 supera al mejor baseline fuerte? "
                                   "No: Mask R-CNN no es superado (Holm, p = 1)."]),
                dbc.ListGroupItem([html.B("RQ2 (E2). "), "¿Las dos etapas superan a la familia de prototipos? Sí, "
                                   "+0,18 de F1 (p = 0,0002), sin atribuirlo solo a la arquitectura."]),
                dbc.ListGroupItem([html.B("RQ3 (E2). "), "¿Cómo cae el recall con el tamaño? En el tercil pequeño "
                                   "ningún modelo pasa de 0,37 (YOLOv7): la ceguera ante tornillos diminutos es compartida."]),
                dbc.ListGroupItem([html.B("RQ4 (E2). "), "¿Hay sobreajuste? La brecha de F1 entre train93 y val "
                                   "es de 0,05 a 0,08 en los modelos de dos etapas, con IC que incluyen 0."]),
            ]), md=6),
            dbc.Col(dbc.Alert([
                html.H5("Hipótesis del mecanismo (modelo original, H-M)"),
                html.P("Si las propuestas de Mask R-CNN se asignan con una similitud de Wasserstein gaussiana "
                       "normalizada por el tamaño de cada tornillo (SGW), en vez de con IoU, los tornillos diminutos "
                       "reciben tantos positivos como los grandes y el recall del tercil pequeño sube al menos 0,05."),
                html.Small(["Mecanismo transferido y adaptado de la Normalized Wasserstein Distance para detección "
                            "aérea de objetos diminutos (Wang, Xu, Yang y Yu, 2021, arXiv:2110.13389; ISPRS 2022). "
                            "La normalización por tamaño con piso y su demostración son propias."]),
            ], color="warning"), md=6),
        ]),
        html.H4("Ficha de datos (Data Card)", className="mt-4"),
        dbc.Table([html.Tbody([html.Tr([html.Th(k), html.Td(v)]) for k, v in [
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
        ]])], bordered=True, size="sm", responsive=True),
        html.H4("Metodología", className="mt-4"),
        dbc.Row([
            paso("1", "EDA", "Tamaños, densidad, escenas y fuga"),
            paso("2", "Revisión sistemática", "PRISMA, 64 estudios, Top 10"),
            paso("3", "Benchmark", "14 modelos × 5 semillas, mismo presupuesto"),
            paso("4", "Protocolo", "Preregistro, partición agrupada, un acceso a test"),
            paso("5", "Ajuste y sobreajuste", "Curvas, brecha train93–val"),
            paso("6", "Comparación estadística", "Wilcoxon, bootstrap, t; Holm; Friedman"),
            paso("7", "Modelo original", "SGW, teorema, 10 semillas, ablación"),
            paso("8", "Evaluación ampliada", "Robustez, escala, 3 conjuntos extra"),
        ]),
        html.H4("Limitaciones y ética", className="mt-4"),
        html.Ul([
            html.Li("El modelo original no supera de forma significativa a los mejores modelos de dos etapas "
                    "(R3 no se cumple): úsese como Mask R-CNN mejorado en recall de tornillos pequeños, no como "
                    "un salto de desempeño."),
            html.Li("La ganancia en tornillos diminutos se concentra en una escena; no se ha confirmado que se "
                    "generalice a escenas nuevas."),
            html.Li("Un falso negativo en desensamblado robótico puede dañar piezas o personas: el sistema debe "
                    "usarse con verificación humana o sensorial."),
            html.Li("El dataset no declara licencia: el dashboard muestra imágenes reducidas solo con fines "
                    "académicos."),
        ]),
        html.P(["Autores: Ivan Berrio, Jose David Castillo y Jaime Besada. Curso de Deep Learning, Entregable 3. "
                f"Versión {data.VERSION}, {data.ACTUALIZADO}."], className="text-muted"),
    ])
