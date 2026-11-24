"""Cola completa del E3 en una máquina (protocolo congelado, experiments/protocolo.md §4).

1. Espera a que termine la búsqueda de kappa (3 corridas con resumen.json o fallo.json).
2. Elige kappa con la regla preregistrada: máximo F1 por grupo en val; si kappa_0 queda a 0.005 o menos del
   máximo, gana kappa_0 (misma regla de empate que el E2). Escribe busqueda/sgw_elegida.json.
3. Corre las finales por rondas de semilla: en cada semilla, sgw, maskrcnn, sgw_roi, sgw_rpn, sgw_c0.
   Cada corrida es un proceso aparte (como en el E2); las terminadas se saltan al relanzar.

Uso: python training/cola_e3.py --datos D:/wdsd_datos/datos_inst [--semillas 42-51] [--variantes sgw,maskrcnn,...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import original_model as om  # noqa: E402

ORDEN = ("sgw", "maskrcnn", "sgw_roi", "sgw_rpn", "sgw_c0")
EMPATE = 0.005


def elegir_kappa(salida: Path) -> float:
    b = salida / "busqueda"
    while True:
        hechas = {k: b / f"sgw_k{k:g}_s42" for k in om.REJILLA_KAPPA}
        if all((c / "resumen.json").exists() or (c / "fallo.json").exists() for c in hechas.values()):
            break
        time.sleep(60)
    f1 = {}
    for k, c in hechas.items():
        if (c / "resumen.json").exists():
            f1[k] = json.loads((c / "resumen.json").read_text(encoding="utf8"))["metricas"]["val"]["F1_grupo"]
    if not f1:
        raise SystemExit("las tres corridas de la búsqueda fallaron")
    mejor = max(f1, key=f1.get)
    elegido = om.KAPPA_DEFECTO if om.KAPPA_DEFECTO in f1 and f1[mejor] - f1[om.KAPPA_DEFECTO] <= EMPATE else mejor
    (b / "sgw_elegida.json").write_text(json.dumps(
        {"elegido": elegido, "por_defecto": om.KAPPA_DEFECTO, "regla": f"max F1_grupo val; empate {EMPATE} -> kappa_0",
         "F1_grupo_val": {str(k): v for k, v in f1.items()}, "fallidas": [str(k) for k in om.REJILLA_KAPPA if k not in f1]},
        indent=1), encoding="utf8")
    print(f"kappa elegido: {elegido}  (val F1 grupo: {f1})", flush=True)
    return elegido


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", type=Path, default=Path(os.environ.get("WDSD_E3_SALIDA", "D:/wdsd_e3/results")))
    ap.add_argument("--semillas", default="42-51")
    ap.add_argument("--variantes", default=",".join(ORDEN))
    a = ap.parse_args()
    i, j = (int(x) for x in a.semillas.split("-")) if "-" in a.semillas else (int(a.semillas),) * 2
    kappa = elegir_kappa(a.salida)
    variantes = [v for v in ORDEN if v in a.variantes.split(",")]
    for s in range(i, j + 1):
        for v in variantes:
            cmd = [sys.executable, "-u", str(RAIZ / "training" / "entrenar_sgw.py"), "final", "--datos", a.datos,
                   "--salida", str(a.salida), "--variante", v, "--kappa", str(kappa), "--semillas", str(s)]
            print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {v} s{s}", flush=True)
            subprocess.run(cmd, check=False)
    print("cola terminada", flush=True)


if __name__ == "__main__":
    main()
