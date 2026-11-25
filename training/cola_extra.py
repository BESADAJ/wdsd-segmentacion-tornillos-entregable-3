"""Cola de los 3 conjuntos adicionales (experiments/protocolo.md §6, tag e3-conjuntos).

Por rondas de semilla (42, 43, 44): en cada una, para cada conjunto, sgw y maskrcnn con la configuración exacta del
WDSD y kappa = 0.45 fijo. Cada corrida es un proceso aparte (training/entrenar_sgw.py final) y las terminadas se
saltan al relanzar. Si antes hay otro trabajo en la GPU, --esperar_pid espera a que termine.

Uso: python training/cola_extra.py [--raiz D:/wdsd_datos/extra] [--salida D:/wdsd_e3/results_extra]
                                   [--conjuntos bbbc039,minneapple,nuinsseg] [--semillas 42-44] [--esperar_pid N]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def vivo(pid: int) -> bool:
    salida = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", type=Path, default=Path("D:/wdsd_datos/extra"))
    ap.add_argument("--salida", type=Path, default=Path("D:/wdsd_e3/results_extra"))
    ap.add_argument("--conjuntos", default="bbbc039,minneapple,nuinsseg")
    ap.add_argument("--semillas", default="42-44")
    ap.add_argument("--esperar_pid", type=int)
    a = ap.parse_args()
    if a.esperar_pid:
        while vivo(a.esperar_pid):
            time.sleep(60)
    i, j = (int(x) for x in a.semillas.split("-"))
    for s in range(i, j + 1):
        for c in a.conjuntos.split(","):
            if not (a.raiz / c / "coco" / "test.json").exists():
                print(f"[{time.strftime('%H:%M')}] {c}: datos no preparados, se salta", flush=True)
                continue
            for v in ("sgw", "maskrcnn"):
                print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {c} {v} s{s}", flush=True)
                subprocess.run([sys.executable, "-u", str(RAIZ / "training" / "entrenar_sgw.py"), "final",
                                "--datos", str(a.raiz / c), "--salida", str(a.salida / c), "--variante", v,
                                "--kappa", "0.45", "--semillas", str(s)], check=False,
                               env={**os.environ, "PYTHONUTF8": "1"})
    print("cola extra terminada", flush=True)


if __name__ == "__main__":
    main()
