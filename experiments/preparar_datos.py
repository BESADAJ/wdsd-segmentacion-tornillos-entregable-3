"""Genera <salida>/coco/{train,val,test}.json con la partición agrupada congelada del E1 (sin regenerarla).

Uso: python experiments/preparar_datos.py <raiz con WDSD-*/WDSD> <particion.csv del E1> <salida> [--src-e2 <src del E2>]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("raiz")
ap.add_argument("particion")
ap.add_argument("salida")
ap.add_argument("--src-e2", default=str(Path(__file__).resolve().parents[3] / "Entregable_2_Deep_Learning" / "src"))
a = ap.parse_args()
sys.path.insert(0, a.src_e2)
from wdsd_inst.datos import preparar  # noqa: E402  mismo código que el E2

res = preparar(Path(a.raiz), Path(a.particion), Path(a.salida), yolo=False)
print(json.dumps(res, indent=1))
for s in ("train", "val", "test"):
    p = Path(a.salida) / "coco" / f"{s}.json"
    print(s, hashlib.sha256(p.read_bytes()).hexdigest()[:16])
