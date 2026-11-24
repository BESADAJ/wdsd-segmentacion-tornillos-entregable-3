import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
# código del E1/E2 (wdsd_inst): por defecto, el repo del E2 al lado; se puede cambiar con WDSD_E2_SRC
E2_SRC = Path(os.environ.get("WDSD_E2_SRC", RAIZ.parents[1] / "Entregable_2_Deep_Learning" / "src"))
if E2_SRC.exists():
    sys.path.insert(0, str(E2_SRC))
