"""Prepara los 3 conjuntos adicionales registrados en experiments/protocolo.md §6 (tag e3-conjuntos).

Para cada conjunto escribe <destino>/<nombre>/img/*.png (RGB de 8 bits) y <destino>/<nombre>/coco/{train,val,test}.json
con el mismo esquema que el WDSD: cada imagen lleva "ruta" (absoluta) y "grupo"; cada anotación lleva la máscara
de instancia como RLE sin comprimir, "bbox" y "area". train93 lo crea después training/entrenar_sgw.py.

Decodificación de las máscaras:
- BBBC039: el canal R codifica los núcleos con 3 "colores" (1, 2, 3) para que los que se tocan tengan valores
  distintos; cada instancia es un componente conexo de un mismo valor. Imágenes de 16 bits -> percentiles 0,5-99,5.
- NuInsSeg: máscaras de instancia etiquetadas ("label masks"), un entero por núcleo.
- MinneApple (réplica de AgML): la máscara es binaria (se perdieron las instancias). Se separan por componentes
  conexos (8-conectividad), como quedó registrado: manzanas que se tocan quedan unidas.
Se descartan componentes de menos de 4 px (ruido de anotación).

Uso: python experiments/preparar_extra.py <crudo> <destino> [--solo bbbc039,nuinsseg,minneapple]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MIN_PX = 4


def rle(m: np.ndarray) -> dict:
    """RLE sin comprimir de COCO (orden de columnas, empieza con ceros)."""
    v = (m > 0).ravel(order="F").astype(np.int8)
    bordes = np.flatnonzero(np.diff(v)) + 1            # posiciones donde cambia el valor
    largos = np.diff(np.concatenate([[0], bordes, [v.size]]))
    counts = ([0] if v[0] == 1 else []) + largos.tolist()  # el primer tramo cuenta ceros
    return {"counts": [int(c) for c in counts], "size": [int(m.shape[0]), int(m.shape[1])]}


def instancias_de_etiquetas(lab: np.ndarray):
    for k in np.unique(lab):
        if k == 0:
            continue
        m = lab == k
        if m.sum() >= MIN_PX:
            yield m


def instancias_de_colores(cod: np.ndarray, conect8=False):
    est = np.ones((3, 3)) if conect8 else None
    for v in np.unique(cod):
        if v == 0:
            continue
        lab, n = ndimage.label(cod == v, structure=est)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() >= MIN_PX:
                yield m


def a_uint8_rgb(a: np.ndarray) -> Image.Image:
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a[..., :3])


def particion_agrupada(grupos: dict[str, str]) -> dict[str, str]:
    """70/15/15 sobre los grupos (no sobre las imágenes), random.Random(0)."""
    gs = sorted(set(grupos.values()))
    random.Random(0).shuffle(gs)
    n = len(gs)
    n_tr, n_va = round(0.70 * n), round(0.15 * n)
    split_g = {g: "train" if i < n_tr else "val" if i < n_tr + n_va else "test" for i, g in enumerate(gs)}
    return {img: split_g[g] for img, g in grupos.items()}


class Escritor:
    def __init__(self, destino: Path):
        self.dest = destino
        (destino / "img").mkdir(parents=True, exist_ok=True)
        (destino / "coco").mkdir(parents=True, exist_ok=True)
        self.d = defaultdict(lambda: {"images": [], "annotations": [],
                                      "categories": [{"id": 1, "name": "objeto"}]})
        self.n_img = self.n_ann = 0
        self.gid: dict[str, int] = {}

    def agregar(self, nombre: str, img: Image.Image, mascaras, split: str, grupo: str):
        self.n_img += 1
        ruta = self.dest / "img" / f"{Path(nombre).stem}.png"
        img.save(ruta)
        w, h = img.size
        g = self.gid.setdefault(grupo, len(self.gid))
        self.d[split]["images"].append({"id": self.n_img, "file_name": ruta.name, "width": w, "height": h,
                                        "ruta": str(ruta).replace("\\", "/"), "grupo": g, "grupo_nombre": grupo})
        for m in mascaras:
            assert m.shape == (h, w), (m.shape, h, w)
            self.n_ann += 1
            ys, xs = np.nonzero(m)
            self.d[split]["annotations"].append({
                "id": self.n_ann, "image_id": self.n_img, "category_id": 1, "iscrowd": 0, "segmentation": [rle(m)],  # lista: el cargador del E2 hace merge(frPyObjects(seg))
                "area": float(m.sum()), "bbox": [float(xs.min()), float(ys.min()),
                                                 float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]})

    def cerrar(self):
        info = {}
        for split, d in self.d.items():
            p = self.dest / "coco" / f"{split}.json"
            p.write_text(json.dumps(d), encoding="utf8")
            areas = [a["area"] for a in d["annotations"]]
            info[split] = {"imagenes": len(d["images"]), "objetos": len(areas),
                           "grupos": len({i["grupo"] for i in d["images"]}),
                           "area_mediana_px": float(np.median(areas)) if areas else None,
                           "objetos_por_imagen_max": int(max(np.bincount([a["image_id"] for a in d["annotations"]])))
                           if areas else 0,
                           "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
        (self.dest / "resumen_preparacion.json").write_text(json.dumps(info, indent=1), encoding="utf8")
        print(json.dumps(info, indent=1))


def bbbc039(crudo: Path, destino: Path):
    c = crudo / "bbbc039"
    w = Escritor(destino / "bbbc039")
    for split, archivo in (("train", "training.txt"), ("val", "validation.txt"), ("test", "test.txt")):
        for nombre in (c / "metadata" / archivo).read_text().split():
            cod = np.array(Image.open(c / "masks" / nombre))[..., 0]
            img = a_uint8_rgb(np.array(Image.open(c / "images" / nombre.replace(".png", ".tif"))))
            w.agregar(nombre, img, list(instancias_de_colores(cod)), split, nombre)  # grupo = imagen
    w.cerrar()


def nuinsseg(crudo: Path, destino: Path):
    c = crudo / "nuinsseg"
    tejidos = sorted({p.parent.parent for p in c.rglob("tissue images/*.png")})
    grupos, pares = {}, {}
    for t in tejidos:
        organo = t.name
        for p in sorted((t / "tissue images").glob("*.png")):
            etiqueta = next((t / "label masks").glob(p.stem + ".*"), None) or \
                next((t / "label masks modify").glob(p.stem + ".*"), None)
            if etiqueta is None:
                print("sin máscara:", p); continue
            clave = f"{organo}__{p.stem}"
            grupos[clave], pares[clave] = organo, (p, etiqueta)
    split = particion_agrupada(grupos)
    w = Escritor(destino / "nuinsseg")
    for clave in sorted(pares):
        p, e = pares[clave]
        lab = np.array(Image.open(e))
        if lab.ndim == 3:
            lab = lab[..., 0]
        w.agregar(re.sub(r"[^\w.-]", "_", clave), a_uint8_rgb(np.array(Image.open(p).convert("RGB"))),
                  list(instancias_de_etiquetas(lab)), split[clave], grupos[clave])
    w.cerrar()


def minneapple(crudo: Path, destino: Path):
    import pyarrow.parquet as pq
    filas = []
    for f in sorted((crudo / "minneapple").glob("*.parquet")):
        filas += pq.read_table(f).to_pylist()
    grupos = {r["image"]["path"]: r["image"]["path"].split("_image")[0] for r in filas}
    split = particion_agrupada(grupos)
    w = Escritor(destino / "minneapple")
    for r in sorted(filas, key=lambda r: r["image"]["path"]):
        nombre = r["image"]["path"]
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        m = np.array(Image.open(io.BytesIO(r["mask"]["bytes"])))
        if m.ndim == 3:
            m = m[..., 0]
        w.agregar(nombre, img, list(instancias_de_colores((m > 0).astype(np.uint8), conect8=True)), split[nombre],
                  grupos[nombre])
    w.cerrar()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crudo", type=Path)
    ap.add_argument("destino", type=Path)
    ap.add_argument("--solo", default="bbbc039,nuinsseg,minneapple")
    a = ap.parse_args()
    for n in a.solo.split(","):
        print(f"== {n}")
        {"bbbc039": bbbc039, "nuinsseg": nuinsseg, "minneapple": minneapple}[n](a.crudo, a.destino)


if __name__ == "__main__":
    main()
