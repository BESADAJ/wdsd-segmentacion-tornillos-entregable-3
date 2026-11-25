"""Modelo original del E3: Mask R-CNN con asignación de etiquetas por similitud gaussiana de Wasserstein
normalizada por el tamaño de cada objeto (SGW, "scale-normalized Gaussian Wasserstein").

Procedencia (detalle en experiments/procedencia.md):
    El mecanismo se transfiere de la detección de objetos diminutos en imágenes aéreas:
    Wang, J., Xu, C., Yang, W. y Yu, L. (2021). A Normalized Gaussian Wasserstein Distance for Tiny Object
    Detection. arXiv:2110.13389. Allí NWD(a, b) = exp(-W2(N_a, N_b) / C), con C una constante global
    (el tamaño medio de los objetos del conjunto).

Modificación algorítmica (lo propio de este trabajo):
    1. La constante global C se reemplaza por una escala que depende del objeto de referencia:
           D(g) = kappa * (sqrt(w_g * h_g) + c),
       con c un piso en píxeles fijado a priori al paso de la rejilla más fina de anclas (4 px en P2).
       Para objetos diminutos (sqrt(w h) << c) la similitud tolera desplazamientos absolutos de unos
       pocos píxeles, como NWD; para objetos grandes tiende a una medida invariante a la escala, como el IoU.
       Con C fija, NWD es demasiado estricta con los objetos grandes y demasiado laxa con los diminutos.
    2. La similitud reemplaza al IoU en las dos asignaciones de Mask R-CNN: anclas -> verdad (RPN) y
       propuestas -> verdad (cabeza RoI, de la que dependen también los ejemplos que entrenan la máscara).
    3. Los umbrales de los emparejadores no cambian (0.7/0.3 en RPN, 0.5 en RoI). kappa se calibra a priori
       para que, en el límite de objetos grandes (c despreciable), SGW = 0.5 coincida con IoU = 0.5:
           traslación de t*w en x:      IoU = (1-t)/(1+t) = 0.5 -> t = 1/3,  s = exp(-1/(3 kappa))          -> kappa = 0.481
           escala r (cajas concéntricas): IoU = 1/r^2 = 0.5 -> r = sqrt(2), s = exp(-(r-1)/(sqrt(2) kappa)) -> kappa = 0.423
       Se toma kappa_0 = 0.45 (punto medio). Así los umbrales significan lo mismo que con IoU en objetos
       grandes y solo cambian para los diminutos. La búsqueda (T = 3, igual que la tasa en el E2) es
       kappa_0 x {0.5, 1, 2}.

Todo el núcleo (distancia, similitud y asignación de la cabeza RoI) está escrito aquí desde la formulación;
de torchvision se usan solo los bloques del Mask R-CNN de referencia del E1 (backbone, RPN, cabezas, Matcher).
"""
from __future__ import annotations

import math

import torch
from torch import nn

PISO_C_PX = 4.0   # paso de la rejilla de anclas en P2; fijado a priori, no se ajusta
KAPPA_DEFECTO = 0.45  # calibración con el IoU en el límite de objetos grandes (docstring del módulo)
REJILLA_KAPPA = tuple(round(KAPPA_DEFECTO * f, 4) for f in (0.5, 1.0, 2.0))


def parametros_gaussianos(cajas: torch.Tensor) -> torch.Tensor:
    """Caja (x1, y1, x2, y2) -> p = (cx, cy, w/2, h/2).

    La caja se modela como N(mu, Sigma) con mu = (cx, cy) y Sigma = diag(w^2/4, h^2/4). Para gaussianas con
    covarianza diagonal, W2(N_a, N_b) = ||p_a - p_b||_2 (Wang et al. 2021, ec. 5), así que p es la
    representación en la que la distancia es euclídea.
    """
    if cajas.ndim != 2 or cajas.shape[-1] != 4:
        raise ValueError(f"se esperaban cajas de forma (N, 4), llegó {tuple(cajas.shape)}")
    x1, y1, x2, y2 = cajas.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2), dim=-1)


def distancia_w2(ref: torch.Tensor, cajas: torch.Tensor) -> torch.Tensor:
    """Matriz (n_ref, n_cajas) de distancias de Wasserstein-2 entre las gaussianas de las cajas."""
    pr, pc = parametros_gaussianos(ref), parametros_gaussianos(cajas)
    dif = pr[:, None, :] - pc[None, :, :]
    # sqrt con un épsilon dentro para que el gradiente exista también en dif = 0
    return torch.sqrt((dif * dif).sum(-1) + 1e-12)


def escala_ref(ref: torch.Tensor, kappa: float, piso: float) -> torch.Tensor:
    """D(g) = kappa * (sqrt(w_g h_g) + c), una por caja de referencia."""
    w = (ref[:, 2] - ref[:, 0]).clamp(min=0)
    h = (ref[:, 3] - ref[:, 1]).clamp(min=0)
    return kappa * (torch.sqrt(w * h) + piso)


class SimilitudSGW(nn.Module):
    """s(g, b) = exp(-W2(N_g, N_b) / D(g)), en (0, 1]. g es la caja de referencia (verdad); b, la candidata.

    No es simétrica: la escala la fija la verdad, que es lo que se quiere al asignar etiquetas.
    No tiene parámetros entrenables: kappa y el piso son hiperparámetros.
    """

    def __init__(self, kappa: float = KAPPA_DEFECTO, piso: float = PISO_C_PX):
        super().__init__()
        if kappa <= 0 or piso <= 0:
            raise ValueError("kappa y el piso deben ser positivos (la cota de Lipschitz es 1 / (kappa * piso))")
        self.kappa, self.piso = float(kappa), float(piso)

    def forward(self, ref: torch.Tensor, cajas: torch.Tensor) -> torch.Tensor:
        if ref.numel() == 0 or cajas.numel() == 0:
            return torch.zeros((ref.shape[0], cajas.shape[0]), dtype=torch.float32, device=cajas.device)
        ref, cajas = ref.float(), cajas.float()
        d = distancia_w2(ref, cajas)
        return torch.exp(-d / escala_ref(ref, self.kappa, self.piso)[:, None])

    def cota_lipschitz(self, en: str = "p") -> float:
        """Cota del teorema 1: K = 1 / (kappa c) respecto a p, y K / sqrt(2) respecto a las esquinas."""
        k = 1.0 / (self.kappa * self.piso)
        return k if en == "p" else k / math.sqrt(2)

    def extra_repr(self) -> str:
        return f"kappa={self.kappa}, piso={self.piso}"


def _asignar_propuestas_sgw(self, propuestas, gt_cajas, gt_etiquetas):
    """Reimplementación de RoIHeads.assign_targets_to_proposals con self.similitud en lugar de box_iou.

    Misma lógica que la de referencia: emparejador con umbral 0.5, fondo -> etiqueta 0, zona ignorada -> -1.
    """
    idx_emparejados, etiquetas = [], []
    for prop, gt, et in zip(propuestas, gt_cajas, gt_etiquetas):
        if gt.numel() == 0:
            disp = prop.device
            idx = torch.zeros((prop.shape[0],), dtype=torch.int64, device=disp)
            lab = torch.zeros((prop.shape[0],), dtype=torch.int64, device=disp)
        else:
            calidad = self.similitud(gt, prop)
            idx_crudo = self.proposal_matcher(calidad)
            idx = idx_crudo.clamp(min=0)
            lab = et[idx].to(dtype=torch.int64)
            lab[idx_crudo == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0
            lab[idx_crudo == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1
        idx_emparejados.append(idx)
        etiquetas.append(lab)
    return idx_emparejados, etiquetas


def aplicar_sgw(modelo: nn.Module, kappa: float = KAPPA_DEFECTO, piso: float = PISO_C_PX,
                en_rpn: bool = True, en_roi: bool = True) -> nn.Module:
    """Cambia la asignación de etiquetas de un MaskRCNN de torchvision. Los interruptores sirven a la ablación."""
    sim = SimilitudSGW(kappa, piso)
    if en_rpn:
        modelo.rpn.box_similarity = sim
    if en_roi:
        modelo.roi_heads.similitud = sim
        modelo.roi_heads.assign_targets_to_proposals = _asignar_propuestas_sgw.__get__(modelo.roi_heads)
    modelo.sgw_config = {"kappa": kappa, "piso": piso, "en_rpn": en_rpn, "en_roi": en_roi}
    return modelo


def crear_modelo(preentrenado: bool = True, kappa: float = KAPPA_DEFECTO, piso: float = PISO_C_PX,
                 en_rpn: bool = True, en_roi: bool = True) -> nn.Module:
    """El Mask R-CNN del E1/E2 (anclas de 8 a 128 px, sin reescalar) con la asignación SGW."""
    return aplicar_sgw(mask_rcnn_base(preentrenado), kappa, piso, en_rpn, en_roi)


def mask_rcnn_base(preentrenado: bool = True) -> nn.Module:
    """Constructor del modelo base: el de wdsd_inst.maskrcnn (E1/E2) si está disponible, para que solo cambie la
    asignación; si no (por ejemplo en CI), una copia con la misma configuración: ResNet-50-FPN, anclas de 8 a 128 px
    con aspectos 0,5/1/2, min_size 720, max_size 1280, 100 detecciones y cabezas de 2 clases."""
    try:
        from wdsd_inst import maskrcnn
        return maskrcnn.crear_modelo(preentrenado)
    except ImportError:
        pass
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    anclas = AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 5)
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if preentrenado else None,
                              weights_backbone=None, num_classes=None if preentrenado else 2, min_size=720,
                              max_size=1280, rpn_anchor_generator=anclas, box_detections_per_img=100)
    if preentrenado:
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return m
