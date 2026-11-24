"""Pruebas del modelo original (E3 §6.5): propiedades demostradas, gradientes, reproducibilidad, entradas
inválidas, serialización e integración con Mask R-CNN."""
import math

import pytest
import torch

from original_model import SimilitudSGW, aplicar_sgw, distancia_w2, parametros_gaussianos


def cajas_aleatorias(n, g, lado_min=2.0, lado_max=80.0):
    xy = torch.rand(n, 2, generator=g) * 1000
    wh = lado_min + torch.rand(n, 2, generator=g) * (lado_max - lado_min)
    return torch.cat([xy, xy + wh], 1)


# --- Teorema 1 (a): rango e identidad -------------------------------------------------------------------
def test_rango_e_identidad():
    g = torch.Generator().manual_seed(0)
    a, b = cajas_aleatorias(50, g), cajas_aleatorias(60, g)
    s = SimilitudSGW()(a, b)
    assert s.shape == (50, 60)
    assert (s >= 0).all() and (s <= 1).all()  # en float32 exp(-d/D) puede dar 0 exacto con cajas muy lejanas
    cerca = a + torch.randn(a.shape, generator=g) * 5.0
    assert (torch.diagonal(SimilitudSGW()(a, cerca)) > 0).all()
    assert torch.allclose(torch.diagonal(SimilitudSGW()(a, a)), torch.ones(50), atol=1e-5)


def test_w2_es_euclidea_en_p():
    g = torch.Generator().manual_seed(1)
    a, b = cajas_aleatorias(20, g), cajas_aleatorias(20, g)
    d = distancia_w2(a, b)
    esperado = torch.cdist(parametros_gaussianos(a), parametros_gaussianos(b))
    assert torch.allclose(d, esperado, atol=1e-4)


# --- Teorema 1 (b): cota de Lipschitz -------------------------------------------------------------------
@pytest.mark.parametrize("kappa,piso", [(0.5, 4.0), (1.0, 4.0), (2.0, 4.0), (1.0, 1.0)])
def test_cota_lipschitz_esquinas(kappa, piso):
    g = torch.Generator().manual_seed(2)
    sim = SimilitudSGW(kappa, piso)
    ref = cajas_aleatorias(200, g, 1.0, 120.0)
    b = cajas_aleatorias(200, g, 1.0, 120.0)
    pert = b + torch.randn(b.shape, generator=g) * 3.0
    num = (torch.diagonal(sim(ref, b)) - torch.diagonal(sim(ref, pert))).abs()
    den = (b - pert).norm(dim=1)
    cociente = (num / den).max().item()
    assert cociente <= sim.cota_lipschitz("esquinas") * (1 + 1e-4)


# --- Teorema 1 (c): sensibilidad no nula donde el IoU es 0 ----------------------------------------------
def test_gradiente_no_nulo_con_cajas_disjuntas():
    from torchvision.ops import box_iou
    gt = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    b = torch.tensor([[120.0, 100.0, 130.0, 110.0]], requires_grad=True)  # disjunta: IoU = 0
    iou = box_iou(gt, b)
    assert iou.item() == 0.0
    SimilitudSGW()(gt, b).sum().backward()
    assert b.grad.abs().sum() > 0
    b2 = b.detach().clone().requires_grad_(True)
    box_iou(gt, b2).sum().backward()
    assert b2.grad.abs().sum() == 0  # el IoU no distingue entre candidatas disjuntas


# --- Teorema 1 (d): comportamiento con la escala --------------------------------------------------------
def test_invarianza_escala_sin_piso_y_limite_nwd():
    g = torch.Generator().manual_seed(3)
    a, b = cajas_aleatorias(30, g), cajas_aleatorias(30, g)
    lam = 7.0
    casi_sin_piso = SimilitudSGW(1.0, 1e-6)
    assert torch.allclose(casi_sin_piso(a, b), casi_sin_piso(a * lam, b * lam), atol=1e-4)
    # identidad de escala exacta con piso: s_c(lam g, lam b) = s_{c/lam}(g, b)
    assert torch.allclose(SimilitudSGW(1.0, 4.0)(a * lam, b * lam), SimilitudSGW(1.0, 4.0 / lam)(a, b), atol=1e-5)


def test_mas_tolerante_que_iou_en_tornillos_diminutos():
    """Un desplazamiento de 2 px hunde el IoU de una caja de 8 px por debajo de 0.5; SGW no cae tanto."""
    from torchvision.ops import box_iou
    gt = torch.tensor([[100.0, 100.0, 108.0, 108.0]])
    b = gt + torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    iou = box_iou(gt, b).item()
    s = SimilitudSGW()(gt, b).item()
    assert iou < 0.5 <= s


def test_calibracion_con_iou_en_objetos_grandes():
    """Con c despreciable frente al objeto, SGW = 0.5 y IoU = 0.5 caen en el mismo punto (+-0.06)."""
    from torchvision.ops import box_iou
    gt = torch.tensor([[0.0, 0.0, 400.0, 400.0]])
    tras = gt + torch.tensor([[400 / 3, 0.0, 400 / 3, 0.0]])        # IoU = 0.5 por traslación
    r = math.sqrt(2)
    esc = torch.tensor([[200 - 200 * r, 200 - 200 * r, 200 + 200 * r, 200 + 200 * r]])  # IoU = 0.5 por escala
    for b in (tras, esc):
        assert abs(box_iou(gt, b).item() - 0.5) < 1e-3
        assert abs(SimilitudSGW()(gt, b).item() - 0.5) < 0.06


# --- Entradas inválidas, vacías y dispositivo -----------------------------------------------------------
def test_entradas_invalidas():
    with pytest.raises(ValueError):
        SimilitudSGW(kappa=0.0)
    with pytest.raises(ValueError):
        SimilitudSGW()(torch.zeros(3, 5), torch.zeros(2, 4))


def test_entradas_vacias():
    s = SimilitudSGW()(torch.zeros(0, 4), torch.rand(5, 4))
    assert s.shape == (0, 5)


# --- Integración con Mask R-CNN -------------------------------------------------------------------------
maskrcnn = pytest.importorskip("wdsd_inst.maskrcnn")


def _modelo(semilla, **kw):
    torch.manual_seed(semilla)
    m = maskrcnn.crear_modelo(preentrenado=False)
    m.transform.min_size, m.transform.max_size = (256,), 256  # imagen pequeña para que la prueba corra en CPU
    return aplicar_sgw(m, **kw)


def _lote():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 256, 256, generator=g)
    cajas = torch.tensor([[40.0, 40.0, 50.0, 50.0], [120.0, 90.0, 132.0, 101.0]])
    masks = torch.zeros(2, 256, 256, dtype=torch.uint8)
    for k, (x1, y1, x2, y2) in enumerate(cajas.int().tolist()):
        masks[k, y1:y2, x1:x2] = 1
    return [img], [{"boxes": cajas, "labels": torch.ones(2, dtype=torch.int64), "masks": masks}]


def test_asignacion_reemplazada():
    m = _modelo(0)
    assert isinstance(m.rpn.box_similarity, SimilitudSGW)
    assert isinstance(m.roi_heads.similitud, SimilitudSGW)
    sin = _modelo(0, en_rpn=False, en_roi=False)
    assert not isinstance(sin.rpn.box_similarity, SimilitudSGW)


def test_todos_los_parametros_reciben_gradiente():
    m = _modelo(0)
    m.train()
    imgs, tgts = _lote()
    perdidas = m(imgs, tgts)
    assert set(perdidas) >= {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    sum(perdidas.values()).backward()
    sin_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    # en el backbone sin preentrenar las primeras capas están congeladas por torchvision (trainable_layers);
    # todo lo que es entrenable debe recibir gradiente finito
    assert not sin_grad, sin_grad[:5]
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_reproducible_con_misma_semilla():
    imgs, _ = _lote()
    a, b = _modelo(1).eval(), _modelo(1).eval()
    with torch.no_grad():
        oa, ob = a(imgs)[0], b(imgs)[0]
    assert torch.equal(oa["boxes"], ob["boxes"]) and torch.equal(oa["scores"], ob["scores"])


def test_guardar_y_cargar_conserva_salidas(tmp_path):
    imgs, _ = _lote()
    a = _modelo(2).eval()
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b = _modelo(99).eval()
    b.load_state_dict(torch.load(tmp_path / "m.pt"))
    with torch.no_grad():
        oa, ob = a(imgs)[0], b(imgs)[0]
    assert torch.allclose(oa["scores"], ob["scores"]) and torch.allclose(oa["boxes"], ob["boxes"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sobreajustar un lote es lento en CPU")
def test_sobreajusta_lote_pequeno():
    disp = torch.device("cuda")
    m = _modelo(0).to(disp).train()
    imgs, tgts = _lote()
    imgs = [i.to(disp) for i in imgs]
    tgts = [{k: v.to(disp) for k, v in t.items()} for t in tgts]
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01, momentum=0.9)
    primera = None
    for _ in range(150):
        opt.zero_grad()
        perdida = sum(m(imgs, tgts).values())
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)  # red sin preentrenar: sin recorte diverge (también con IoU)
        opt.step()
        primera = primera if primera is not None else perdida.item()
    assert perdida.item() < 0.25 * primera
