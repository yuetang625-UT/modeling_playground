import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.multimodal_predictor import MultimodalPredictor, masked_mse_loss

B, L = 3, 20

BASE_CFG = {
    "d_seq": 16,
    "d_factor": 8,
    "n_factors": 6,
    "n_filters": 32,
    "dropout": 0.0,
    "num_targets": 2,
}


def _seq(b=B, l=L):
    x = torch.zeros(b, 4, l)
    x[:, 0] = 1.0
    return x


def test_output_shapes_sequence_only():
    model = MultimodalPredictor(BASE_CFG)
    out = model(sequence=_seq())
    assert out["pred"].shape == (B, 2)
    assert out["alphas"].shape == (B, 1)


def test_output_shapes_all_modalities():
    cfg = {**BASE_CFG, "use_structure": True, "use_signal": True, "use_factors": True}
    model = MultimodalPredictor(cfg)
    out = model(
        sequence=_seq(),
        structure=torch.rand(B, 1, L),
        signal=torch.rand(B, 2, L),
        factors=(torch.rand(B, 6, L) > 0.5).float(),
    )
    assert out["pred"].shape == (B, 2)
    assert out["alphas"].shape == (B, 4)


def test_missing_optional_inputs_filled_with_zeros():
    cfg = {**BASE_CFG, "use_structure": True, "use_signal": True, "use_factors": True}
    model = MultimodalPredictor(cfg)
    # passing None explicitly should not crash
    out = model(sequence=_seq(), structure=None, signal=None, factors=None)
    assert out["pred"].shape == (B, 2)


def test_masked_mse_no_mask():
    pred = torch.ones(4, 3)
    target = torch.zeros(4, 3)
    loss = masked_mse_loss(pred, target)
    assert loss.item() == 1.0


def test_masked_mse_with_mask():
    pred = torch.ones(4, 2)
    target = torch.zeros(4, 2)
    mask = torch.tensor([1, 0, 1, 0], dtype=torch.float)
    loss = masked_mse_loss(pred, target, mask)
    # denom = mask.sum() = 2, numerator = 2 valid rows * 2 cols * 1.0 = 4 -> loss = 2.0
    assert loss.item() == 2.0


def test_entropy_loss_is_scalar():
    cfg = {**BASE_CFG, "use_structure": True}
    model = MultimodalPredictor(cfg)
    out = model(sequence=_seq())
    loss = model.entropy_loss(out["alphas"])
    assert loss.shape == ()


def test_modality_names():
    cfg = {**BASE_CFG, "use_structure": True, "use_signal": True}
    model = MultimodalPredictor(cfg)
    assert model.modality_names == ["sequence", "structure", "signal"]
