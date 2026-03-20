import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.conditional_flow import ConditionalFlowModel, FlowDenoiser
from models.mock_mlm import MockSequenceBackbone
from utils.simple_tokenizer import encode_sequence, pad_batch

torch.manual_seed(0)


def _make_model(hidden_dim=32):
    backbone = MockSequenceBackbone(vocab_size=8, hidden_dim=hidden_dim, num_layers=1, num_heads=4)
    denoiser = FlowDenoiser(hidden_dim=hidden_dim, num_heads=4, num_layers=1, ff_dim=64, dropout=0.0)
    return ConditionalFlowModel(backbone=backbone, denoiser=denoiser)


def _make_batch():
    seqs = ["AUGCAUG", "GCAUGCA"]
    motifs = ["AUG", "GCA"]
    token_ids = pad_batch([encode_sequence(s) for s in seqs])
    motif_ids = pad_batch([encode_sequence(m, add_special_tokens=False) for m in motifs], pad_id=0)
    motif_len = torch.tensor([3, 3], dtype=torch.long)
    window_start = torch.tensor([2, 3], dtype=torch.long)
    window_end = torch.tensor([6, 7], dtype=torch.long)
    return {
        "token_ids": token_ids,
        "motif_ids": motif_ids,
        "motif_len": motif_len,
        "window_start": window_start,
        "window_end": window_end,
    }


def test_training_loss_is_scalar():
    model = _make_model()
    loss = model(_make_batch())
    assert loss.shape == ()
    assert loss.item() > 0


def test_backbone_frozen():
    model = _make_model()
    for p in model.backbone.parameters():
        assert not p.requires_grad


def test_generate_output_shape():
    model = _make_model()
    batch = _make_batch()
    out = model.generate_local_window(
        batch["token_ids"],
        batch["motif_ids"],
        batch["motif_len"],
        batch["window_start"],
        batch["window_end"],
        num_steps=3,
    )
    # window width is max(window_end - window_start) = 4
    assert out.shape == (2, 4)
    assert out.dtype == torch.long


def test_generate_tokens_in_vocab():
    model = _make_model()
    batch = _make_batch()
    out = model.generate_local_window(
        batch["token_ids"], batch["motif_ids"], batch["motif_len"],
        batch["window_start"], batch["window_end"], num_steps=3,
    )
    assert out.min().item() >= 0
    assert out.max().item() < 8  # vocab_size
