# Multimodal sequence predictor -- gated attention over multiple input tracks.
# Sequence (one-hot CNN) + optional structure/signal/factor channels -> regression targets.
#
# Designed for RNA-like sequences; intentionally small so it's easy to swap parts.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse_loss(pred, target, mask=None):
    loss = (pred - target) ** 2
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


class SequenceCNN(nn.Module):
    # one-hot [B, C, L] -> per-position features [B, D, L]
    def __init__(self, in_channels=4, d_seq=64, n_layers=3, kernel_size=7):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(ch, d_seq, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(d_seq),
                nn.ReLU(inplace=True),
            ]
            ch = d_seq
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BinaryFactorEmbedder(nn.Module):
    """
    Binary factor map [B, N, L] -> [B, D+1, L].
    Last channel is log count so the model knows how many factors are active.
    """

    def __init__(self, n_factors, d_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_factors, d_factor) * 0.02)

    def forward(self, x):
        n_active = x.sum(dim=1, keepdim=True)  # [B, 1, L]
        # weighted mean of active factor embeddings
        mean_emb = torch.einsum("bnl,nd->bld", x, self.weight).permute(0, 2, 1)
        mean_emb = mean_emb / n_active.clamp(min=1.0)
        log_count = (n_active + 1.0).log()
        return torch.cat([mean_emb, log_count], dim=1)

    @torch.no_grad()
    def factor_importance(self):
        return self.weight.norm(dim=1)


class GatedModalityAttention(nn.Module):
    """
    Soft attention over modality pooled summaries using tanh/sigmoid gating.
    entropy_reg penalizes collapsed attention (all weight on one modality).
    """

    def __init__(self, d_modalities: List[int], d_attn=64, entropy_reg=0.05):
        super().__init__()
        self.entropy_reg = entropy_reg
        self.V = nn.ModuleList([nn.Linear(2 * d, d_attn, bias=True) for d in d_modalities])
        self.U = nn.ModuleList([nn.Linear(2 * d, d_attn, bias=False) for d in d_modalities])
        self.w = nn.Parameter(torch.randn(d_attn) * 0.02)

    def forward(self, feats: List[torch.Tensor]):
        scores = []
        for i, f in enumerate(feats):
            h = torch.cat([f.mean(dim=-1), f.max(dim=-1).values], dim=-1)
            g = torch.tanh(self.V[i](h)) * torch.sigmoid(self.U[i](h))
            scores.append((g * self.w).sum(dim=-1, keepdim=True))

        alphas = torch.softmax(torch.cat(scores, dim=1), dim=1)
        weighted_feats = [feats[i] * alphas[:, i : i + 1, None] for i in range(len(feats))]
        return weighted_feats, alphas

    def entropy_loss(self, alphas):
        entropy = -(alphas * alphas.clamp(min=1e-8).log()).sum(dim=1).mean()
        return -self.entropy_reg * entropy


class TextCNNHead(nn.Module):
    # multi-kernel conv + global max pool, standard TextCNN setup
    def __init__(self, in_channels, kernel_sizes=(3, 5, 9), n_filters=128, dropout=0.2, num_targets=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, n_filters, ks, padding=ks // 2),
                nn.ReLU(inplace=True),
            )
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), num_targets)

    def forward(self, x):
        pooled = [conv(x).max(dim=-1).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class MultimodalPredictor(nn.Module):
    """
    Sequence-level regressor fusing multiple per-position tracks.

    Inputs (sequence is always required):
      sequence  [B, 4, L]   one-hot RNA
      structure [B, 1, L]   optional (e.g. SHAPE reactivity)
      signal    [B, 2, L]   optional aux channels
      factors   [B, N, L]   optional binary feature map

    Output dict has 'pred' [B, T] and 'alphas' [B, M].
    """

    def __init__(self, config: Dict):
        super().__init__()

        d_seq = config.get("d_seq", 64)
        d_factor = config.get("d_factor", 32)
        n_factors = config.get("n_factors", 64)
        n_filters = config.get("n_filters", 128)
        dropout = config.get("dropout", 0.2)
        entropy_reg = config.get("entropy_reg", 0.05)
        num_targets = config.get("num_targets", 1)

        self.use_structure = config.get("use_structure", False)
        self.use_signal = config.get("use_signal", False)
        self.use_factors = config.get("use_factors", False)

        self.sequence_encoder = SequenceCNN(in_channels=4, d_seq=d_seq)

        if self.use_structure:
            self.structure_norm = nn.BatchNorm1d(1)
        if self.use_factors:
            self.factor_encoder = BinaryFactorEmbedder(n_factors=n_factors, d_factor=d_factor)

        self._modality_names: List[str] = ["sequence"]
        self._modality_dims: List[int] = [d_seq]
        total_channels = d_seq

        if self.use_structure:
            self._modality_names.append("structure")
            self._modality_dims.append(1)
            total_channels += 1

        if self.use_signal:
            self._modality_names.append("signal")
            self._modality_dims.append(2)
            total_channels += 2

        if self.use_factors:
            self._modality_names.append("factors")
            self._modality_dims.append(d_factor + 1)
            total_channels += d_factor + 1

        self.gated_attention = GatedModalityAttention(self._modality_dims, d_attn=64, entropy_reg=entropy_reg)
        self.head = TextCNNHead(
            in_channels=total_channels,
            kernel_sizes=(3, 5, 9),
            n_filters=n_filters,
            dropout=dropout,
            num_targets=num_targets,
        )

    def forward(self, sequence, structure=None, signal=None, factors=None):
        if sequence is None:
            raise ValueError("sequence is required (one-hot, [B, 4, L])")

        B, _, L = sequence.shape
        feats: List[torch.Tensor] = [self.sequence_encoder(sequence)]

        if self.use_structure:
            if structure is None:
                structure = torch.zeros(B, 1, L, device=sequence.device, dtype=sequence.dtype)
            feats.append(self.structure_norm(structure))

        if self.use_signal:
            if signal is None:
                signal = torch.zeros(B, 2, L, device=sequence.device, dtype=sequence.dtype)
            feats.append(signal)

        if self.use_factors:
            if factors is None:
                n = self.factor_encoder.weight.shape[0]
                factors = torch.zeros(B, n, L, device=sequence.device, dtype=sequence.dtype)
            feats.append(self.factor_encoder(factors))

        weighted_feats, alphas = self.gated_attention(feats)
        fused = torch.cat(weighted_feats, dim=1)
        pred = self.head(fused)

        return {"pred": pred, "alphas": alphas}

    def entropy_loss(self, alphas):
        return self.gated_attention.entropy_loss(alphas)

    @property
    def modality_names(self):
        return list(self._modality_names)

    @torch.no_grad()
    def factor_importance(self):
        if not self.use_factors:
            raise RuntimeError("Factor channel not enabled.")
        return self.factor_encoder.factor_importance()
