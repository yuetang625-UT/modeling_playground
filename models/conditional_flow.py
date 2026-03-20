# Flow-based local sequence editor.
# Frozen backbone encodes the full sequence; a small denoiser learns to
# predict velocity in hidden-state space conditioned on motif + flanking context.
#
# This is a prototype -- backbone is swapped via duck typing (see SequenceBackbone).

import math
from typing import Dict, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceBackbone(Protocol):
    # anything that can encode, embed tokens, and decode logits works here
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor: ...
    def token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor: ...
    def decode_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half < 1:
            raise ValueError("dim must be >= 2")
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class FlowDenoiser(nn.Module):
    """
    Transformer denoiser over a local window of hidden states.
    Conditions on motif + left/right context + optional global context.
    """

    def __init__(self, hidden_dim=64, num_heads=4, num_layers=3, ff_dim=128, dropout=0.1, use_global_context=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_global_context = use_global_context

        self.time_emb = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        n_cond = 4 if use_global_context else 3
        self.cond_proj = nn.Linear(hidden_dim * n_cond, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, t, motif_emb, left_ctx_emb, right_ctx_emb, global_emb=None):
        t_emb = self.time_proj(self.time_emb(t))
        x = x_t + t_emb.unsqueeze(1)

        cond_parts = [motif_emb, left_ctx_emb, right_ctx_emb]
        if self.use_global_context:
            if global_emb is None:
                raise ValueError("global_emb required when use_global_context=True")
            cond_parts.append(global_emb)

        # prepend a single conditioning token then strip it after the encoder
        cond = self.cond_proj(torch.cat(cond_parts, dim=-1)).unsqueeze(1)
        x = torch.cat([cond, x], dim=1)
        x = self.encoder(x)
        x = x[:, 1:, :]
        return self.out(x)


class ConditionalFlowModel(nn.Module):
    """
    Local window generator: frozen backbone + trainable denoiser.

    Training: flow matching loss in hidden-state space + optional reconstruction.
    Inference: Euler integration from noise to predicted hidden states, then decode.

    # TODO: handle variable window lengths per sample (currently pads to max in batch)
    """

    def __init__(self, backbone: SequenceBackbone, denoiser: FlowDenoiser):
        super().__init__()
        self.backbone = backbone
        self.denoiser = denoiser

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        hidden_dim = denoiser.hidden_dim
        self.hidden_adapter = nn.Linear(hidden_dim, hidden_dim)

    @torch.no_grad()
    def encode_sequence(self, token_ids):
        return self.backbone.encode(token_ids)

    @staticmethod
    def _extract_window(hidden, start, end, window_len):
        B, _, D = hidden.shape
        out = torch.zeros(B, window_len, D, device=hidden.device, dtype=hidden.dtype)
        for b in range(B):
            s, e = int(start[b].item()), int(end[b].item())
            n = min(max(e - s, 0), window_len)
            if n > 0:
                out[b, :n] = hidden[b, s : s + n]
        return out

    @staticmethod
    def _mean_pool_region(hidden, start, end):
        B, L, D = hidden.shape
        out = torch.zeros(B, D, device=hidden.device, dtype=hidden.dtype)
        for b in range(B):
            s = max(0, min(int(start[b].item()), L))
            e = max(0, min(int(end[b].item()), L))
            if e > s:
                out[b] = hidden[b, s:e].mean(dim=0)
        return out

    @torch.no_grad()
    def pool_motif(self, motif_ids, motif_len):
        emb = self.backbone.token_embedding(motif_ids)
        B, max_len, D = emb.shape
        mask = (torch.arange(max_len, device=motif_ids.device).unsqueeze(0) < motif_len.unsqueeze(1)).float()
        emb = emb * mask.unsqueeze(-1)
        return emb.sum(dim=1) / motif_len.unsqueeze(1).float().clamp(min=1.0)

    def forward(self, batch: Dict[str, torch.Tensor], recon_weight=0.1):
        device = next(self.denoiser.parameters()).device

        token_ids = batch["token_ids"].to(device)
        motif_ids = batch["motif_ids"].to(device)
        motif_len = batch["motif_len"].to(device)
        window_start = batch["window_start"].to(device)
        window_end = batch["window_end"].to(device)

        B, L = token_ids.shape
        hidden = self.encode_sequence(token_ids)
        window_len = int((window_end - window_start).max().item())

        x_1 = self._extract_window(hidden, window_start, window_end, window_len)

        actual_len = (window_end - window_start).clamp(min=0, max=window_len)
        win_mask = torch.arange(window_len, device=device).unsqueeze(0) < actual_len.unsqueeze(1)

        motif_emb = self.pool_motif(motif_ids, motif_len)
        left_ctx = self._mean_pool_region(hidden, torch.zeros_like(window_start), window_start)
        right_ctx = self._mean_pool_region(hidden, window_end, torch.full_like(window_end, L))
        global_ctx = hidden.mean(dim=1)

        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        target_v = x_1 - x_0

        pred_v = self.denoiser(
            x_t=x_t, t=t,
            motif_emb=motif_emb,
            left_ctx_emb=left_ctx,
            right_ctx_emb=right_ctx,
            global_emb=global_ctx,
        )

        diff = (pred_v - target_v) ** 2
        valid = win_mask.unsqueeze(-1).float()
        velocity_loss = (diff * valid).sum() / (valid.sum().clamp(min=1.0) * x_1.size(-1))

        # reconstruction loss on the window tokens
        pred_x1 = x_t + (1 - t[:, None, None]) * pred_v
        logits = self.backbone.decode_logits(self.hidden_adapter(pred_x1))

        _, W, V = logits.shape
        labels = torch.full((B, W), -100, dtype=torch.long, device=device)
        for b in range(B):
            s, e = int(window_start[b].item()), int(window_end[b].item())
            n = min(max(e - s, 0), W)
            if n > 0:
                labels[b, :n] = token_ids[b, s : s + n]
        labels[~win_mask] = -100

        recon_loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
        return velocity_loss + recon_weight * recon_loss

    @torch.no_grad()
    def generate_local_window(self, token_ids, motif_ids, motif_len, window_start, window_end, num_steps=50):
        """Euler integration from noise -> decode window tokens."""
        device = token_ids.device
        hidden = self.encode_sequence(token_ids)

        motif_emb = self.pool_motif(motif_ids, motif_len)
        left_ctx = self._mean_pool_region(hidden, torch.zeros_like(window_start), window_start)
        right_ctx = self._mean_pool_region(hidden, window_end, torch.full_like(window_end, token_ids.size(1)))
        global_ctx = hidden.mean(dim=1)

        W = int((window_end - window_start).max().item())
        B, _, D = hidden.shape
        x = torch.randn(B, W, D, device=device)

        dt = 1.0 / max(num_steps, 1)
        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device)
            v = self.denoiser(
                x_t=x, t=t,
                motif_emb=motif_emb,
                left_ctx_emb=left_ctx,
                right_ctx_emb=right_ctx,
                global_emb=global_ctx,
            )
            x = x + dt * v

        logits = self.backbone.decode_logits(self.hidden_adapter(x))
        return logits.argmax(dim=-1)
