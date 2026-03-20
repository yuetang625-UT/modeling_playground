import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.multimodal_predictor import MultimodalPredictor, masked_mse_loss
from utils.simple_tokenizer import one_hot_encode_rna

torch.manual_seed(7)

sequences = [
    "AUGCAUGCAUGCAUGC",
    "GGGAAAUUCCGAUGCA",
    "AUAUAUGCGCGCAAUU",
    "CCGAUGAUGGCUUAAA",
]

seq_batch = torch.stack([one_hot_encode_rna(s) for s in sequences])  # [B, 4, L]
B, _, L = seq_batch.shape

structure = torch.rand(B, 1, L)
signal = torch.zeros(B, 2, L)
signal[:, 0] = torch.rand(B, L)
signal[:, 1] = (torch.rand(B, L) > 0.25).float()
factors = (torch.rand(B, 12, L) > 0.85).float()

model = MultimodalPredictor({
    "d_seq": 32,
    "d_factor": 16,
    "n_factors": 12,
    "n_filters": 64,
    "dropout": 0.1,
    "num_targets": 2,
    "use_structure": True,
    "use_signal": True,
    "use_factors": True,
})

out = model(sequence=seq_batch, structure=structure, signal=signal, factors=factors)
pred, alphas = out["pred"], out["alphas"]

target = torch.randn_like(pred)
loss = masked_mse_loss(pred, target) + model.entropy_loss(alphas)

print("pred shape:", tuple(pred.shape))
print("modalities:", model.modality_names)
print("attention (sample 0):", alphas[0].detach().tolist())
print("loss:", float(loss.detach()))
