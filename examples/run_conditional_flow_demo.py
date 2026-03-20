import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.conditional_flow import ConditionalFlowModel, FlowDenoiser
from models.mock_mlm import MockSequenceBackbone
from utils.simple_tokenizer import decode_ids, encode_sequence, normalize_sequence, pad_batch

torch.manual_seed(11)

sequences = ["AUGCAUGGCAUUGCAU", "GGCAUUAACCGGAUUA", "AUAUGCGCGAAAUUCG"]
motifs = ["AUG", "GGA", "UUC"]

token_ids = pad_batch([encode_sequence(s) for s in sequences])
motif_ids = pad_batch([encode_sequence(m, add_special_tokens=False) for m in motifs], pad_id=0)
motif_len = torch.tensor([len(normalize_sequence(m)) for m in motifs], dtype=torch.long)

window_start = torch.tensor([4, 5, 3], dtype=torch.long)
window_end   = torch.tensor([8, 9, 7], dtype=torch.long)

backbone = MockSequenceBackbone(vocab_size=8, hidden_dim=32, num_layers=1, num_heads=4)
denoiser = FlowDenoiser(hidden_dim=32, num_heads=4, num_layers=1, ff_dim=64, dropout=0.1)
model = ConditionalFlowModel(backbone=backbone, denoiser=denoiser)

batch = {
    "token_ids": token_ids,
    "motif_ids": motif_ids,
    "motif_len": motif_len,
    "window_start": window_start,
    "window_end": window_end,
}

loss = model(batch, recon_weight=0.2)
generated = model.generate_local_window(
    token_ids, motif_ids, motif_len,
    window_start, window_end,
    num_steps=8,
)

print("train loss:", float(loss.detach()))
print("generated shape:", tuple(generated.shape))
print("sample 0:", decode_ids(generated[0].tolist()))
