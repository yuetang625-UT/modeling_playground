# small mock backbone for testing ConditionalFlowModel without a real pretrained model

import torch
import torch.nn as nn


class MockSequenceBackbone(nn.Module):
    def __init__(self, vocab_size=8, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def token_embedding(self, token_ids):
        return self.embedding(token_ids)

    def encode(self, token_ids):
        return self.encoder(self.embedding(token_ids))

    def decode_logits(self, hidden_states):
        return self.lm_head(hidden_states)
