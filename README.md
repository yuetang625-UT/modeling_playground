# Sequence Modeling Prototypes

Personal practice and test code for two sequence-modeling patterns:

1. **Multimodal sequence-level regression** — a CNN-based predictor with gated modality attention
2. **Conditional local sequence editing** — a small denoising model operating on frozen representations

Both use toy inputs and a mock backbone. No proprietary code or data.

## Layout

```
models/multimodal_predictor.py      — multimodal CNN regressor
models/conditional_flow.py          — conditional local editor
models/mock_mlm.py                  — mock backbone
utils/simple_tokenizer.py           — toy nucleotide tokenizer
examples/run_multimodal_demo.py
examples/run_conditional_flow_demo.py
```

## Run

```bash
pip install -r requirements.txt
python examples/run_multimodal_demo.py
python examples/run_conditional_flow_demo.py
```

## Swap in your own backbone

Any model implementing the three methods below works as a drop-in:

```python
class MyBackbone(nn.Module):
    def encode(self, token_ids): ...
    def token_embedding(self, token_ids): ...
    def decode_logits(self, hidden_states): ...
```
