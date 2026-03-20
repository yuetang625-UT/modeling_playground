# toy tokenizer for RNA-like sequences (A/C/G/U + specials)

from typing import Dict, Iterable, List

import torch

TOKENS = ["<pad>", "<bos>", "<eos>", "A", "C", "G", "U", "N"]
TOKEN_TO_ID: Dict[str, int] = {token: idx for idx, token in enumerate(TOKENS)}
ID_TO_TOKEN: Dict[int, str] = {idx: token for token, idx in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]


def normalize_sequence(seq: str) -> str:
    return seq.upper().replace("T", "U")


def encode_sequence(seq: str, add_special_tokens=True) -> List[int]:
    seq = normalize_sequence(seq)
    ids = [TOKEN_TO_ID.get(ch, TOKEN_TO_ID["N"]) for ch in seq]
    if add_special_tokens:
        ids = [BOS_ID] + ids + [EOS_ID]
    return ids


def decode_ids(ids: Iterable[int], skip_special_tokens=True) -> str:
    chars = []
    for idx in ids:
        token = ID_TO_TOKEN.get(int(idx), "N")
        if skip_special_tokens and token.startswith("<"):
            continue
        chars.append(token)
    return "".join(chars)


def pad_batch(id_lists: List[List[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    max_len = max(len(ids) for ids in id_lists)
    batch = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def one_hot_encode_rna(seq: str) -> torch.Tensor:
    # returns [4, L], channels are A/C/G/U; unknown bases all-zero
    seq = normalize_sequence(seq)
    ch_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    out = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, base in enumerate(seq):
        if base in ch_map:
            out[ch_map[base], i] = 1.0
    return out
