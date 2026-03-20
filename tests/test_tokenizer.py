import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from utils.simple_tokenizer import (
    decode_ids,
    encode_sequence,
    normalize_sequence,
    one_hot_encode_rna,
    pad_batch,
)


def test_normalize_dna_to_rna():
    assert normalize_sequence("augcat") == "AUGCAU"


def test_encode_decode_roundtrip():
    seq = "AUGCGU"
    ids = encode_sequence(seq, add_special_tokens=False)
    assert decode_ids(ids) == seq


def test_encode_adds_special_tokens():
    ids = encode_sequence("AUG", add_special_tokens=True)
    # should start with BOS and end with EOS
    assert ids[0] == 1   # BOS
    assert ids[-1] == 2  # EOS
    assert len(ids) == 5


def test_unknown_base_maps_to_N():
    ids = encode_sequence("AXG", add_special_tokens=False)
    decoded = decode_ids(ids)
    assert decoded == "ANG"


def test_pad_batch_shape():
    seqs = ["AUG", "AUGCGU", "A"]
    id_lists = [encode_sequence(s, add_special_tokens=False) for s in seqs]
    batch = pad_batch(id_lists)
    assert batch.shape == (3, 6)
    assert batch[2, 1] == 0  # padded positions


def test_one_hot_shape_and_values():
    t = one_hot_encode_rna("AUCG")
    assert t.shape == (4, 4)
    # each position should sum to 1
    assert torch.allclose(t.sum(dim=0), torch.ones(4))


def test_one_hot_unknown_base_all_zero():
    t = one_hot_encode_rna("N")
    assert t.sum().item() == 0.0
