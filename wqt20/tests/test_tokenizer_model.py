"""Tests for the WQT20 tokenizer and model."""
import pytest
from wqt20.tokenizer import build_tokenizer, build_atomic_vocab, ATOMIC_VOCAB
from wqt20.model import build_model, build_config, count_parameters


def test_atomic_vocab_nonempty():
    vocab = build_atomic_vocab()
    assert len(vocab) > 500
    # special tokens ARE included in the atomic vocab
    assert "<pad>" in vocab
    assert "<A:GROUP_PAULIS>" in vocab
    assert "<LEVEL:HAMILTONIAN>" in vocab
    assert "<H:Ising>" in vocab


def test_tokenizer_builds_and_encodes():
    tok = build_tokenizer(vocab_size=4096)
    vs = tok.get_vocab_size()
    assert vs >= 4096
    enc = tok.encode("<STATE> <LEVEL:HAMILTONIAN> <A:GROUP_PAULIS>")
    ids = enc.ids
    assert len(ids) > 0
    # atomic tokens should be single tokens
    tid = tok.token_to_id("<A:GROUP_PAULIS>")
    assert tid is not None and tid >= 0


def test_model_param_count():
    cfg = build_config(vocab_size=4096)
    model = build_model(vocab_size=4096)
    info = count_parameters(model)
    assert 15e6 < info["total"] < 25e6, f"params {info['total_M']:.2f}M out of range"
    assert cfg.num_hidden_layers == 8
    assert cfg.hidden_size == 384
    assert cfg.num_attention_heads == 6
    assert cfg.num_key_value_heads == 2
    assert cfg.intermediate_size == 1536
    assert cfg.tie_word_embeddings is True
