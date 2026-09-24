"""WQT20 model: ~20M Llama-style causal Transformer, HF-compatible.

Uses standard HuggingFace LlamaConfig + LlamaForCausalLM so the model is
automatically compatible with SafeTensors, transformers, vLLM, and GGUF
conversion. No exotic architecture — the novelty is in the tokenizer + data.

Config (Part II):
  layers=8, hidden=384, heads=6, KV heads=2, FFN=1536, SwiGLU, RMSNorm,
  RoPE, tied embeddings, context 2048. ~18.9M params at vocab 4096.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from .tokenizer import build_tokenizer, load_tokenizer, build_atomic_vocab

# ---------------------------------------------------------------------------
# Default config (Part II)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    hidden_size=384,
    intermediate_size=1536,
    num_hidden_layers=8,
    num_attention_heads=6,
    num_key_value_heads=2,
    hidden_act="silu",          # Llama implements SwiGLU via silu(gate)*up
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    rope_theta=10000.0,
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)


def build_config(vocab_size: int) -> LlamaConfig:
    cfg = LlamaConfig(vocab_size=vocab_size, **DEFAULT_CONFIG)
    return cfg


def build_model(vocab_size: int = 4096) -> LlamaForCausalLM:
    """Build a fresh WQT20 model from config."""
    cfg = build_config(vocab_size)
    model = LlamaForCausalLM(cfg)
    return model


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_M": total / 1e6}


def build_tokenizer_hf(tokenizer_dir: str | Path | None = None) -> PreTrainedTokenizerFast:
    """Build or load the WQT20 tokenizer as an HF PreTrainedTokenizerFast."""
    from tokenizers import Tokenizer
    if tokenizer_dir is not None and (Path(tokenizer_dir) / "tokenizer.json").exists():
        tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    else:
        tok = build_tokenizer(vocab_size=4096, save_dir=Path(__file__).parent / "tokenizer")
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        sep_token="<sep>", mask_token="<mask>",
    )
    return hf_tok


def save_model(model: LlamaForCausalLM, tokenizer: PreTrainedTokenizerFast, out_dir: str | Path) -> None:
    """Save in HF format: config.json, model.safetensors, tokenizer files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)


def load_model(model_dir: str | Path, device: str = "auto") -> tuple:
    """Load a saved WQT20 model + tokenizer."""
    import torch
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


if __name__ == "__main__":
    tok = build_tokenizer_hf()
    vs = len(tok)  # includes added atomic tokens
    print(f"tokenizer vocab: {vs}")
    model = build_model(vocab_size=vs)
    info = count_parameters(model)
    print(f"params: {info['total_M']:.2f}M (trainable {info['trainable']/1e6:.2f}M)")
    cfg = model.config
    print(f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads} ffn={cfg.intermediate_size}")
