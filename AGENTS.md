# AGENTS.md — WestQuant Open Source

## Project

Two parallel projects sharing one core:
1. **WQT20** — ~20M-parameter open-source Transformer for quantum representation scheduling
2. **WestQuant Ecosystem** — WQIR, transformation registry, verifier, RepGraph, SDK, benchmarks

Principle: **AI schedules. Deterministic mathematics executes. Independent verification certifies.**

## Environment

- macOS Apple Silicon (M5 Max), 128GB RAM
- MPS available; device priority: MPS → CPU (no CUDA assumption)
- Python 3.10, venv at `.venv/`
- Activate: `source .venv/bin/activate`
- Deps: torch, transformers, tokenizers, safetensors, numpy<2, pyyaml, tqdm, pytest

## Commands

```bash
# Tests
./scripts/run_tests.sh
# or: python -m pytest wqt20/tests/ -v

# Build tokenizer
./scripts/build_tokenizer.sh

# Smoke training (H1 gate: beats random)
./scripts/run_smoke.sh
# or: python -m wqt20.train --n 3000 --epochs 8 --batch-size 64 --lr 1e-4

# Inspect model
python -m wqt20.model
```

## Architecture conventions

- WQT20 uses standard HF `LlamaConfig` + `LlamaForCausalLM` (no exotic architecture in v0.1)
- Tokenizer: ~4561 tokens = atomic structured tokens (un-splittable added tokens) + byte-level BPE fallback
- Model: 19.06M params, 8 layers, hidden 384, 6 heads, 2 KV heads, FFN 1536, tied embeddings
- Vocab size for the model must use `len(tokenizer)` (includes added tokens), NOT `tokenizer.vocab_size`

## Code conventions

- Ecosystem code (WQIR, actions, verify, repgraph, sdk) is dependency-free (numpy only)
- WQT20 model code depends on transformers + torch
- Data generators are seedable and deterministic
- Provenance ledger is JSONL (append-only)
- Tests in `wqt20/tests/` — run with pytest

## Key design rules

- The model NEVER executes transformations; it only ranks legal actions
- The registry produces the legal action set — invalid actions are structurally impossible
- Every equivalence-preserving transformation must be independently verified
- Failed/dead-end trajectories are stored (not just winners)
- Structural held-out splits (not random rows) for evaluation

## Implementation order (Part XIX)

1. WQIR schema ✓
2. Transformation registry ✓
3. Deterministic verifier ✓
4. RepGraph format ✓
5. Benchmark harness ✓
6. Dataset generators ✓
7. Baseline search ✓
8. WQT20 tokenizer ✓
9. Toy Transformer ✓
10. Smoke dataset ✓
11. **Verify learned policy beats random** ✓ (H1 gate PASSED)
12. Scale to 20M ✓ (19.06M)
13. Generate oracle/search trajectories — next
14. Train WQT20-PILOT — pending
15. Freeze evaluation methodology — pending
16. Train WQT20-1.0 — pending
17. Integrate Qiskit + TKET — pending
18. Open release — pending

## Version history

- v0.1.0: initial scaffold + smoke training (H1 gate passed)
