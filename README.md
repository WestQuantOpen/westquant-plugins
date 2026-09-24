# WestQuant Open Source

**WQT20** — an approximately 20M-parameter open-source Transformer specialized in
quantum representation scheduling. **WestQuant Ecosystem** — the WQIR schema,
transformation registry, deterministic verifier, Representation Graph, SDK,
and benchmarks that surround the model.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

WQT20 is a compact learned policy over a structured quantum representation
space. It ranks which transformation is promising. The ecosystem executes it.
Verification decides what survives. The model is **not** a chatbot, a
code-generator, or a replacement for Qiskit/TKET/PyZX.

## Status

- [x] WQIR schema (12 levels)
- [x] Transformation registry (31 actions, 12 families)
- [x] Deterministic verifier (Pauli algebra, commutation, grouping, Trotter depth)
- [x] Representation Graph storage (JSONL)
- [x] WQT20 tokenizer (~4561 tokens, quantum-native structured + BPE)
- [x] WQT20 model (19.06M params, Llama-style, HF-compatible)
- [x] Data generators (graph Hamiltonians: MaxCut, MWIS, TFIM)
- [x] Baseline search (random, fixed, greedy, frequency)
- [x] WQT20-guided policy search
- [x] Eval harness (recall@k, regret, Pareto, structural splits)
- [x] Smoke training — **H1 gate PASSED** (recall@1: 0.967 vs random 0.123)
- [ ] Pilot training (50k trajectories)
- [ ] WQT20-1.0
- [ ] Qiskit/TKET integration
- [ ] Open release

## Quick start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers tokenizers safetensors numpy pyyaml tqdm pytest

# Run tests
./scripts/run_tests.sh

# Build tokenizer
./scripts/build_tokenizer.sh

# Smoke training (validates H1: learned policy beats random)
./scripts/run_smoke.sh
```

## Architecture

```
westquant/
├── wqt20/                 # the model
│   ├── tokenizer.py        # ~4561 BPE, quantum-native structured tokens
│   ├── model.py            # 19M Llama-style, HF LlamaConfig, tied embeddings
│   ├── train.py            # smoke → pilot → 1.0
│   ├── data/               # generators, trajectories, provenance
│   ├── search/             # baselines + WQT20-guided policy
│   ├── eval/               # recall@k, regret, pareto, structural splits
│   └── tests/
├── ecosystem/             # the distribution machine
│   ├── wqir.py             # WestQuant IR schema (12 levels)
│   ├── actions.py          # transformation registry (31 actions)
│   ├── repgraph.py         # Representation Graph storage
│   ├── verify.py           # deterministic verifier API
│   └── sdk.py              # westquant.Search(...) public API
├── configs/                # smoke.yaml, pilot.yaml
├── scripts/                # build_tokenizer, run_smoke, run_tests
└── MEGAPLAN.md
```

## Model

| Config | Value |
|--------|-------|
| Architecture | decoder-only Llama-style |
| Layers | 8 |
| Hidden | 384 |
| Attention heads | 6 |
| KV heads | 2 (GQA) |
| FFN | 1536 (SwiGLU) |
| Norm | RMSNorm |
| Position | RoPE |
| Vocab | ~4561 (structured + BPE) |
| Context | 2048 |
| Embeddings | tied |
| **Parameters** | **19.06M** |

Standard HuggingFace `LlamaConfig` + `LlamaForCausalLM` — compatible with
SafeTensors, `AutoModelForCausalLM`, `AutoTokenizer`, and GGUF conversion.

## Research hypotheses

- **H1** Small specialized models learn quantum-representation policies ✓ (smoke)
- **H2** Representation scheduling generalizes (size / instance / topology)
- **H3** Mathematical representation matters independently of gate-level opt
- **H4** Hardware changes the optimal representation path
- **H5** Search efficiency > action-imitation accuracy

## License

Apache-2.0
