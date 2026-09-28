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

## Public vs Private

This repo has two layers:

- **`plugins/`** — public. Pushed to [github.com/WestQuantOpen](https://github.com/WestQuantOpen).
  Contains Qiskit, TKET, PyZX adapters and the WestQuant SDK. Links to the WQT20
  model on HuggingFace.
- **`wqt20/`, `ecosystem/`, `data/`, `configs/`, `scripts/`** — private. Stays
  local. Contains model code, training infrastructure, generated data, and
  checkpoints. Not pushed to public GitHub.

The `.gitignore` excludes all private directories. Only `plugins/` and
public-facing docs are committed to the public repo.

## Status

- [x] WQIR schema (12 levels)
- [x] Transformation registry (31 actions, 12 families)
- [x] Deterministic verifier (Pauli algebra, commutation, grouping, Trotter depth)
- [x] Representation Graph storage (JSONL)
- [x] WQT20 tokenizer (~4561 tokens, quantum-native structured + BPE)
- [x] WQT20 model (19.06M params, Llama-style, HF-compatible)
- [x] Data generators (12 curriculum domains, A-L)
- [x] **1M training examples generated** (50s, 20K/s, 1.1GB)
- [x] Baseline search (random, fixed, greedy, frequency)
- [x] WQT20-guided policy search
- [x] Eval harness (recall@k, regret, Pareto, structural splits)
- [x] Smoke training — **H1 gate PASSED** (recall@1: 0.967 vs random 0.123)
- [x] Public plugins scaffold (Qiskit, TKET, PyZX, SDK)
- [x] **WQT20M-Beta trained** (19.06M params, 102.9M records, 15 domains)
- [x] **10x validation suite** — all 6 gates PASSED
- [x] **WQT20M-Beta release** — model card + HuggingFace-ready
- [ ] Publish model to HuggingFace
- [ ] Full plugin implementations
- [ ] WQT20M-1.0 (policy Top-1, legality, objective counterfactuals)

## Quick start (local development)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers tokenizers safetensors numpy pyyaml tqdm pytest

# Run tests
./scripts/run_tests.sh

# Build tokenizer
./scripts/build_tokenizer.sh

# Generate 1M training examples
python -m wqt20.data.pipeline --n 1000000 --out data/wqt20-curriculum-v1 --workers 8

# Smoke training (validates H1: learned policy beats random)
./scripts/run_smoke.sh
```

## Using the public plugins

```bash
pip install westquant-plugins[qiskit]
```

```python
from westquant import Search
result = Search(
    problem="MAXCUT",
    backend="ibm_brisbane",
    policy="WQT20",
    objectives={"two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2},
).run()
```

## Architecture

```
westquant/
├── plugins/               # PUBLIC — pushed to GitHub
│   ├── qiskit/             # Qiskit adapter
│   ├── tket/               # TKET adapter
│   ├── pyzx/               # PyZX adapter
│   ├── westquant_sdk/      # westquant.Search(...) SDK
│   ├── README.md           # public-facing readme
│   └── pyproject.toml      # public package config
├── wqt20/                 # PRIVATE — model code (not pushed)
│   ├── tokenizer.py        # ~4561 BPE, quantum-native structured tokens
│   ├── model.py            # 19M Llama-style, HF LlamaConfig, tied embeddings
│   ├── train.py            # smoke → pilot → 1.0
│   ├── data/               # curriculum generators (A-L), pipeline, provenance
│   ├── search/             # baselines + WQT20-guided policy
│   ├── eval/               # recall@k, regret, pareto, structural splits
│   └── tests/
├── ecosystem/             # PRIVATE — WQIR, registry, verifier (not pushed)
├── data/                  # PRIVATE — generated datasets (not pushed)
├── configs/                # PRIVATE — training configs (not pushed)
├── scripts/                # PRIVATE — build/run scripts (not pushed)
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
