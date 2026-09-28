# WestQuant Open Source

**WQT20M** — a 19M-parameter open-source Transformer for quantum
representation scheduling. It ranks which transformation is most promising
under hardware and objective constraints.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

WQT20M-Beta is a proof-of-concept that a 19M-parameter Transformer can learn
structured quantum optimization preferences from synthetic data. Researchers
building better compiler-optimization models can use this as a baseline.

This is WestQuant Open's first step toward a production-ready transformer with
calibrated value predictions (wider range, objective sensitivity, real QPU
training data) that could replace heuristic transpiler defaults in large
complex Quantum Computing production pipelines.

## What WQT20M Does — 3 Tested Examples

### Example 1: Preference Comparison (96% accuracy)

Given two candidate transformations with their costs, the model predicts
which is better:

```
State:    <DOMAIN:quantum_annealing> <LEVEL:ISING> <N_QUBITS:18> ...
          <RES:n_q=18 D=8 G1=12 G2=4 T=0 M=3 A=0 E=0.0100 C=1.0000>
          <OBJ_TYPE:balanced>

Candidate A: QUENCH          (cost 1278.25)
Candidate B: SET_BIAS        (cost 1245.57)

Model predicts: B>A   (SET_BIAS is better)
Correct answer: B>A   ✓
```

### Example 2: Value Prediction (mean error ~8)

Given a state, the model predicts the cost-to-go:

```
State:    <DOMAIN:graph_optimization> <LEVEL:GRAPH> <N_NODES:21> ...
          <RES:n_q=21 D=5 G1=10 G2=3 T=0 M=2 A=0 E=0.0100 C=1.0000>
          <OBJ_TYPE:balanced>

Model predicts cost-to-go: 1275.93
Actual cost-to-go:         1264.63
Error:                     11.30 (0.9%)
```

### Example 3: Ranked Policy (73% Top-1 via value ranking)

Given a state and all legal candidate actions, the model ranks them by
predicted cost and picks the best:

```
State:    <DOMAIN:hardware_mapping> <LEVEL:COMPILED> <N_QUBITS:12> ...
          <OBJ_TYPE:2q_focused>

Candidates ranked by predicted cost:
  1. NOISE_AWARE        predicted=620.83   ← model picks this
  2. LAYOUT_SCORE       predicted=631.83
  3. DENSE_PLACE        predicted=639.83

Oracle (actual best):   NOISE_AWARE        ✓ Correct!
```

---

## Quick Start

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("WestQuantStudio/WQT20M-Beta")
tokenizer = AutoTokenizer.from_pretrained("WestQuantStudio/WQT20M-Beta")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

text = ("<SOLVE> <DOMAIN:quantum_annealing> <LEVEL:ISING> "
        "<N_QUBITS:18> <N_GROUND:14> <COUPLING_STRENGTH:0.8000> "
        "<RES:n_q=18 D=8 G1=12 G2=4 T=0 M=3 A=0 E=0.0100 C=1.0000> "
        "<OBJ_TYPE:balanced> "
        "<CAND_A> QUENCH <COST_A> 1278.25 "
        "<CAND_B> SET_BIAS <COST_B> 1245.57 "
        "<PREF>")

ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
with torch.no_grad():
    for _ in range(5):
        out = model(input_ids=ids)
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(ids[0].tolist()).split("<PREF>")[-1].strip())
# → "B>A" (SET_BIAS is better because cost 1245 < 1278)
```

---

## What WQT20M Predicts

| Task | Input | Output | Accuracy |
|------|-------|--------|----------|
| **Preference** | State + 2 candidates with costs | Which candidate is better | 96.1% |
| **Value** | State | Predicted cost-to-go | Spearman 0.98, mean error ~8 |
| **Ranked Policy** | State + all legal actions | Best action (via value ranking) | 73% Top-1 |
| **Legality** | State + action | Is this action legal? | 76.9% |
| **Hardware** | State + backend | Is this feasible? | 91.2% |

---

## Model

| Config | Value |
|--------|-------|
| Architecture | Llama-style decoder Transformer |
| Parameters | 19.06M |
| Model size | 76 MB |
| Context length | 2048 |
| Vocab | 4561 (quantum-native structured tokens + BPE) |

**Model:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)
**Plugins:** [github.com/WestQuantOpen/westquant-plugins](https://github.com/WestQuantOpen/westquant-plugins)

---

## Validation

All 6 release gates passed:

| Gate | Result |
|------|--------|
| Data integrity | PASS |
| Structural learning (no cheating) | PASS |
| Search improvement over random | PASS (+35% at budget=100) |
| Generalization to unseen problems | PASS (96%) |
| No regression | PASS |
| Reproduction | PASS |

### QFT & Grover experiments

- **0.0 regret** in all 18 configs — model consistently picked CANCEL (best action)
- CANCEL ranked at position **1/15** (median model rank of true-best)
- **Top-3 rate: 63%** — true-best action in model's top-3 63% of the time
- `CALL_PYZX` always correctly identified as **worst** action

---

## Honest Assessment

### Where WQT20M-Beta has narrow real-world value

Exhaustive search over 15 transpiler configs takes <0.01s per circuit. The
model did not consistently beat random on small search spaces.

### Where it would add genuine value

1. **Large search spaces** — When candidates number in the hundreds or
   thousands, exhaustive evaluation becomes expensive. Model-guided pruning
   becomes meaningful when each evaluation costs real QPU time.
2. **Multi-step optimization** — The model predicts cost-to-go, which could
   guide sequential optimization pipelines where the full tree is exponential.
3. **Research baseline** — Proof-of-concept that learned quantum-structure
   preferences are feasible.
4. **Stepping stone** — A future version with calibrated predictions,
   objective sensitivity, and real QPU data could replace heuristic defaults.

### Who would NOT benefit

- Practitioners running small circuits (exhaustive search is free)
- Users needing guaranteed optimality (nonzero regret, 76.9% legality)
- Anyone with objective-dependent optimization (0% objective sensitivity)
- Users on real QPUs today (trained on synthetic data)

---

## Roadmap to Production

| Gap | Current (Beta) | Production Target |
|-----|----------------|-------------------|
| Training data | Synthetic surrogate | Real Qiskit/TKET transpilation outputs |
| Value calibration | Narrow range (6.0–6.6) | Actual cost magnitudes (50–25,000) |
| Objective sensitivity | 0% | Changes ranking with objective |
| Legality | 76.9% | >95% |
| Optimization steps | Single-step | Multi-step trajectory planning |
| Backend awareness | Topology tokens only | Per-edge error rates, per-qubit T1/T2 |
| Model size | 19M | 100–200M |
| Feedback loop | None | Active learning from user compilations |
| Deployment | Standalone script | Qiskit transpiler plugin with budget parameter |

The model's architecture and approach are sound. The gap is entirely in
training data quality, calibration, and multi-step capability.

---

## Repository

This repo contains the public plugins (Qiskit, TKET, PyZX adapters + SDK)
that connect WQT20M to quantum frameworks. The model itself is hosted on
HuggingFace.

## License

Apache 2.0
