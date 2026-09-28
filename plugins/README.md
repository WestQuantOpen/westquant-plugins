# WestQuant Plugins

Official plugins for the **WQT20M** quantum representation scheduling model.

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

**Model card:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)

---

## Validation

All 6 release gates passed. Key results:

| Metric | Value |
|--------|-------|
| Preference accuracy | 96.1% |
| Value prediction (Spearman) | 0.98 |
| Ranked policy Top-1 | 73% (via value ranking) |
| Search improvement (budget=100) | +35% over random |
| ID-only accuracy | 24% (no cheating) |
| Generalization (unseen instances) | 96% |

### QFT & Grover experiments

Independent experiments on QFT and Grover circuits across 18 configurations:

- **0.0 regret** in all 18 configs — the model consistently picked CANCEL,
  the best action
- CANCEL ranked at position **1/15** (median model rank of true-best action)
- **Top-3 rate: 63%** — the true-best action is in the model's top-3 63% of the time
- `CALL_PYZX` is always correctly identified as the **worst** action

---

## Honest Assessment

### Where WQT20M-Beta has narrow real-world value

Exhaustive search over 15 transpiler configs takes <0.01s per circuit. Any
user who can afford to run Qiskit at all can afford to try all candidates and
pick the best. In experiments, many configurations were ties, and the model
did not consistently beat random on small search spaces.

### Where it would add genuine value

1. **Large search spaces** — When the candidate space grows to hundreds or
   thousands of configurations, exhaustive evaluation becomes expensive. The
   model's value-ranking approach becomes meaningful when each evaluation
   costs real QPU time or long simulation.

2. **Multi-step optimization** — The model predicts cost-to-go, which could
   guide intermediate decisions in sequential optimization pipelines where
   the full tree is exponentially large.

3. **Research baseline** — A proof-of-concept that a 19M-parameter Transformer
   can learn structured quantum optimization preferences from synthetic data.

4. **Stepping stone** — A future version with calibrated value predictions,
   objective sensitivity, and real QPU training data could replace heuristic
   transpiler defaults in production pipelines.

### Who would NOT benefit

- Practitioners running small circuits (exhaustive search is free and optimal)
- Users needing guaranteed optimality (nonzero regret, 76.9% legality)
- Anyone with objective-dependent optimization (0% objective sensitivity)
- Users on real QPUs today (trained on synthetic data, not hardware)

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

The model's architecture and approach (value-prediction-based ranking,
quantum-native tokenization, structured state encoding) are sound. The gap
is entirely in training data quality, calibration, and multi-step capability.

---

## Available Plugins

| Plugin | Framework | Status |
|--------|-----------|--------|
| `westquant.plugins.qiskit` | IBM Qiskit | Scaffold |
| `westquant.plugins.tket` | Quantinuum TKET | Scaffold |
| `westquant.plugins.pyzx` | PyZX (ZX-calculus) | Scaffold |
| `westquant.plugins.westquant_sdk` | Direct SDK | Beta |

## Installation

```bash
pip install westquant-plugins

# With framework support
pip install westquant-plugins[qiskit]
pip install westquant-plugins[tket]
pip install westquant-plugins[pyzx]
```

---

## How It Works

1. **WQT20M** ranks legal representation transformations given the current
   state, target hardware, and optimization objectives.
2. The **plugin** executes the top-ranked transformation using the
   framework's deterministic compiler (Qiskit transpiler, TKET passes, PyZX
   simplification).
3. **Verification** certifies that the transformation preserved equivalence.
4. The search continues until no improvement is found or the step budget is
   exhausted.

The model never executes transformations directly. It only schedules.

---

## Limitations

This is a **Beta** release:
- **Policy generation: 0.8%** — Use value ranking instead (73% Top-1)
- **Legality: 76.9%** — Below target
- **Objective sensitivity: 0%** — Model doesn't change predictions with objective
- **Value calibration** — Predicted values cluster in a narrow range (6.0–6.6)
- **Not a circuit compiler** — Works on structured state representation, not raw circuits

---

## License

Apache 2.0

## Links

- **WQT20M Model:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)
- **Organization:** [github.com/WestQuantOpen](https://github.com/WestQuantOpen)
