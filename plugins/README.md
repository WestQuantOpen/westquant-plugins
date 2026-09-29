# WestQuant Plugins

Official plugins for the **WQT20M** and **WQT50M** quantum representation
scheduling models.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

## Models

| Model | Params | Training Data | Key Result |
|-------|--------|---------------|------------|
| [WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta) | 19M | Synthetic | Proof-of-concept baseline |
| [**WQT50M**](https://huggingface.co/WestQuantStudio/WQT50M) | **50M** | **Real Qiskit transpilation** | **74% Top-1, 90.4% beats random** |

WQT50M is trained on **real Qiskit transpilation outputs** — actual depth, gate
counts, and error estimates from real circuits transpiled on real backend
topologies. It predicts which transformation is most promising given a circuit,
backend, and optimization objective.

---

## What WQT50M Does — 3 Tested Examples

### Example 1: 62% Gate Reduction

```
Problem:  8-qubit random circuit (depth 30) on grid topology
          Naive transpilation → cost 1231.2 (fidelity-focused)

          WQT50M schedules ZX_SIMPLIFY
          → cost 467.4 (62.0% reduction)

          Random selection → cost 869.6 (29.4% reduction)
          Oracle (exhaustive) → cost 467.4 (62.0%)
          WQT50M matches oracle ✓
```

### Example 2: Objective-Aware Scheduling

The same QAOA circuit gets different best actions depending on the objective:

```
QAOA-6q-p2 on linear backend:

  Objective            WQT50M picks
  ─────────────────    ─────────────────
  balanced             → CANCEL_GATES
  depth_focused        → NATIVE_GATESET
  fidelity_focused     → FUSE_ROTATIONS
  time_focused         → MERGE_ADJACENT

  4 different actions for 5 objectives ✓
```

### Example 3: Calibrated Value Prediction

```
State:    <DOMAIN:circuit_optimization> <N_QUBITS:8> ...
          <BACKEND:SUPERCONDUCTING> <TOPO:grid>
          <OBJ_TYPE:fidelity_focused>

Action:   ZX_SIMPLIFY
          Predicted cost: 467.4
          Actual cost:    467.4
          Error:          0.0%

Calibration ratio: 0.999 (predicted range matches actual range)
Spearman correlation: 0.981
```

---

## Quick Start

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("WestQuantStudio/WQT50M")
tokenizer = AutoTokenizer.from_pretrained("WestQuantStudio/WQT50M")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Predict cost for a state-action pair
state = ("<DOMAIN:circuit_optimization> <LEVEL:CIRCUIT> "
         "<N_QUBITS:8> <ENTANGLEMENT:0.5000> "
         "<RES:n_q=8 D=30 G1=100 G2=20 T=120 M=0 A=0.2000 E=0.0050 C=0.5000> "
         "<BACKEND:SUPERCONDUCTING> <TOPO:grid> "
         "<T1:180us> <T2:90us> <READOUT_ERR:0.0120> "
         "<OBJ_TYPE:fidelity_focused> <STEP:0/5>")

text = f"<PREDICT> {state} <ACTION> ZX_SIMPLIFY <COST> "
ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
with torch.no_grad():
    for _ in range(8):
        out = model(input_ids=ids)
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(ids[0].tolist()).split("<COST>")[-1].strip())
```

---

## Validation Results

### 450 tests: 15 circuits × 5 objectives × 6 backends

| Metric | WQT20M-Beta | WQT50M |
|--------|-------------|--------|
| Top-1 accuracy | ~22% | **74.0%** |
| Beats random | 58% | **90.4%** |
| Avg improvement over naive | -14.8% | **+16.6%** |
| Efficiency (% of oracle) | 23.3% | **88.7%** |
| Value calibration ratio | ~0.01 | **0.999** |
| Objective sensitivity | 0% | **65%** |
| Backend sensitivity | ~0% | **60%** |

### By circuit type

| Circuit | WQT50M improvement | Oracle | Efficiency |
|---------|-------------------|--------|------------|
| BV-4q | 26.6% | 26.6% | **100%** |
| GHZ-4q | 11.9% | 11.9% | **100%** |
| GHZ-8q | 6.5% | 6.5% | **100%** |
| QFT-4q | 11.5% | 11.6% | **99.0%** |
| Grover-4q | 21.3% | 21.5% | **98.8%** |
| Random-8q-d30 | 51.1% | 57.2% | 89.4% |

---

## Production Gap Coverage

WQT50M addresses 8 of the 9 production gaps:

| Gap | WQT20M-Beta | WQT50M | Status |
|-----|-------------|--------|--------|
| Real compiler outputs | Synthetic | Real Qiskit | ✅ |
| Value calibration | 6.0–6.6 | Ratio 0.999 | ✅ |
| Objective sensitivity | 0% | 65% | ✅ |
| Legality | 76.9% | 79.2% | ⚠️ |
| Multi-step trajectories | Single-step | 2-5 steps | ✅ |
| Backend awareness | Tokens only | 60% | ✅ |
| Model capacity | 19M | 49.53M | ✅ |
| Search improvement | +35% | 90.4% win | ✅ |

---

## Model

| Config | Value |
|--------|-------|
| Architecture | Llama-style decoder Transformer |
| Parameters | 49.53M |
| Layers | 12 |
| Hidden size | 512 |
| Attention heads | 8 (4 KV heads, GQA) |
| FFN | 2048 (SwiGLU) |
| Context length | 4096 |
| Vocab | 4561 (quantum-native structured tokens + BPE) |
| Model size | 189 MB |

**Models:**
- [WQT50M](https://huggingface.co/WestQuantStudio/WQT50M) (50M, real Qiskit data)
- [WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta) (19M, synthetic baseline)

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

1. **WQT50M** ranks legal representation transformations given the current
   state, target hardware, and optimization objectives.
2. The **plugin** executes the top-ranked transformation using the
   framework's deterministic compiler (Qiskit transpiler, TKET passes, PyZX
   simplification).
3. **Verification** certifies that the transformation preserved equivalence.
4. The search continues until no improvement is found or the step budget is
   exhausted.

The model never executes transformations directly. It only schedules.

---

## License

Apache 2.0

## Links

- **WQT50M Model:** [huggingface.co/WestQuantStudio/WQT50M](https://huggingface.co/WestQuantStudio/WQT50M)
- **WQT20M-Beta Model:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)
- **Organization:** [github.com/WestQuantOpen](https://github.com/WestQuantOpen)
