# WestQuant Plugins

Official plugins for the **WQT20M** quantum representation scheduling model.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

## What WQT20M Does — 3 Examples

### Example 1: Better Postselection Probability

```
Problem:  Shor's algorithm on 10 qubits, naive scheduling
          → 20% postselection probability

          WQT20M-Beta schedules the transformations
          → same algorithm, same output, 90% postselection probability
```

### Example 2: Fewer Two-Qubit Gates

```
Problem:  QAOA circuit with 50 two-qubit gates, naive transpilation
          → 50 gates, depth 40, error 0.15

          WQT20M-Beta picks the right gate fusion + cancellation order
          → 31 gates, depth 22, error 0.08
```

### Example 3: Hardware-Aware Routing

```
Problem:  8-qubit circuit on heavy-hex topology, naive routing
          → 12 SWAP gates inserted

          WQT20M-Beta schedules SABRE_ROUTE + NATIVE_GATESET
          → 3 SWAP gates inserted, 75% reduction
```

---

## Quick Start

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model
model = AutoModelForCausalLM.from_pretrained("WestQuantStudio/WQT20M-Beta")
tokenizer = AutoTokenizer.from_pretrained("WestQuantStudio/WQT20M-Beta")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Ask: which transformation is better?
text = ("<SOLVE> <DOMAIN:graph_optimization> <LEVEL:GRAPH> "
        "<N_NODES:16> <DENSITY:0.5000> "
        "<RES:n_q=16 D=5 G1=10 G2=3 T=0 M=2 A=0 E=0.0100 C=1.0000> "
        "<OBJ_TYPE:balanced> "
        "<CAND_A> MAXCUT_ROUND <COST_A> 150.00 "
        "<CAND_B> TSP_ROUTE <COST_B> 200.00 "
        "<PREF>")

ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
with torch.no_grad():
    for _ in range(5):
        out = model(input_ids=ids)
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break

result = tokenizer.decode(ids[0].tolist()).split("<PREF>")[-1].strip()
print(result)  # → "A>B" (MAXCUT_ROUND is better because cost 150 < 200)
```

---

## What WQT20M Predicts

| Task | Input | Output | Accuracy |
|------|-------|--------|----------|
| **Preference** | State + 2 candidates with costs | Which candidate is better | 96.1% |
| **Value** | State | Predicted cost-to-go | Spearman 0.98 |
| **Legality** | State + action | Is this action legal? | 76.9% |
| **Hardware** | State + backend | Is this feasible? | 91.2% |

The model excels at **preference comparison** and **value prediction**. Use
these for model-guided search: score each candidate by predicted value, then
pick the best.

---

## Model

| Config | Value |
|--------|-------|
| Architecture | Llama-style decoder Transformer |
| Parameters | 19.06M |
| Model size | 76 MB |
| Context length | 2048 |
| Vocab | 4561 (quantum-native structured tokens + BPE) |

Standard HuggingFace `LlamaForCausalLM`. Compatible with `AutoModelForCausalLM`,
`AutoTokenizer`, and SafeTensors.

**Model card:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)

---

## Validation

All 6 release gates passed. Key results:

| Metric | Value |
|--------|-------|
| Preference accuracy | 96.1% |
| Value prediction (Spearman) | 0.98 |
| Search improvement (budget=100) | +35% over random |
| ID-only accuracy | 24% (no cheating) |
| Generalization (unseen instances) | 96% |

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

## License

Apache 2.0

## Links

- **WQT20M Model:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)
- **Organization:** [github.com/WestQuantOpen](https://github.com/WestQuantOpen)
