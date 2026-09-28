# WestQuant Open Source

**WQT20M** — a 19M-parameter open-source Transformer for quantum
representation scheduling. It ranks which transformation is most promising
under hardware and objective constraints.

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

model = AutoModelForCausalLM.from_pretrained("WestQuantStudio/WQT20M-Beta")
tokenizer = AutoTokenizer.from_pretrained("WestQuantStudio/WQT20M-Beta")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Which transformation is better?
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

print(tokenizer.decode(ids[0].tolist()).split("<PREF>")[-1].strip())
# → "A>B" (MAXCUT_ROUND is better because cost 150 < 200)
```

---

## What WQT20M Predicts

| Task | Input | Output | Accuracy |
|------|-------|--------|----------|
| **Preference** | State + 2 candidates with costs | Which candidate is better | 96.1% |
| **Value** | State | Predicted cost-to-go | Spearman 0.98 |
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

---

## Public vs Private

- **`plugins/`** — public. Contains Qiskit, TKET, PyZX adapters and the
  WestQuant SDK. Links to WQT20M on HuggingFace.
- **`wqt20/`, `ecosystem/`, `data/`, `configs/`, `scripts/`** — private.
  Model code, training infrastructure, generated data, checkpoints. Not
  pushed to public GitHub.

## License

Apache 2.0
