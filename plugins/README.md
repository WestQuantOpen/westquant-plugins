# WestQuant Plugins

Official plugins for connecting the **WQT20** quantum representation scheduling
model to quantum computing frameworks.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

## WQT20 Model

WQT20 is an approximately 20M-parameter open-source Transformer specialized in
quantum representation scheduling — selecting which mathematical and circuit
transformations are most promising under hardware and objective constraints.

**Current release:** [WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)

**Validation:** All 6 release gates passed (10x suite, 2000 examples/task).
Preference 96.1%, Value Spearman 0.982, Search +35-100% over random.

WQT20 is **not** a chatbot, code generator, or replacement for Qiskit/TKET/PyZX.
It is a compact learned policy that ranks legal transformations. The plugins
execute those transformations. Verification certifies the results.

## Available Plugins

| Plugin | Framework | Status |
|--------|-----------|--------|
| `westquant.plugins.qiskit` | IBM Qiskit | Scaffold |
| `westquant.plugins.tket` | Quantinuum TKET | Scaffold |
| `westquant.plugins.pyzx` | PyZX (ZX-calculus) | Scaffold |
| `westquant.plugins.westquant_sdk` | Direct SDK | Beta (WQT20M-Beta) |

## Installation

```bash
pip install westquant-plugins

# With framework support
pip install westquant-plugins[qiskit]
pip install westquant-plugins[tket]
pip install westquant-plugins[pyzx]
```

## Quick Start

```python
from westquant import Search

result = Search(
    problem="MAXCUT",
    backend="ibm_brisbane",
    policy="WQT20",
    model_id="WestQuantStudio/WQT20M-Beta",
    objectives={
        "two_qubit_gates": 0.5,
        "depth": 0.3,
        "estimated_error": 0.2,
    },
).run()
```

### Model-Guided Preference Comparison

```python
from westquant import Search

s = Search(model_id="WestQuantStudio/WQT20M-Beta")
s._load_model()  # loads from HuggingFace

state = ("<DOMAIN:graph_optimization> <LEVEL:GRAPH> "
         "<N_NODES:16> <DENSITY:0.5> "
         "<RES:n_q=16 D=5 G1=10 G2=3 T=0 M=2 A=0 E=0.01 C=1.0> "
         "<OBJ_TYPE:balanced>")

# Compare two candidate transformations
pref = s._predict_preference(state, "MAXCUT_ROUND", 150.0, "TSP_ROUTE", 200.0)
print(pref)  # "A>B" (because COST_A < COST_B)

# Predict cost-to-go
value = s._predict_value(state)
print(value)  # predicted cost-to-go from this state

### Qiskit Plugin

```python
from westquant.plugins.qiskit import QiskitAdapter

adapter = QiskitAdapter(model="WestQuantStudio/WQT20M-Beta", backend="ibm_brisbane")
result = adapter.optimize(circuit, objectives={"two_qubit_gates": 0.5, "depth": 0.3})
```

### TKET Plugin

```python
from westquant.plugins.tket import TKETAdapter

adapter = TKETAdapter(model="WestQuantStudio/WQT20M-Beta", backend="Quantinuum:H2-1")
result = adapter.optimize(circuit)
```

### PyZX Plugin

```python
from westquant.plugins.pyzx import PyZXAdapter

adapter = PyZXAdapter(model="WestQuantStudio/WQT20M-Beta")
result = adapter.optimize(circuit)
```

## How It Works

1. **WQT20** ranks legal representation transformations given the current state,
   target hardware, and optimization objectives.
2. The **plugin** executes the top-ranked transformation using the framework's
   deterministic compiler (Qiskit transpiler, TKET passes, PyZX simplification).
3. **Verification** certifies that the transformation preserved equivalence.
4. The search continues until no improvement is found or the step budget is
   exhausted.

The model never executes transformations directly. It only schedules.

## License

Apache-2.0

## Links

- **WQT20 Model:** [huggingface.co/WestQuantStudio/WQT20M-Beta](https://huggingface.co/WestQuantStudio/WQT20M-Beta)
- **Organization:** [github.com/WestQuantOpen](https://github.com/WestQuantOpen)
- **Paper:** *(published when ready)*
