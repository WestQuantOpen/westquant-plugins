# WestQuant Plugins

Official plugins for connecting the **WQT20** quantum representation scheduling
model to quantum computing frameworks.

> AI schedules. Deterministic mathematics executes. Independent verification certifies.

## WQT20 Model

WQT20 is an approximately 20M-parameter open-source Transformer specialized in
quantum representation scheduling — selecting which mathematical and circuit
transformations are most promising under hardware and objective constraints.

**Model card:** [https://huggingface.co/westquant/WQT20-1.0](https://huggingface.co/westquant/WQT20-1.0) *(published when ready)*

WQT20 is **not** a chatbot, code generator, or replacement for Qiskit/TKET/PyZX.
It is a compact learned policy that ranks legal transformations. The plugins
execute those transformations. Verification certifies the results.

## Available Plugins

| Plugin | Framework | Status |
|--------|-----------|--------|
| `westquant.plugins.qiskit` | IBM Qiskit | Scaffold |
| `westquant.plugins.tket` | Quantinuum TKET | Scaffold |
| `westquant.plugins.pyzx` | PyZX (ZX-calculus) | Scaffold |
| `westquant.plugins.westquant_sdk` | Direct SDK | Scaffold |

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
    objectives={
        "two_qubit_gates": 0.5,
        "depth": 0.3,
        "estimated_error": 0.2,
    },
).run()
```

### Qiskit Plugin

```python
from westquant.plugins.qiskit import QiskitAdapter

adapter = QiskitAdapter(model="westquant/WQT20-1.0", backend="ibm_brisbane")
result = adapter.optimize(circuit, objectives={"two_qubit_gates": 0.5, "depth": 0.3})
```

### TKET Plugin

```python
from westquant.plugins.tket import TKETAdapter

adapter = TKETAdapter(model="westquant/WQT20-1.0", backend="Quantinuum:H2-1")
result = adapter.optimize(circuit)
```

### PyZX Plugin

```python
from westquant.plugins.pyzx import PyZXAdapter

adapter = PyZXAdapter(model="westquant/WQT20-1.0")
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

- **WQT20 Model:** [huggingface.co/westquant/WQT20-1.0](https://huggingface.co/westquant/WQT20-1.0)
- **Organization:** [github.com/WestQuantOpen](https://github.com/WestQuantOpen)
- **Paper:** *(published when ready)*
