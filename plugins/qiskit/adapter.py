"""WestQuant Qiskit Plugin.

Adapter that connects WQT50M/WQT20M representation scheduling to IBM Qiskit.
The model ranks transformations; Qiskit executes them; metrics are measured.

Usage:
    from westquant.plugins.qiskit import QiskitAdapter

    adapter = QiskitAdapter(
        model="WestQuantStudio/WQT50M",
        backend="linear",
        objective="balanced",
    )
    result = adapter.optimize(circuit)
    print(result["improvement"])  # e.g. 0.62 = 62% cost reduction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


BASIS_GATES = ["cx", "rz", "sx", "x", "measure"]


def _make_strategies(coupling_map=None):
    """Define real transpilation strategies using the model's vocabulary."""
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        InverseCancellation, ConsolidateBlocks, CommutativeCancellation,
        Optimize1qGatesDecomposition, RemoveBarriers, Collect2qBlocks,
    )
    from qiskit.circuit.library import CXGate, HGate, XGate, ZGate

    cm = coupling_map
    strategies = {}

    strategies["CANCEL_GATES"] = lambda qc: PassManager([
        InverseCancellation([CXGate(), HGate(), XGate(), ZGate()])
    ]).run(qc.copy())

    strategies["COMMUTE_GATES"] = lambda qc: PassManager([
        CommutativeCancellation()
    ]).run(qc.copy())

    strategies["FUSE_ROTATIONS"] = lambda qc: PassManager([
        Optimize1qGatesDecomposition()
    ]).run(qc.copy())

    strategies["ZX_SIMPLIFY"] = lambda qc: PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(),
    ]).run(qc.copy())

    strategies["REMOVE_REDUNDANT"] = lambda qc: PassManager([
        RemoveBarriers(),
        InverseCancellation([CXGate(), HGate(), XGate(), ZGate()]),
        Optimize1qGatesDecomposition(),
    ]).run(qc.copy())

    strategies["MERGE_ADJACENT"] = lambda qc: PassManager([
        RemoveBarriers(),
        CommutativeCancellation(),
        Optimize1qGatesDecomposition(),
        InverseCancellation([CXGate(), HGate(), XGate(), ZGate()]),
    ]).run(qc.copy())

    if cm:
        strategies["SABRE_ROUTE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, coupling_map=cm,
            routing_method="sabre", layout_method="sabre",
            optimization_level=1)
        strategies["DENSE_PLACE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, coupling_map=cm,
            layout_method="dense", optimization_level=1)
        strategies["BASIC_ROUTE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, coupling_map=cm,
            routing_method="basic", layout_method="trivial",
            optimization_level=1)
    else:
        strategies["SABRE_ROUTE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, optimization_level=1)
        strategies["DENSE_PLACE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, optimization_level=1)
        strategies["BASIC_ROUTE"] = lambda qc: transpile(
            qc, basis_gates=BASIS_GATES, optimization_level=1)

    strategies["NATIVE_GATESET"] = lambda qc: transpile(
        qc, basis_gates=BASIS_GATES, optimization_level=2)

    strategies["PULSE_OPTIMIZE"] = lambda qc: transpile(
        qc, basis_gates=BASIS_GATES, optimization_level=3)

    strategies["STOP"] = lambda qc: qc.copy()

    return strategies


def _measure_circuit(qc, backend=None):
    """Measure real circuit metrics."""
    ops = qc.count_ops()
    n_2q = sum(c for g, c in ops.items() if g in ('cx', 'cz', 'ecr', 'rzz'))
    n_1q = sum(c for g, c in ops.items() if g in ('rz', 'sx', 'x', 'h', 'z', 's', 't'))
    n_meas = ops.get('measure', 0)

    depth = qc.depth()
    total_gates = qc.size()

    est_error = 0.0
    est_duration = 0.0
    if backend and backend.get("gate_errors"):
        avg_2q_err = np.mean(list(backend["gate_errors"].values())) if backend["gate_errors"] else 0.01
        avg_1q_err = avg_2q_err * 0.1
        est_error = n_2q * avg_2q_err + n_1q * avg_1q_err
        avg_2q_dur = np.mean(list(backend["gate_durations"].values())) if backend.get("gate_durations") else 500
        est_duration = n_2q * avg_2q_dur + n_1q * 50 + n_meas * 1000

    return {
        "n_qubits": qc.num_qubits,
        "depth": depth,
        "total_gates": total_gates,
        "n_2q_gates": n_2q,
        "n_1q_gates": n_1q,
        "n_measurements": n_meas,
        "est_error": est_error,
        "est_duration_ns": est_duration,
    }


OBJECTIVE_WEIGHTS = {
    "depth_focused": {"depth": 10.0, "2q": 2.0, "gates": 0.1, "error": 0.0, "duration": 0.0},
    "2q_focused": {"depth": 1.0, "2q": 10.0, "gates": 0.1, "error": 0.0, "duration": 0.0},
    "fidelity_focused": {"depth": 0.5, "2q": 1.0, "gates": 0.0, "error": 1000.0, "duration": 0.0},
    "time_focused": {"depth": 0.0, "2q": 0.5, "gates": 0.0, "error": 0.0, "duration": 0.001},
    "balanced": {"depth": 2.0, "2q": 5.0, "gates": 1.0, "error": 100.0, "duration": 0.0001},
}


def _compute_cost(metrics, objective="balanced"):
    """Compute objective-dependent cost. Lower = better."""
    w = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["balanced"])
    return (
        w["depth"] * metrics["depth"] +
        w["2q"] * metrics["n_2q_gates"] +
        w["gates"] * metrics["total_gates"] +
        w["error"] * metrics["est_error"] +
        w["duration"] * metrics["est_duration_ns"]
    )


def _encode_state(qc, objective, backend=None, step=0, max_steps=5):
    """Encode circuit + backend into the model's state format."""
    m = _measure_circuit(qc, backend)
    n_q = m["n_qubits"]
    entanglement = min(m["n_2q_gates"] / max(n_q - 1, 1), 1.0)
    coherence = 1.0 / (1.0 + m["depth"] * 0.01)

    if backend:
        topo = backend["topology"]
        backend_type = backend["backend_type"]
        avg_2q_err = np.mean(list(backend["gate_errors"].values())) if backend["gate_errors"] else 0.01
        avg_t1 = np.mean(list(backend["qubit_t1"].values())) if backend.get("qubit_t1") else 200e-6
        avg_t2 = np.mean(list(backend["qubit_t2"].values())) if backend.get("qubit_t2") else 100e-6
        avg_readout = np.mean(list(backend["readout_error"].values())) if backend.get("readout_error") else 0.01
    else:
        topo = "all_to_all"
        backend_type = "SUPERCONDUCTING"
        avg_2q_err = 0.005
        avg_t1 = 200e-6
        avg_t2 = 100e-6
        avg_readout = 0.01

    return (
        f"<DOMAIN:circuit_optimization> <LEVEL:CIRCUIT> "
        f"<N_QUBITS:{n_q}> <ENTANGLEMENT:{entanglement:.4f}> "
        f"<RES:n_q={n_q} D={m['depth']} G1={m['n_1q_gates']} G2={m['n_2q_gates']} "
        f"T={m['total_gates']} M={m['n_measurements']} A={m['est_error']:.4f} "
        f"E={avg_2q_err:.4f} C={coherence:.4f}> "
        f"<BACKEND:{backend_type}> <TOPO:{topo}> "
        f"<T1:{avg_t1*1e6:.0f}us> <T2:{avg_t2*1e6:.0f}us> "
        f"<READOUT_ERR:{avg_readout:.4f}> "
        f"<OBJ_TYPE:{objective}> "
        f"<STEP:{step}/{max_steps}>"
    )


def _make_backend(name):
    """Create a backend configuration by name."""
    from qiskit.transpiler import CouplingMap
    import random
    rng = random.Random(42)

    if name in ("simulator", "all_to_all", None):
        return {
            "coupling_map": None,
            "topology": "all_to_all",
            "n_qubits": 8,
            "gate_errors": {(i, j): 0.003 for i in range(8) for j in range(8) if i != j},
            "gate_durations": {(i, j): 400 for i in range(8) for j in range(8) if i != j},
            "qubit_t1": {i: 250e-6 for i in range(8)},
            "qubit_t2": {i: 120e-6 for i in range(8)},
            "readout_error": {i: 0.005 for i in range(8)},
            "backend_type": "SUPERCONDUCTING",
        }
    if name in ("linear", "line"):
        cm = CouplingMap.from_line(8)
        return {
            "coupling_map": cm,
            "topology": "linear", "n_qubits": 8,
            "gate_errors": {(i, i+1): 0.005 + 0.002 * i for i in range(7)},
            "gate_durations": {(i, i+1): 500 + 20 * i for i in range(7)},
            "qubit_t1": {i: 150e-6 + 10e-6 * i for i in range(8)},
            "qubit_t2": {i: 80e-6 + 5e-6 * i for i in range(8)},
            "readout_error": {i: 0.01 + 0.002 * i for i in range(8)},
            "backend_type": "SUPERCONDUCTING",
        }
    if name in ("ring",):
        cm = CouplingMap.from_ring(8)
        return {
            "coupling_map": cm,
            "topology": "ring", "n_qubits": 8,
            "gate_errors": {(i, (i+1) % 8): 0.004 + 0.001 * i for i in range(8)},
            "gate_durations": {(i, (i+1) % 8): 450 + 15 * i for i in range(8)},
            "qubit_t1": {i: 200e-6 - 5e-6 * i for i in range(8)},
            "qubit_t2": {i: 100e-6 - 3e-6 * i for i in range(8)},
            "readout_error": {i: 0.008 + 0.001 * i for i in range(8)},
            "backend_type": "SUPERCONDUCTING",
        }
    if name in ("grid",):
        cm = CouplingMap.from_grid(2, 4)
        backend = {
            "coupling_map": cm,
            "topology": "grid", "n_qubits": 8,
            "gate_errors": {}, "gate_durations": {},
            "qubit_t1": {i: 180e-6 for i in range(8)},
            "qubit_t2": {i: 90e-6 for i in range(8)},
            "readout_error": {i: 0.012 for i in range(8)},
            "backend_type": "SUPERCONDUCTING",
        }
        for edge in cm.get_edges():
            backend["gate_errors"][tuple(edge)] = 0.006 + rng.random() * 0.004
            backend["gate_durations"][tuple(edge)] = 480 + rng.randint(0, 80)
        return backend
    if name in ("heavy_hex", "ibm"):
        cm = CouplingMap.from_heavy_hex(3)
        backend = {
            "coupling_map": cm,
            "topology": "heavy_hex", "n_qubits": 8,
            "gate_errors": {}, "gate_durations": {},
            "qubit_t1": {i: 150e-6 + rng.random() * 50e-6 for i in range(8)},
            "qubit_t2": {i: 80e-6 + rng.random() * 30e-6 for i in range(8)},
            "readout_error": {i: 0.01 + rng.random() * 0.01 for i in range(8)},
            "backend_type": "SUPERCONDUCTING",
        }
        for edge in cm.get_edges():
            backend["gate_errors"][tuple(edge)] = 0.005 + rng.random() * 0.005
            backend["gate_durations"][tuple(edge)] = 500 + rng.randint(0, 100)
        return backend
    if name in ("ion_trap", "ion", "trapped_ion"):
        return {
            "coupling_map": None,
            "topology": "all_to_all", "n_qubits": 8,
            "gate_errors": {(i, j): 0.001 for i in range(8) for j in range(8) if i != j},
            "gate_durations": {(i, j): 5000 for i in range(8) for j in range(8) if i != j},
            "qubit_t1": {i: 10.0 for i in range(8)},
            "qubit_t2": {i: 2.0 for i in range(8)},
            "readout_error": {i: 0.002 for i in range(8)},
            "backend_type": "TRAPPED_ION",
        }
    # Default: all-to-all
    return _make_backend("simulator")


@dataclass
class QiskitAdapter:
    """Bridge between WQT50M scheduling and Qiskit compilation.

    The model ranks legal transformations. Qiskit executes the chosen
    transformation. Metrics are measured on the real output circuit.

    Example:
        from qiskit.circuit.library import QFT
        adapter = QiskitAdapter(model="WestQuantStudio/WQT50M",
                                backend="linear", objective="balanced")
        result = adapter.optimize(QFT(4).decompose().measure_all())
        print(f"Improvement: {result['improvement']:.1%}")
    """
    model: str = "WestQuantStudio/WQT50M"
    backend: str = "simulator"
    objective: str = "balanced"
    max_steps: int = 5
    _model_obj: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"
    _backend_config: Any = None

    def load(self):
        """Load the model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model_obj = AutoModelForCausalLM.from_pretrained(self.model).to(self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model_obj.eval()
        except ImportError:
            raise ImportError("Install transformers + torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model}': {e}")

    def _predict_cost(self, state_text, action_name):
        """Predict cost for a state-action pair."""
        import torch
        import re
        text = f"<PREDICT> {state_text} <ACTION> {action_name} <COST> "
        ids = self._tokenizer.encode(text, add_special_tokens=False,
                                     return_tensors="pt").to(self._device)
        with torch.no_grad():
            for _ in range(8):
                if ids.shape[1] > 512:
                    break
                out = self._model_obj(input_ids=ids)
                nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, nxt], dim=1)
                if nxt.item() == self._tokenizer.eos_token_id:
                    break
        gen = self._tokenizer.decode(ids[0].tolist())
        after = gen.split("<COST>")[-1].strip()
        if "<eos>" in after:
            after = after.split("<eos>")[0].strip()
        m = re.search(r"(\d+\.?\d*)", after)
        return float(m.group(1)) if m else 0.0

    def _rank_actions(self, state_text, action_names):
        """Rank actions by predicted cost (lowest = best)."""
        scored = [(name, self._predict_cost(state_text, name)) for name in action_names]
        scored.sort(key=lambda x: x[1])
        return scored

    def optimize(self, circuit: Any, max_steps: int = None) -> Dict:
        """Optimize a circuit using WQT50M-guided scheduling + Qiskit execution.

        The model ranks legal transformations at each step. Qiskit applies
        the top-ranked transformation. The result is measured.

        Args:
            circuit: A Qiskit QuantumCircuit
            max_steps: Override max optimization steps

        Returns:
            Dict with keys: status, original, optimized, improvement,
            trajectory, model, backend, objective
        """
        if self._model_obj is None:
            self.load()

        try:
            from qiskit import transpile
        except ImportError:
            raise ImportError("Install qiskit: pip install qiskit")

        max_steps = max_steps or self.max_steps
        backend = _make_backend(self.backend)
        self._backend_config = backend

        # Transpile to basis gates first (naive baseline)
        try:
            current = transpile(circuit, basis_gates=BASIS_GATES,
                               optimization_level=0)
        except Exception:
            current = circuit.copy()

        naive_metrics = _measure_circuit(current, backend)
        naive_cost = _compute_cost(naive_metrics, self.objective)

        strategies = _make_strategies(backend["coupling_map"])
        action_names = list(strategies.keys())

        trajectory = []
        best_circuit = current
        best_cost = naive_cost

        for step in range(max_steps):
            state = _encode_state(current, self.objective, backend, step, max_steps)
            ranked = self._rank_actions(state, action_names)

            # Pick the top action (skip STOP unless it's the only option)
            chosen = None
            for name, pred_cost in ranked:
                if name == "STOP" and step == 0:
                    continue
                try:
                    optimized = strategies[name](current)
                    opt_metrics = _measure_circuit(optimized, backend)
                    opt_cost = _compute_cost(opt_metrics, self.objective)
                    chosen = name
                    break
                except Exception:
                    continue

            if chosen is None:
                break

            trajectory.append({
                "step": step,
                "action": chosen,
                "predicted_cost": ranked[0][1],
                "actual_cost": opt_cost,
                "depth": opt_metrics["depth"],
                "gates": opt_metrics["total_gates"],
                "n_2q": opt_metrics["n_2q_gates"],
            })

            if opt_cost < best_cost:
                best_cost = opt_cost
                best_circuit = optimized
                current = optimized
            else:
                # No improvement — stop
                break

        best_metrics = _measure_circuit(best_circuit, backend)
        improvement = (naive_cost - best_cost) / naive_cost if naive_cost > 0 else 0.0

        return {
            "status": "optimized",
            "model": self.model,
            "backend": self.backend,
            "objective": self.objective,
            "original": {
                "depth": naive_metrics["depth"],
                "gates": naive_metrics["total_gates"],
                "n_2q": naive_metrics["n_2q_gates"],
                "cost": naive_cost,
            },
            "optimized": {
                "depth": best_metrics["depth"],
                "gates": best_metrics["total_gates"],
                "n_2q": best_metrics["n_2q_gates"],
                "cost": best_cost,
            },
            "improvement": improvement,
            "improvement_pct": improvement * 100,
            "n_steps": len(trajectory),
            "trajectory": trajectory,
            "circuit": best_circuit,
        }

    def compare(self, circuit: Any) -> Dict:
        """Compare WQT50M-guided vs random vs exhaustive (oracle).

        Returns a dict showing how WQT50M compares to random selection
        and exhaustive search over all strategies.
        """
        if self._model_obj is None:
            self.load()

        try:
            from qiskit import transpile
        except ImportError:
            raise ImportError("Install qiskit: pip install qiskit")

        backend = _make_backend(self.backend)
        strategies = _make_strategies(backend["coupling_map"])

        try:
            base = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=0)
        except Exception:
            base = circuit.copy()

        naive_metrics = _measure_circuit(base, backend)
        naive_cost = _compute_cost(naive_metrics, self.objective)

        # Evaluate all strategies
        all_results = {}
        for name, fn in strategies.items():
            try:
                opt = fn(base.copy())
                m = _measure_circuit(opt, backend)
                all_results[name] = {"cost": _compute_cost(m, self.objective), "metrics": m}
            except Exception:
                all_results[name] = None

        valid = {k: v for k, v in all_results.items() if v is not None}
        if not valid:
            return {"status": "error", "message": "No strategies succeeded"}

        oracle_name = min(valid, key=lambda k: valid[k]["cost"])
        oracle_cost = valid[oracle_name]["cost"]

        # WQT50M pick
        state = _encode_state(base, self.objective, backend)
        ranked = self._rank_actions(state, list(strategies.keys()))
        wqt50m_choice = ranked[0][0]
        if all_results.get(wqt50m_choice) is None:
            for name, _ in ranked[1:]:
                if all_results.get(name) is not None:
                    wqt50m_choice = name
                    break
        wqt50m_cost = all_results[wqt50m_choice]["cost"]

        # Random pick (average of 10 trials)
        import random
        rng = random.Random(42)
        random_costs = []
        for _ in range(10):
            name = rng.choice(list(strategies.keys()))
            while all_results.get(name) is None:
                name = rng.choice(list(strategies.keys()))
            random_costs.append(all_results[name]["cost"])
        random_avg = np.mean(random_costs)

        return {
            "naive_cost": naive_cost,
            "wqt50m": {"choice": wqt50m_choice, "cost": wqt50m_cost},
            "random": {"avg_cost": random_avg},
            "oracle": {"choice": oracle_name, "cost": oracle_cost},
            "wqt50m_improvement": (naive_cost - wqt50m_cost) / naive_cost,
            "random_improvement": (naive_cost - random_avg) / naive_cost,
            "oracle_improvement": (naive_cost - oracle_cost) / naive_cost,
            "wqt50m_correct": wqt50m_choice == oracle_name,
            "all_strategies": {k: v["cost"] for k, v in valid.items()},
        }


__all__ = ["QiskitAdapter"]
