"""WestQuant TKET Plugin.

Adapter that connects WQT50M representation scheduling to Quantinuum TKET.
The model ranks transformations; TKET executes them; metrics are measured.

Usage:
    from westquant.plugins.tket import TKETAdapter
    from pytket import Circuit

    c = Circuit(4)
    c.H(0); c.CX(0,1); c.CX(1,2); c.CX(2,3)
    c.measure_all()

    adapter = TKETAdapter(model="WestQuantStudio/WQT50M", objective="balanced")
    result = adapter.optimize(c)
    print(f"Improvement: {result['improvement_pct']:.1f}%")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


def _tket_to_qiskit(circuit):
    """Convert a TKET circuit to Qiskit for measurement."""
    from pytket.extensions.qiskit import tk_to_qiskit
    return tk_to_qiskit(circuit)


def _qiskit_to_tket(circuit):
    """Convert a Qiskit circuit to TKET."""
    from pytket.extensions.qiskit import qiskit_to_tk
    return qiskit_to_tk(circuit)


def _make_tket_strategies(backend_type="superconducting"):
    """Define real TKET compilation passes using the model's vocabulary."""
    from pytket import Circuit
    from pytket.passes import (
        RemoveRedundancies, CommuteThroughMultisq, DecomposeMultiQubitsCX,
        RebaseTket, FullPeepholeOptimise, CliffordSimp,
        SynthesiseTKET, SynthesiseUMod, SimplifyInitial,
        AutoRebase, AutoSimplify,
    )

    strategies = {}

    strategies["CANCEL_GATES"] = lambda c: _apply_passes(c, [
        CommuteThroughMultisq(), RemoveRedundancies(),
    ])

    strategies["COMMUTE_GATES"] = lambda c: _apply_passes(c, [
        CommuteThroughMultisq(),
    ])

    strategies["FUSE_ROTATIONS"] = lambda c: _apply_passes(c, [
        RemoveRedundancies(), SynthesiseTKET(),
    ])

    strategies["ZX_SIMPLIFY"] = lambda c: _apply_passes(c, [
        FullPeepholeOptimise(),
    ])

    strategies["REMOVE_REDUNDANT"] = lambda c: _apply_passes(c, [
        CommuteThroughMultisq(), RemoveRedundancies(), SynthesiseTKET(),
    ])

    strategies["MERGE_ADJACENT"] = lambda c: _apply_passes(c, [
        CommuteThroughMultisq(), RemoveRedundancies(),
        CliffordSimp(), SynthesiseTKET(),
    ])

    strategies["SABRE_ROUTE"] = lambda c: _apply_passes(c, [
        DecomposeMultiQubitsCX(), FullPeepholeOptimise(),
    ])

    strategies["DENSE_PLACE"] = lambda c: _apply_passes(c, [
        FullPeepholeOptimise(),
    ])

    strategies["BASIC_ROUTE"] = lambda c: _apply_passes(c, [
        DecomposeMultiQubitsCX(),
    ])

    strategies["NATIVE_GATESET"] = lambda c: _apply_passes(c, [
        SynthesiseTKET(),
    ])

    if backend_type == "trapped_ion":
        strategies["PULSE_OPTIMIZE"] = lambda c: _apply_passes(c, [
            SynthesiseUMod(),
        ])
    else:
        strategies["PULSE_OPTIMIZE"] = lambda c: _apply_passes(c, [
            FullPeepholeOptimise(), CliffordSimp(),
        ])

    strategies["STOP"] = lambda c: c.copy()

    return strategies


def _apply_passes(circuit, passes):
    """Apply a list of TKET passes to a circuit."""
    c = circuit.copy()
    for p in passes:
        p.apply(c)
    return c


def _measure_tket(circuit):
    """Measure TKET circuit metrics by converting to Qiskit."""
    try:
        from qiskit import QuantumCircuit
        qc = _tket_to_qiskit(circuit)
        ops = qc.count_ops()
        n_2q = sum(c for g, c in ops.items() if g in ('cx', 'cz', 'ecr', 'rzz'))
        n_1q = sum(c for g, c in ops.items() if g in ('rz', 'sx', 'x', 'h', 'z', 's', 't'))
        n_meas = ops.get('measure', 0)
        return {
            "n_qubits": qc.num_qubits,
            "depth": qc.depth(),
            "total_gates": qc.size(),
            "n_2q_gates": n_2q,
            "n_1q_gates": n_1q,
            "n_measurements": n_meas,
            "est_error": 0.0,
            "est_duration_ns": 0.0,
        }
    except Exception:
        # Fallback: use TKET's own metrics
        return {
            "n_qubits": circuit.n_qubits,
            "depth": circuit.depth(),
            "total_gates": len(circuit.get_commands()),
            "n_2q_gates": sum(1 for g in circuit.get_commands() if g.op.get_type().name in ('CX', 'CZ', 'ZZPhase')),
            "n_1q_gates": sum(1 for g in circuit.get_commands() if g.op.get_type().name in ('X', 'Z', 'H', 'S', 'T', 'Rz', 'Rx', 'Ry')),
            "n_measurements": 0,
            "est_error": 0.0,
            "est_duration_ns": 0.0,
        }


OBJECTIVE_WEIGHTS = {
    "depth_focused": {"depth": 10.0, "2q": 2.0, "gates": 0.1, "error": 0.0, "duration": 0.0},
    "2q_focused": {"depth": 1.0, "2q": 10.0, "gates": 0.1, "error": 0.0, "duration": 0.0},
    "fidelity_focused": {"depth": 0.5, "2q": 1.0, "gates": 0.0, "error": 1000.0, "duration": 0.0},
    "time_focused": {"depth": 0.0, "2q": 0.5, "gates": 0.0, "error": 0.0, "duration": 0.001},
    "balanced": {"depth": 2.0, "2q": 5.0, "gates": 1.0, "error": 100.0, "duration": 0.0001},
}


def _compute_cost(metrics, objective="balanced"):
    w = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["balanced"])
    return (
        w["depth"] * metrics["depth"] +
        w["2q"] * metrics["n_2q_gates"] +
        w["gates"] * metrics["total_gates"] +
        w["error"] * metrics["est_error"] +
        w["duration"] * metrics["est_duration_ns"]
    )


def _encode_state(metrics, objective, backend_type="SUPERCONDUCTING", step=0, max_steps=5):
    n_q = metrics["n_qubits"]
    entanglement = min(metrics["n_2q_gates"] / max(n_q - 1, 1), 1.0)
    coherence = 1.0 / (1.0 + metrics["depth"] * 0.01)
    avg_2q_err = 0.005
    avg_t1 = 200e-6
    avg_t2 = 100e-6
    avg_readout = 0.01
    if backend_type == "TRAPPED_ION":
        avg_2q_err = 0.001
        avg_t1 = 10.0
        avg_t2 = 2.0
        avg_readout = 0.002

    return (
        f"<DOMAIN:circuit_optimization> <LEVEL:CIRCUIT> "
        f"<N_QUBITS:{n_q}> <ENTANGLEMENT:{entanglement:.4f}> "
        f"<RES:n_q={n_q} D={metrics['depth']} G1={metrics['n_1q_gates']} G2={metrics['n_2q_gates']} "
        f"T={metrics['total_gates']} M={metrics['n_measurements']} A={metrics['est_error']:.4f} "
        f"E={avg_2q_err:.4f} C={coherence:.4f}> "
        f"<BACKEND:{backend_type}> <TOPO:all_to_all> "
        f"<T1:{avg_t1*1e6:.0f}us> <T2:{avg_t2*1e6:.0f}us> "
        f"<READOUT_ERR:{avg_readout:.4f}> "
        f"<OBJ_TYPE:{objective}> "
        f"<STEP:{step}/{max_steps}>"
    )


@dataclass
class TKETAdapter:
    """Bridge between WQT50M scheduling and TKET compilation.

    The model ranks legal transformations. TKET applies the chosen
    compilation pass. Metrics are measured on the output circuit.

    Example:
        from pytket import Circuit
        from westquant.plugins.tket import TKETAdapter

        c = Circuit(4)
        c.H(0); c.CX(0,1); c.CX(1,2); c.CX(2,3)
        c.measure_all()

        adapter = TKETAdapter(model="WestQuantStudio/WQT50M", objective="balanced")
        result = adapter.optimize(c)
        print(f"Improvement: {result['improvement_pct']:.1f}%")
    """
    model: str = "WestQuantStudio/WQT50M"
    backend: str = "superconducting"
    objective: str = "balanced"
    max_steps: int = 5
    _model_obj: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

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
        scored = [(name, self._predict_cost(state_text, name)) for name in action_names]
        scored.sort(key=lambda x: x[1])
        return scored

    def optimize(self, circuit: Any, max_steps: int = None) -> Dict:
        """Optimize a TKET circuit using WQT50M-guided scheduling.

        Args:
            circuit: A pytket Circuit
            max_steps: Override max optimization steps

        Returns:
            Dict with keys: status, original, optimized, improvement, trajectory
        """
        if self._model_obj is None:
            self.load()

        try:
            from pytket import Circuit
        except ImportError:
            raise ImportError("Install pytket: pip install pytket")

        backend_type = "TRAPPED_ION" if self.backend in ("ion", "ion_trap", "trapped_ion", "quantinuum") else "SUPERCONDUCTING"
        max_steps = max_steps or self.max_steps

        current = circuit.copy()
        naive_metrics = _measure_tket(current)
        naive_cost = _compute_cost(naive_metrics, self.objective)

        strategies = _make_tket_strategies(backend_type.lower() if backend_type == "TRAPPED_ION" else "superconducting")
        action_names = list(strategies.keys())

        trajectory = []
        best_circuit = current
        best_cost = naive_cost

        for step in range(max_steps):
            state = _encode_state(naive_metrics, self.objective, backend_type, step, max_steps)
            ranked = self._rank_actions(state, action_names)

            chosen = None
            for name, pred_cost in ranked:
                if name == "STOP" and step == 0:
                    continue
                try:
                    optimized = strategies[name](current)
                    opt_metrics = _measure_tket(optimized)
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
                naive_metrics = opt_metrics
            else:
                break

        best_metrics = _measure_tket(best_circuit)
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


__all__ = ["TKETAdapter"]
