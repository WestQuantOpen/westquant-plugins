"""WestQuant PyZX Plugin.

Adapter that connects WQT50M representation scheduling to PyZX for
ZX-calculus-based circuit optimization. The model decides when ZX rewrites
are promising; PyZX executes them; metrics are measured.

Usage:
    import pyzx as zx
    from westquant.plugins.pyzx import PyZXAdapter

    # Load or build a circuit in PyZX
    circ = zx.Circuit.from_qasm_file("circuit.qasm")
    g = circ.to_graph()

    adapter = PyZXAdapter(model="WestQuantStudio/WQT50M", objective="balanced")
    result = adapter.optimize(g)
    print(f"Improvement: {result['improvement_pct']:.1f}%")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


def _make_pyzx_strategies():
    """Define real PyZX simplification strategies using the model's vocabulary."""
    import pyzx as zx

    strategies = {}

    strategies["CANCEL_GATES"] = lambda g: _apply_zx(g, [
        zx.simplify.spider_simp,
        zx.simplify.id_simp,
    ])

    strategies["COMMUTE_GATES"] = lambda g: _apply_zx(g, [
        zx.simplify.pivot_simp,
    ])

    strategies["FUSE_ROTATIONS"] = lambda g: _apply_zx(g, [
        zx.simplify.spider_simp,
        zx.simplify.to_z_simp,
    ])

    strategies["ZX_SIMPLIFY"] = lambda g: _full_reduce(g)

    strategies["REMOVE_REDUNDANT"] = lambda g: _apply_zx(g, [
        zx.simplify.spider_simp,
        zx.simplify.id_simp,
        zx.simplify.to_z_simp,
    ])

    strategies["MERGE_ADJACENT"] = lambda g: _apply_zx(g, [
        zx.simplify.spider_simp,
        zx.simplify.pivot_simp,
        zx.simplify.lcomp_simp,
    ])

    strategies["SABRE_ROUTE"] = lambda g: _full_reduce(g)

    strategies["DENSE_PLACE"] = lambda g: _full_reduce(g)

    strategies["BASIC_ROUTE"] = lambda g: _apply_zx(g, [
        zx.simplify.spider_simp,
    ])

    strategies["NATIVE_GATESET"] = lambda g: _apply_zx(g, [
        zx.simplify.to_z_simp,
        zx.simplify.to_x_simp,
    ])

    strategies["PULSE_OPTIMIZE"] = lambda g: _full_reduce(g)

    strategies["STOP"] = lambda g: g.copy()

    return strategies


def _apply_zx(graph, simplifications):
    """Apply a list of PyZX simplification functions to a graph."""
    g = graph.copy()
    for simp in simplifications:
        simp(g)
    return g


def _full_reduce(graph):
    """Apply PyZX full reduction."""
    import pyzx as zx
    g = graph.copy()
    zx.simplify.full_reduce(g)
    return g


def _measure_pyzx(graph):
    """Measure PyZX graph metrics."""
    import pyzx as zx
    try:
        circ = zx.Circuit.from_graph(graph)
        stats = circ.stats()
        # PyZX stats: 'qubits', 'gates', 'cnots', 'depth'
        n_gates = circ.gates.__len__() if hasattr(circ, 'gates') else stats.get('gates', 0)
        n_2q = sum(1 for g in circ.gates if hasattr(g, 'name') and g.name in ('CNOT', 'CZ')) if hasattr(circ, 'gates') else stats.get('cnots', 0)
        depth = stats.get('depth', 0) if isinstance(stats, dict) else 0
        return {
            "n_qubits": stats.get('qubits', 0) if isinstance(stats, dict) else graph.qubits(),
            "depth": depth,
            "total_gates": n_gates,
            "n_2q_gates": n_2q,
            "n_1q_gates": n_gates - n_2q,
            "n_measurements": 0,
            "est_error": 0.0,
            "est_duration_ns": 0.0,
        }
    except Exception:
        # Fallback: count graph vertices
        n_vertices = graph.vcount()
        n_edges = graph.ecount()
        return {
            "n_qubits": graph.qubits(),
            "depth": n_edges,
            "total_gates": n_vertices,
            "n_2q_gates": n_edges,
            "n_1q_gates": max(0, n_vertices - n_edges),
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


def _encode_state(metrics, objective, step=0, max_steps=5):
    n_q = metrics["n_qubits"]
    entanglement = min(metrics["n_2q_gates"] / max(n_q - 1, 1), 1.0)
    coherence = 1.0 / (1.0 + metrics["depth"] * 0.01)
    return (
        f"<DOMAIN:circuit_optimization> <LEVEL:CIRCUIT> "
        f"<N_QUBITS:{n_q}> <ENTANGLEMENT:{entanglement:.4f}> "
        f"<RES:n_q={n_q} D={metrics['depth']} G1={metrics['n_1q_gates']} G2={metrics['n_2q_gates']} "
        f"T={metrics['total_gates']} M={metrics['n_measurements']} A={metrics['est_error']:.4f} "
        f"E=0.0050 C={coherence:.4f}> "
        f"<BACKEND:SUPERCONDUCTING> <TOPO:all_to_all> "
        f"<T1:200us> <T2:100us> <READOUT_ERR:0.0100> "
        f"<OBJ_TYPE:{objective}> "
        f"<STEP:{step}/{max_steps}>"
    )


@dataclass
class PyZXAdapter:
    """Bridge between WQT50M scheduling and PyZX ZX-calculus simplification.

    The model ranks legal ZX rewrites. PyZX executes the chosen
    simplification. Metrics are measured on the output graph.

    Example:
        import pyzx as zx
        from westquant.plugins.pyzx import PyZXAdapter

        circ = zx.Circuit.from_qasm_file("circuit.qasm")
        g = circ.to_graph()

        adapter = PyZXAdapter(model="WestQuantStudio/WQT50M", objective="balanced")
        result = adapter.optimize(g)
        print(f"Improvement: {result['improvement_pct']:.1f}%")
    """
    model: str = "WestQuantStudio/WQT50M"
    objective: str = "balanced"
    max_steps: int = 3
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

    def optimize(self, graph: Any, max_steps: int = None) -> Dict:
        """Optimize a PyZX graph using WQT50M-guided scheduling.

        Args:
            graph: A PyZX Graph object
            max_steps: Override max optimization steps

        Returns:
            Dict with keys: status, original, optimized, improvement, trajectory
        """
        if self._model_obj is None:
            self.load()

        try:
            import pyzx as zx
        except ImportError:
            raise ImportError("Install pyzx: pip install pyzx")

        max_steps = max_steps or self.max_steps

        current = graph.copy()
        naive_metrics = _measure_pyzx(current)
        naive_cost = _compute_cost(naive_metrics, self.objective)

        strategies = _make_pyzx_strategies()
        action_names = list(strategies.keys())

        trajectory = []
        best_graph = current
        best_cost = naive_cost

        for step in range(max_steps):
            state = _encode_state(naive_metrics, self.objective, step, max_steps)
            ranked = self._rank_actions(state, action_names)

            chosen = None
            for name, pred_cost in ranked:
                if name == "STOP" and step == 0:
                    continue
                try:
                    optimized = strategies[name](current)
                    opt_metrics = _measure_pyzx(optimized)
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
                best_graph = optimized
                current = optimized
                naive_metrics = opt_metrics
            else:
                break

        best_metrics = _measure_pyzx(best_graph)
        improvement = (naive_cost - best_cost) / naive_cost if naive_cost > 0 else 0.0

        return {
            "status": "optimized",
            "model": self.model,
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
            "graph": best_graph,
        }

    def simplify_with_policy(self, graph: Any) -> Any:
        """Simplify a PyZX graph using WQT50M-guided strategy selection.

        Convenience method that returns just the optimized graph.
        """
        result = self.optimize(graph)
        return result["graph"]


__all__ = ["PyZXAdapter"]
