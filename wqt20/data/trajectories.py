"""Representation scheduling trajectories for WQT20 training.

A trajectory is a sequence of (state, legal_actions, oracle_action, value)
steps. The model is trained on next-action ranking (T4) and value prediction
(T7). Failed/low-value actions are included so the model learns dead ends.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .generators import (
    Graph, Hamiltonian, random_graph, maxcut_hamiltonian, mwis_hamiltonian,
    tfim_hamiltonian, serialize_state, serialize_legal_actions,
    serialize_action_value, action_value, oracle_action,
)
from ecosystem.actions import REGISTRY
from ecosystem.wqir import WQIRLevel


@dataclass
class TrajectoryStep:
    state_text: str
    legal_text: str
    action: str
    delta_2q: int
    delta_depth: int
    value: float
    is_oracle: bool


@dataclass
class Trajectory:
    problem: str
    family: str
    n_qubits: int
    steps: List[TrajectoryStep] = field(default_factory=list)

    def to_text(self) -> str:
        """Full serialized trajectory for next-token training."""
        lines = []
        for s in self.steps:
            lines.append(s.state_text)
            lines.append(s.legal_text)
            lines.append(serialize_action_value(s.action, s.delta_2q, s.delta_depth, s.value))
        return "\n".join(lines)


LEGAL_ACTIONS_BY_LEVEL = {
    "HAMILTONIAN": ["CANONICALIZE", "REORDER_TERMS", "GROUP_PAULIS", "TROTTERIZE",
                    "CHANGE_ENCODING", "TAPER_SYMMETRY", "CHANGE_BASIS", "STOP"],
    "OPERATOR_REPRESENTATION": ["GROUP_PAULIS", "CHANGE_BASIS", "TAPER_SYMMETRY", "STOP"],
    "ANSATZ": ["CHANGE_PRODUCT_FORMULA", "CHANGE_ANSATZ", "DECOMPOSE_UNITARY", "STOP"],
}


def generate_trajectory(
    h: Hamiltonian,
    level: str = "HAMILTONIAN",
    backend: str = "SIMULATOR",
    topo: str = "all_to_all",
    objective_weights: Dict[str, float] = None,
    rng: random.Random = None,
) -> Trajectory:
    """Generate a single scheduling trajectory for a Hamiltonian."""
    rng = rng or random.Random(42)
    legal = LEGAL_ACTIONS_BY_LEVEL.get(level, ["STOP"])
    state_text = serialize_state(h, level=level, backend=backend, topo=topo,
                                 objective_weights=objective_weights)
    oracle = oracle_action(h, legal)
    steps = []
    for action in legal:
        d2q, dd, val = action_value(h, action)
        steps.append(TrajectoryStep(
            state_text=state_text, legal_text=serialize_legal_actions(legal),
            action=action, delta_2q=d2q, delta_depth=dd, value=val,
            is_oracle=(action == oracle),
        ))
    return Trajectory(h.problem, h.graph.family, h.n_qubits, steps)


def generate_dataset(
    n: int = 1000,
    n_qubits_range: Tuple[int, int] = (4, 10),
    families: List[str] = None,
    problems: List[str] = None,
    seed: int = 42,
) -> List[Trajectory]:
    """Generate a dataset of scheduling trajectories."""
    rng = random.Random(seed)
    families = families or ["ER", "3regular", "cycle", "path", "grid"]
    problems = problems or ["MAXCUT", "MWIS", "TFIM"]
    trajs = []
    for _ in range(n):
        n_q = rng.randint(*n_qubits_range)
        family = rng.choice(families)
        problem = rng.choice(problems)
        p = rng.uniform(0.2, 0.6)
        g = random_graph(n_q, p, family, rng)
        if problem == "MAXCUT":
            h = maxcut_hamiltonian(g, rng)
        elif problem == "MWIS":
            h = mwis_hamiltonian(g, rng, penalty=rng.uniform(5.0, 20.0))
        else:
            h = tfim_hamiltonian(n_q, rng)
        backend = rng.choice(["SIMULATOR", "SUPERCONDUCTING", "TRAPPED_ION"])
        topo = rng.choice(["all_to_all", "heavy_hex", "grid", "linear"])
        obj = {"G2": round(rng.uniform(0.3, 0.6), 2),
               "D": round(rng.uniform(0.2, 0.4), 2),
               "E": round(1.0 - 0.0, 2)}
        s = sum(obj.values())
        obj = {k: round(v / s, 2) for k, v in obj.items()}
        trajs.append(generate_trajectory(h, backend=backend, topo=topo,
                                         objective_weights=obj, rng=rng))
    return trajs


def trajectory_to_examples(traj: Trajectory) -> List[Dict]:
    """Convert a trajectory to training examples (one per legal action).

    Each example: (state_text + legal_text) -> action token, with value label.
    Used for next-action ranking (T4) and value prediction (T7).
    """
    examples = []
    for step in traj.steps:
        text = step.state_text + " " + step.legal_text
        examples.append({
            "text": text,
            "action": step.action,
            "action_token": f"<A:{step.action}>",
            "value": step.value,
            "delta_2q": step.delta_2q,
            "delta_depth": step.delta_depth,
            "is_oracle": step.is_oracle,
            "problem": traj.problem,
            "family": traj.family,
            "n_qubits": traj.n_qubits,
        })
    return examples


def build_training_examples(n_trajectories: int = 1000, seed: int = 42) -> List[Dict]:
    """Generate trajectories and flatten to training examples."""
    trajs = generate_dataset(n_trajectories, seed=seed)
    examples = []
    for t in trajs:
        examples.extend(trajectory_to_examples(t))
    return examples
