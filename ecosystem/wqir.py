"""WestQuant Intermediate Representation (WQIR) schema.

Every training state identifies its level in the representation stack:

    Problem -> Mathematical formulation -> Hamiltonian -> Encoding ->
    Operator representation -> Algorithm -> Ansatz -> Logical circuit ->
    Synthesized circuit -> Routed circuit -> Native-gate circuit ->
    Hardware execution state

A WQIRState carries the level, a structured payload, the backend spec, the
objective spec, and a resource vector. Transformations move a state between
levels (or refine within a level) and must declare source/target level,
preconditions, preserved invariants, equivalence class, and verification
method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class WQIRLevel(IntEnum):
    PROBLEM = 1
    MATHEMATICAL_FORMULATION = 2
    HAMILTONIAN = 3
    ENCODING = 4
    OPERATOR_REPRESENTATION = 5
    ALGORITHM = 6
    ANSATZ = 7
    LOGICAL_CIRCUIT = 8
    SYNTHESIZED_CIRCUIT = 9
    ROUTED_CIRCUIT = 10
    NATIVE_GATE_CIRCUIT = 11
    HARDWARE_EXECUTION_STATE = 12

    @property
    def token(self) -> str:
        return f"<LEVEL:{self.name}>"


LEVEL_TOKENS = [lvl.token for lvl in WQIRLevel]


class EquivalenceClass(str):
    EXACT = "EXACT"
    GLOBAL_PHASE = "GLOBAL_PHASE"
    SPECTRAL = "SPECTRAL"
    GROUND_STATE = "GROUND_STATE"
    OBJECTIVE = "OBJECTIVE"
    APPROXIMATE = "APPROXIMATE"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    UNKNOWN = "UNKNOWN"


EQUIVALENCE_TOKENS = [f"<EQ:{c}>" for c in [
    EquivalenceClass.EXACT, EquivalenceClass.GLOBAL_PHASE,
    EquivalenceClass.SPECTRAL, EquivalenceClass.GROUND_STATE,
    EquivalenceClass.OBJECTIVE, EquivalenceClass.APPROXIMATE,
    EquivalenceClass.NOT_EQUIVALENT, EquivalenceClass.UNKNOWN,
]]


@dataclass
class ResourceVector:
    """r = (n_q, D, G1, G2, T, M, A, E, C)."""
    n_qubits: int = 0
    depth: int = 0
    gates_1q: int = 0
    gates_2q: int = 0
    t_count: int = 0
    measurements: int = 0
    ancillas: int = 0
    estimated_error: float = 0.0
    cost: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_q": self.n_qubits, "D": self.depth, "G1": self.gates_1q,
            "G2": self.gates_2q, "T": self.t_count, "M": self.measurements,
            "A": self.ancillas, "E": self.estimated_error, "C": self.cost,
        }

    def dominates(self, other: "ResourceVector") -> bool:
        """Pareto dominance: self dominates other if <= on all and < on some."""
        d = self.as_dict()
        o = other.as_dict()
        return all(d[k] <= o[k] for k in d) and any(d[k] < o[k] for k in d)


@dataclass
class BackendSpec:
    name: str = "simulator"
    technology: str = "simulator"  # superconducting, trapped_ion, neutral_atom, simulator, ft_ideal
    n_qubits: int = 8
    topology: str = "all_to_all"  # all_to_all, heavy_hex, grid, linear, ring
    native_1q: List[str] = field(default_factory=lambda: ["rz", "rx", "ry", "h", "x", "y", "z", "s", "t"])
    native_2q: List[str] = field(default_factory=lambda: ["cx", "cz", "rxx", "ryy", "rzz", "swap", "iswap"])
    gate_error_1q: float = 0.0
    gate_error_2q: float = 0.0
    readout_error: float = 0.0

    @property
    def token(self) -> str:
        return f"<BACKEND:{self.technology.upper()}>"


@dataclass
class ObjectiveSpec:
    """Weighted multi-objective. Weights sum to 1."""
    weights: Dict[str, float] = field(default_factory=lambda: {"G2": 0.5, "D": 0.3, "E": 0.2})

    def scalar(self, rv: ResourceVector) -> float:
        d = rv.as_dict()
        return sum(self.weights.get(k, 0.0) * d.get(k, 0.0) for k in self.weights)


@dataclass
class WQIRState:
    state_id: str
    level: WQIRLevel
    problem: str = "UNKNOWN"
    payload: Dict[str, Any] = field(default_factory=dict)
    backend: BackendSpec = field(default_factory=BackendSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    resources: ResourceVector = field(default_factory=ResourceVector)
    history: List[str] = field(default_factory=list)
    verification: str = "UNKNOWN"

    def legal_actions(self, registry) -> List[str]:
        return registry.legal_actions(self)
