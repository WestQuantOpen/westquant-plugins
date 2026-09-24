"""Transformation registry.

An action has: ACTION_ID, ACTION_FAMILY, SOURCE_TYPE, TARGET_TYPE, PARAMETERS,
PRECONDITIONS, VERIFIER, ESTIMATED_COST.

At every state the registry produces the *legal* action set. WQT20 scores
legal actions; the search chooses top-k; deterministic tools execute them.
This makes invalid action generation structurally impossible wherever the
preconditions are checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .wqir import WQIRLevel, WQIRState


@dataclass
class Transformation:
    action_id: str
    family: str
    source_level: WQIRLevel
    target_level: WQIRLevel
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: Optional[Callable[[WQIRState], bool]] = None
    verifier: str = "UNKNOWN"
    estimated_cost: float = 1.0
    reversible: bool = False

    @property
    def token(self) -> str:
        return f"<A:{self.action_id}>"

    def is_legal(self, state: WQIRState) -> bool:
        if state.level != self.source_level and self.source_level != WQIRLevel.PROBLEM:
            # transformations may also refine within the same level
            if state.level != self.target_level and self.target_level != state.level:
                return False
        if self.preconditions is not None:
            return self.preconditions(state)
        return True


# --- Action families (Part V) ---

ACTION_DEFS: List[Transformation] = [
    Transformation("CANONICALIZE", "CANONICALIZE", WQIRLevel.HAMILTONIAN, WQIRLevel.HAMILTONIAN, verifier="EXACT_LINEAR", estimated_cost=0.1, reversible=True),
    Transformation("CHANGE_BASIS", "CHANGE_BASIS", WQIRLevel.OPERATOR_REPRESENTATION, WQIRLevel.OPERATOR_REPRESENTATION, verifier="UNITARY_EQUIV", estimated_cost=0.5, reversible=True),
    Transformation("CHANGE_ENCODING", "CHANGE_ENCODING", WQIRLevel.ENCODING, WQIRLevel.ENCODING, verifier="GROUND_STATE_EQUIV", estimated_cost=1.0, reversible=True),
    Transformation("MAP_FERMIONS", "MAP_FERMIONS", WQIRLevel.HAMILTONIAN, WQIRLevel.OPERATOR_REPRESENTATION, verifier="EXACT_LINEAR", estimated_cost=2.0, reversible=True),
    Transformation("TAPER_SYMMETRY", "TAPER_SYMMETRY", WQIRLevel.OPERATOR_REPRESENTATION, WQIRLevel.OPERATOR_REPRESENTATION, verifier="SPECTRAL_EQUIV", estimated_cost=1.5),
    Transformation("GROUP_PAULIS", "GROUP_PAULIS", WQIRLevel.OPERATOR_REPRESENTATION, WQIRLevel.OPERATOR_REPRESENTATION, verifier="COMMUTATION_GRAPH", estimated_cost=0.2, reversible=True),
    Transformation("REORDER_TERMS", "REORDER_TERMS", WQIRLevel.HAMILTONIAN, WQIRLevel.HAMILTONIAN, verifier="EXACT_LINEAR", estimated_cost=0.1, reversible=True),
    Transformation("TROTTERIZE", "TROTTERIZE", WQIRLevel.HAMILTONIAN, WQIRLevel.ANSATZ, verifier="PRODUCT_FORMULA", estimated_cost=3.0),
    Transformation("CHANGE_PRODUCT_FORMULA", "CHANGE_PRODUCT_FORMULA", WQIRLevel.ANSATZ, WQIRLevel.ANSATZ, verifier="PRODUCT_FORMULA", estimated_cost=1.0, reversible=True),
    Transformation("DECOMPOSE_UNITARY", "DECOMPOSE_UNITARY", WQIRLevel.ANSATZ, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=5.0),
    Transformation("CHANGE_ANSATZ", "CHANGE_ANSATZ", WQIRLevel.ANSATZ, WQIRLevel.ANSATZ, verifier="EXPRESSIVITY", estimated_cost=2.0, reversible=True),
    Transformation("REDUCE_ACTIVE_SPACE", "REDUCE_ACTIVE_SPACE", WQIRLevel.HAMILTONIAN, WQIRLevel.HAMILTONIAN, verifier="APPROXIMATE", estimated_cost=1.0),
    Transformation("CIRCUIT_REWRITE", "CIRCUIT_REWRITE", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=0.5, reversible=True),
    Transformation("ZX_REWRITE", "ZX_REWRITE", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="ZX_CALCULUS", estimated_cost=0.5, reversible=True),
    Transformation("COMMUTE", "COMMUTE", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=0.1, reversible=True),
    Transformation("CANCEL", "CANCEL", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=0.1, reversible=True),
    Transformation("FUSE_ROTATIONS", "FUSE_ROTATIONS", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=0.1, reversible=True),
    Transformation("DECOMPOSE_GATE", "DECOMPOSE_GATE", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=0.3, reversible=True),
    Transformation("SELECT_NATIVE_GATESET", "SELECT_NATIVE_GATESET", WQIRLevel.SYNTHESIZED_CIRCUIT, WQIRLevel.NATIVE_GATE_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=4.0),
    Transformation("SELECT_SYNTHESIS_ENGINE", "SELECT_SYNTHESIS_ENGINE", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.SYNTHESIZED_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=4.0),
    Transformation("SELECT_ROUTING_ENGINE", "SELECT_ROUTING_ENGINE", WQIRLevel.SYNTHESIZED_CIRCUIT, WQIRLevel.ROUTED_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=4.0),
    Transformation("MAP_QUBITS", "MAP_QUBITS", WQIRLevel.ROUTED_CIRCUIT, WQIRLevel.ROUTED_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=2.0, reversible=True),
    Transformation("INSERT_SWAP_NETWORK", "INSERT_SWAP_NETWORK", WQIRLevel.ROUTED_CIRCUIT, WQIRLevel.ROUTED_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=3.0),
    Transformation("CALL_QISKIT", "CALL_QISKIT", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.NATIVE_GATE_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=8.0),
    Transformation("CALL_TKET", "CALL_TKET", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.NATIVE_GATE_CIRCUIT, verifier="UNITARY_EQUIV", estimated_cost=8.0),
    Transformation("CALL_PYZX", "CALL_PYZX", WQIRLevel.LOGICAL_CIRCUIT, WQIRLevel.LOGICAL_CIRCUIT, verifier="ZX_CALCULUS", estimated_cost=4.0),
    Transformation("EVALUATE", "EVALUATE", WQIRLevel.NATIVE_GATE_CIRCUIT, WQIRLevel.HARDWARE_EXECUTION_STATE, verifier="SIMULATOR", estimated_cost=10.0),
    Transformation("VERIFY", "VERIFY", WQIRLevel.HAMILTONIAN, WQIRLevel.HAMILTONIAN, verifier="EXACT_LINEAR", estimated_cost=1.0),
    Transformation("BRANCH", "BRANCH", WQIRLevel.PROBLEM, WQIRLevel.PROBLEM, estimated_cost=0.0),
    Transformation("BACKTRACK", "BACKTRACK", WQIRLevel.PROBLEM, WQIRLevel.PROBLEM, estimated_cost=0.0),
    Transformation("STOP", "STOP", WQIRLevel.PROBLEM, WQIRLevel.PROBLEM, estimated_cost=0.0),
]


ACTION_TOKENS = [t.token for t in ACTION_DEFS]
ACTION_FAMILIES = sorted({t.family for t in ACTION_DEFS})


class TransformationRegistry:
    def __init__(self, transforms: List[Transformation] = None):
        self.transforms = {t.action_id: t for t in (transforms or ACTION_DEFS)}

    def legal_actions(self, state: WQIRState) -> List[str]:
        legal = []
        for t in self.transforms.values():
            if t.is_legal(state):
                legal.append(t.action_id)
        # STOP is always legal
        if "STOP" not in legal:
            legal.append("STOP")
        return legal

    def get(self, action_id: str) -> Optional[Transformation]:
        return self.transforms.get(action_id)


REGISTRY = TransformationRegistry()
