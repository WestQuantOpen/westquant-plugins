"""WQT20 data generators: the mathematical curriculum (Part III).

Dependency-free, deterministic, seedable generators for the smoke curriculum.
Each generator produces structured objects (Pauli strings, Hamiltonians,
graphs, commutation structures) and the serialized WQT20 token format.

The smoke curriculum focuses on the representation-scheduling signal that is
both learnable and verifiable: Pauli commutation grouping and Trotter ordering
for graph Hamiltonians (MaxCut, MWIS, Ising).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ecosystem.verify import (
    pauli_commutes, commutation_graph, greedy_group, n_groups,
    trotter_depth, pauli_weight,
)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]
    family: str

    def adjacency(self) -> np.ndarray:
        a = np.zeros((self.n, self.n), dtype=int)
        for u, v in self.edges:
            a[u][v] = a[v][u] = 1
        return a

    def density(self) -> float:
        max_e = self.n * (self.n - 1) / 2
        return len(self.edges) / max(max_e, 1)


def random_graph(n: int, p: float, family: str, rng: random.Random) -> Graph:
    edges = []
    if family == "ER":
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.append((i, j))
    elif family == "3regular":
        # random 3-regular via stub matching (small n)
        stubs = list(range(n)) * 3
        rng.shuffle(stubs)
        i = 0
        while i + 1 < len(stubs):
            u, v = stubs[i], stubs[i + 1]
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
            i += 2
    elif family == "cycle":
        for i in range(n):
            edges.append((i, (i + 1) % n))
    elif family == "path":
        for i in range(n - 1):
            edges.append((i, i + 1))
    elif family == "grid":
        side = int(np.sqrt(n))
        for r in range(side):
            for c in range(side):
                idx = r * side + c
                if c + 1 < side:
                    edges.append((idx, idx + 1))
                if r + 1 < side:
                    edges.append((idx, idx + side))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.append((i, j))
    return Graph(n, edges, family)


# ---------------------------------------------------------------------------
# Hamiltonian generation (graph -> Pauli terms)
# ---------------------------------------------------------------------------

@dataclass
class Hamiltonian:
    problem: str           # MAXCUT, MWIS, ISING, TFIM
    n_qubits: int
    terms: List[str]       # Pauli strings
    coeffs: List[float]
    graph: Graph

    def n_terms(self) -> int:
        return len(self.terms)


def maxcut_hamiltonian(g: Graph, rng: random.Random) -> Hamiltonian:
    """MaxCut H = sum_{(i,j)} 0.5*(I - Z_i Z_j). Pauli terms are ZZ strings."""
    terms = []
    coeffs = []
    for u, v in g.edges:
        s = ["I"] * g.n
        s[u] = "Z"
        s[v] = "Z"
        terms.append("".join(s))
        coeffs.append(rng.uniform(0.3, 1.0))
    return Hamiltonian("MAXCUT", g.n, terms, coeffs, g)


def mwis_hamiltonian(g: Graph, rng: random.Random, penalty: float = 10.0) -> Hamiltonian:
    """MWIS penalty Hamiltonian: ZZ on edges + Z field terms."""
    terms = []
    coeffs = []
    for u, v in g.edges:
        s = ["I"] * g.n
        s[u] = "Z"
        s[v] = "Z"
        terms.append("".join(s))
        coeffs.append(penalty * rng.uniform(0.5, 1.5))
    for i in range(g.n):
        s = ["I"] * g.n
        s[i] = "Z"
        terms.append("".join(s))
        coeffs.append(-rng.uniform(0.5, 2.0))
    return Hamiltonian("MWIS", g.n, terms, coeffs, g)


def tfim_hamiltonian(n: int, rng: random.Random, jz: float = 1.0, hx: float = 0.5) -> Hamiltonian:
    """Transverse-field Ising: -J ZZ - h X."""
    g = Graph(n, [(i, i + 1) for i in range(n - 1)], "path")
    terms = []
    coeffs = []
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = s[i + 1] = "Z"
        terms.append("".join(s))
        coeffs.append(jz * rng.uniform(0.5, 1.5))
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        terms.append("".join(s))
        coeffs.append(hx * rng.uniform(0.3, 0.8))
    return Hamiltonian("TFIM", n, terms, coeffs, g)


# ---------------------------------------------------------------------------
# Serialization to WQT20 token format
# ---------------------------------------------------------------------------

def serialize_state(h: Hamiltonian, level: str = "HAMILTONIAN", backend: str = "SIMULATOR",
                    topo: str = "all_to_all", objective_weights: Dict[str, float] = None) -> str:
    """Serialize a Hamiltonian state into the WQT20 token format."""
    if objective_weights is None:
        objective_weights = {"G2": 0.5, "D": 0.3, "E": 0.2}
    parts = [
        "<STATE>",
        f"<LEVEL:{level}>",
        f"<PROBLEM:{h.problem}>",
        f"<H:{h.problem}>",
        f"n_q={h.n_qubits} terms={h.n_terms()}",
        f"<BACKEND:{backend}>",
        f"<topo:{topo}>",
        "<TERMS>",
    ]
    for t, c in zip(h.terms, h.coeffs):
        parts.append(f"{t}={c:.3f}")
    parts.append("</TERMS>")
    parts.append("<OBJECTIVE>")
    for k, v in objective_weights.items():
        parts.append(f"<W:{k}>={v:.2f}")
    parts.append("<target:minimize>")
    parts.append("<RES>")
    parts.append(f"depth={trotter_depth(h.terms)} gates_2q={2*h.n_terms()} measurements={n_groups(h.terms)}")
    parts.append("</STATE>")
    return " ".join(parts)


def serialize_legal_actions(actions: List[str]) -> str:
    return "<LEGAL> " + " ".join(f"<A:{a}>" for a in actions) + " </LEGAL>"


def serialize_action_value(action: str, delta_2q: int, delta_depth: int, value: float) -> str:
    return (f"<NEXT> <A:{action}> <DELTA_2Q>={delta_2q} "
            f"<DELTA_DEPTH>={delta_depth} <VALUE>={value:.4f}")


# ---------------------------------------------------------------------------
# Deterministic oracle: compute action values
# ---------------------------------------------------------------------------

def action_value(h: Hamiltonian, action: str) -> Tuple[int, int, float]:
    """Compute deterministic (delta_2q, delta_depth, value) for an action.

    value in [0,1]: higher = more promising for the minimize objective.
    The signal is a clear function of state structure so the model can learn
    the conditional policy:
      - GROUP_PAULIS best when many commuting terms (high compression)
      - TAPER_SYMMETRY best when many qubits (symmetry reduction)
      - CHANGE_ENCODING best when dense graph (encoding compression)
      - REORDER_TERMS best when many non-commuting terms (Trotter error)
    """
    from ecosystem.verify import n_groups as _n_groups
    base_groups = _n_groups(h.terms)
    base_depth = trotter_depth(h.terms, order=1)
    base_2q = 2 * h.n_terms()
    n_terms = h.n_terms()
    n_q = h.n_qubits
    density = h.graph.density()
    compression = (n_terms - base_groups) / max(n_terms, 1)

    if action == "GROUP_PAULIS":
        delta_2q = 0
        delta_depth = 0
        # strong signal: value scales with commutation compression
        value = 0.2 + 0.7 * compression
    elif action == "REORDER_TERMS":
        delta_2q = 0
        delta_depth = 0
        # best when terms are non-commuting (low compression = high reorder benefit)
        value = 0.2 + 0.6 * (1.0 - compression)
    elif action == "TROTTERIZE":
        delta_2q = base_2q
        delta_depth = base_depth
        value = 0.1
    elif action == "CHANGE_ENCODING":
        delta_2q = -int(0.1 * base_2q)
        delta_depth = -int(0.1 * base_depth)
        # best when dense graph
        value = 0.15 + 0.65 * density
    elif action == "TAPER_SYMMETRY":
        delta_2q = -int(0.2 * base_2q)
        delta_depth = -int(0.15 * base_depth)
        # best when many qubits
        value = 0.15 + 0.05 * (n_q - 4)  # 4->0.15, 10->0.45
        value = min(value, 0.6)
    elif action == "CHANGE_BASIS":
        delta_2q = 0
        delta_depth = 0
        value = 0.2
    elif action == "CANONICALIZE":
        delta_2q = 0
        delta_depth = 0
        value = 0.15
    elif action == "STOP":
        delta_2q = 0
        delta_depth = 0
        value = 0.05
    else:
        delta_2q = 0
        delta_depth = 0
        value = 0.1
    return delta_2q, delta_depth, value


def oracle_action(h: Hamiltonian, legal: List[str]) -> str:
    """Return the highest-value legal action (the oracle)."""
    best = max(legal, key=lambda a: action_value(h, a)[2])
    return best
