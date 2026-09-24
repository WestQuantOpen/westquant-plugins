"""Deterministic transformation/verifier APIs.

These are the ground-truth engines. WQT20 never executes transformations; it
only ranks them. The verifier certifies equivalence and computes metrics.

v0.1 implements exact, dependency-free verifiers for the smoke curriculum
(Pauli algebra, commutation, grouping, Trotter depth). Heavy tools (Qiskit,
TKET, PyZX) are wired behind the same interface via adapters when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# --- Pauli algebra (Domain C) ---

PAULI = {"I", "X", "Y", "Z"}
PAULI_MUL: Dict[Tuple[str, str], Tuple[str, complex]] = {
    ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
}


def pauli_multiply(a: str, b: str) -> Tuple[str, complex]:
    """Multiply two single-qubit Pauli letters -> (result_letter, phase)."""
    return PAULI_MUL[(a, b)]


def pauli_string_multiply(a: str, b: str) -> Tuple[str, complex]:
    """Multiply two equal-length Pauli strings -> (string, phase)."""
    assert len(a) == len(b)
    out = []
    phase = 1 + 0j
    for pa, pb in zip(a, b):
        r, p = pauli_multiply(pa, pb)
        out.append(r)
        phase *= p
    return "".join(out), phase


def pauli_commutes(a: str, b: str) -> bool:
    """Two Pauli strings commute iff they differ on an even number of
    non-identity positions (phase is real)."""
    _, phase = pauli_string_multiply(a, b)
    _, phase_rev = pauli_string_multiply(b, a)
    return abs(phase - phase_rev) < 1e-12


def pauli_weight(s: str) -> int:
    return sum(1 for c in s if c != "I")


# --- Commutation graph + greedy grouping (deterministic) ---

def commutation_graph(terms: List[str]) -> np.ndarray:
    n = len(terms)
    g = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if pauli_commutes(terms[i], terms[j]):
                g[i, j] = g[j, i] = True
    return g


def greedy_group(terms: List[str]) -> List[List[int]]:
    """Greedy graph-coloring grouping of mutually commuting terms.
    Returns list of groups (index lists). Minimizes number of groups."""
    g = commutation_graph(terms)
    n = len(terms)
    groups: List[List[int]] = []
    assigned = [False] * n
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if all(g[j][k] for k in group):
                group.append(j)
                assigned[j] = True
        groups.append(group)
    return groups


def n_groups(terms: List[str]) -> int:
    return len(greedy_group(terms))


# --- Trotter depth estimate (Domain F) ---

def trotter_depth(terms: List[str], order: int = 1) -> int:
    """Estimate product-formula depth. order=1: one layer per term.
    order=2: Strang splitting roughly doubles."""
    if order == 1:
        return len(terms)
    elif order == 2:
        return 2 * len(terms) - 1
    return (2 ** (order - 1)) * len(terms)


# --- Equivalence verifier ---

@dataclass
class VerifyResult:
    equivalence: str  # EXACT, GROUND_STATE, SPECTRAL, NOT_EQUIVALENT, ...
    valid: bool
    detail: str = ""


def verify_hamiltonian_equivalent(h1_terms: List[str], h1_coeffs: List[float],
                                  h2_terms: List[str], h2_coeffs: List[float]) -> VerifyResult:
    """Exact equivalence iff same Pauli terms with same coefficients (order-free)."""
    d1 = dict(zip(sorted(h1_terms), h1_coeffs))
    d2 = dict(zip(sorted(h2_terms), h2_coeffs))
    if d1 == d2:
        return VerifyResult("EXACT", True, "identical term sets")
    return VerifyResult("NOT_EQUIVALENT", False, "term sets differ")
