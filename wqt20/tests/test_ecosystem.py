"""Tests for the WQT20 ecosystem: WQIR, transformations, verifier, RepGraph."""
import pytest
from ecosystem.wqir import WQIRLevel, WQIRState, ResourceVector, BackendSpec, ObjectiveSpec
from ecosystem.actions import REGISTRY, Transformation
from ecosystem.repgraph import RepresentationGraph, RepGraphEdge
from ecosystem.verify import (
    pauli_commutes, pauli_string_multiply, commutation_graph, greedy_group,
    n_groups, trotter_depth,
)


def test_wqir_levels():
    assert len(WQIRLevel) == 12
    assert WQIRLevel.PROBLEM == 1
    assert WQIRLevel.HARDWARE_EXECUTION_STATE == 12


def test_resource_pareto():
    a = ResourceVector(gates_2q=10, depth=5)
    b = ResourceVector(gates_2q=12, depth=6)
    assert a.dominates(b)
    assert not b.dominates(a)


def test_registry_legal_actions():
    state = WQIRState(state_id="s1", level=WQIRLevel.HAMILTONIAN, problem="MAXCUT")
    legal = REGISTRY.legal_actions(state)
    assert "STOP" in legal
    # CANONICALIZE and REORDER_TERMS are HAMILTONIAN-level actions
    assert "CANONICALIZE" in legal
    assert "REORDER_TERMS" in legal


def test_pauli_commutation():
    assert pauli_commutes("ZZ", "XX")  # commute (differ on 2 positions)
    assert not pauli_commutes("XZ", "YI")  # anticommute


def test_pauli_multiplication():
    # X*Y = iZ, Y*X = -iZ
    s1, p1 = pauli_string_multiply("X", "Y")
    assert s1 == "Z" and abs(p1 - 1j) < 1e-12
    s2, p2 = pauli_string_multiply("Y", "X")
    assert s2 == "Z" and abs(p2 + 1j) < 1e-12
    # "XY" * "YX" = (X*Y)(Y*X) = (iZ)(-iZ) = ZZ, phase = i*(-i) = 1
    s3, p3 = pauli_string_multiply("XY", "YX")
    assert s3 == "ZZ" and abs(p3 - 1.0) < 1e-12


def test_greedy_grouping():
    terms = ["ZZI", "IZZ", "ZIZ", "XII"]
    groups = greedy_group(terms)
    assert all(len(g) >= 1 for g in groups)
    assert n_groups(terms) >= 1


def test_trotter_depth():
    assert trotter_depth(["A", "B", "C"], order=1) == 3
    assert trotter_depth(["A", "B"], order=2) == 3


def test_repgraph():
    rg = RepresentationGraph()
    state = WQIRState(state_id="s1", level=WQIRLevel.HAMILTONIAN, problem="MAXCUT")
    rg.add_state(state)
    rg.add_edge(RepGraphEdge("s1", "s2", "GROUP_PAULIS", {"G2": -4}, 0.2, "EXACT", 0.8))
    assert len(rg.edges) == 1
    assert rg.edges_from("s1")[0].action_id == "GROUP_PAULIS"
