"""Tests for WQT20 data generators and trajectories."""
import random
import pytest
from wqt20.data.generators import (
    random_graph, maxcut_hamiltonian, mwis_hamiltonian, tfim_hamiltonian,
    action_value, oracle_action, serialize_state,
)
from wqt20.data.trajectories import generate_dataset, trajectory_to_examples, build_training_examples
from ecosystem.verify import n_groups


def test_random_graph_er():
    rng = random.Random(42)
    g = random_graph(6, 0.5, "ER", rng)
    assert g.n == 6
    assert all(0 <= u < 6 and 0 <= v < 6 for u, v in g.edges)


def test_maxcut_hamiltonian():
    rng = random.Random(42)
    g = random_graph(5, 0.6, "ER", rng)
    h = maxcut_hamiltonian(g, rng)
    assert h.problem == "MAXCUT"
    assert h.n_qubits == 5
    assert h.n_terms() == len(g.edges)
    # all terms should be ZZ strings
    for t in h.terms:
        assert "Z" in t


def test_action_value_signal():
    rng = random.Random(42)
    g = random_graph(8, 0.5, "ER", rng)
    h = maxcut_hamiltonian(g, rng)
    # GROUP_PAULIS should have higher value when there's compression
    _, _, v_group = action_value(h, "GROUP_PAULIS")
    _, _, v_stop = action_value(h, "STOP")
    assert v_group > v_stop


def test_oracle_action():
    rng = random.Random(42)
    g = random_graph(6, 0.5, "ER", rng)
    h = maxcut_hamiltonian(g, rng)
    legal = ["GROUP_PAULIS", "TROTTERIZE", "REORDER_TERMS", "STOP"]
    oracle = oracle_action(h, legal)
    assert oracle in legal


def test_generate_dataset():
    trajs = generate_dataset(10, seed=42)
    assert len(trajs) == 10
    for t in trajs:
        assert len(t.steps) > 0


def test_trajectory_to_examples():
    trajs = generate_dataset(5, seed=42)
    examples = []
    for t in trajs:
        examples.extend(trajectory_to_examples(t))
    assert len(examples) > 0
    assert "text" in examples[0]
    assert "action_token" in examples[0]
    assert "value" in examples[0]


def test_serialize_state():
    rng = random.Random(42)
    g = random_graph(4, 0.5, "ER", rng)
    h = maxcut_hamiltonian(g, rng)
    s = serialize_state(h)
    assert "<STATE>" in s
    assert "<LEVEL:HAMILTONIAN>" in s
    assert "<PROBLEM:MAXCUT>" in s
