"""Tests for WQT20 evaluation metrics and splits."""
import numpy as np
import pytest
from wqt20.eval.metrics import (
    oracle_recall_at_k, normalized_search_regret, mean_reciprocal_rank,
    search_efficiency,
)
from wqt20.eval.splits import SPLITS, make_all_splits, SplitSpec


def test_recall_at_k():
    assert oracle_recall_at_k(["A", "B", "C"], "A", k=1) == 1.0
    assert oracle_recall_at_k(["B", "A", "C"], "A", k=1) == 0.0
    assert oracle_recall_at_k(["B", "A", "C"], "A", k=2) == 1.0


def test_normalized_regret():
    r = normalized_search_regret(wqt_objective=0.5, baseline_objective=0.8,
                                  best_objective=0.4, minimize=True)
    assert r > 0  # wqt is better than baseline


def test_mean_reciprocal_rank():
    # [["A","B"],["B","A"]] with oracles ["A","A"]:
    # first: A at rank 1 -> 1/1, second: A at rank 2 -> 1/2. MRR = 0.75
    mrr = mean_reciprocal_rank([["A", "B"], ["B", "A"]], ["A", "A"])
    assert mrr == 0.75
    # oracle at rank 1 in both -> MRR = 1.0
    mrr2 = mean_reciprocal_rank([["A", "B"], ["A", "B"]], ["A", "A"])
    assert mrr2 == 1.0


def test_search_efficiency():
    eff = search_efficiency(wqt_final_objective=0.4, baseline_final_objective=0.8,
                            wqt_n_evals=10, baseline_n_evals=50)
    assert eff["improvement"] > 0
    assert eff["eval_reduction"] > 0


def test_splits_exist():
    names = [s.name for s in SPLITS]
    assert "unseen_size" in names
    assert "unseen_graph_family" in names
    assert "unseen_problem" in names
    assert "unseen_backend" in names
    assert len(SPLITS) >= 5


def test_make_all_splits():
    examples = [
        {"family": "ER", "n_qubits": 5, "problem": "MAXCUT", "backend": "SIMULATOR", "seed": 100},
        {"family": "grid", "n_qubits": 9, "problem": "TFIM", "backend": "SUPERCONDUCTING", "seed": 900},
    ]
    splits = make_all_splits(examples)
    for name, (train, test) in splits.items():
        assert isinstance(train, list)
        assert isinstance(test, list)
