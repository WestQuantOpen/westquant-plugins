"""WQT20 evaluation metrics (Part XIV).

Primary: search efficiency. Also: oracle recall@k, normalized search regret,
Pareto hypervolume, generalization gap, invalid rate, inference overhead.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def oracle_recall_at_k(
    predicted_ranking: List[str], oracle_action: str, k: int = 1,
) -> float:
    """1.0 if oracle is in top-k predicted actions, else 0.0."""
    return 1.0 if oracle_action in predicted_ranking[:k] else 0.0


def normalized_search_regret(
    wqt_objective: float, baseline_objective: float, best_objective: float,
    minimize: bool = True, eps: float = 1e-8,
) -> float:
    """R = (J_wqt - J_best) / (|J_baseline - J_best| + eps)."""
    if minimize:
        num = wqt_objective - best_objective
        den = abs(baseline_objective - best_objective) + eps
    else:
        num = best_objective - wqt_objective
        den = abs(best_objective - baseline_objective) + eps
    return float(num / den)


def top_k_accuracy(predicted: List[str], oracle: str, k: int = 1) -> float:
    return oracle_recall_at_k(predicted, oracle, k)


def mean_reciprocal_rank(rankings: List[List[str]], oracles: List[str]) -> float:
    """MRR over a set of queries."""
    total = 0.0
    for ranking, oracle in zip(rankings, oracles):
        for i, a in enumerate(ranking, 1):
            if a == oracle:
                total += 1.0 / i
                break
    return total / max(len(rankings), 1)


def pareto_hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Approximate 2-D Pareto hypervolume (minimization) relative to ref."""
    from ecosystem.wqir import ResourceVector
    # filter dominated points
    pts = points.copy()
    mask = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                mask[i] = False
    front = pts[mask]
    if len(front) == 0:
        return 0.0
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev = ref.copy()
    for p in front:
        hv += (prev[0] - p[0]) * (prev[1] - p[1])
        prev = p.copy()
    return float(hv)


def evaluate_policy(
    examples: List[Dict],
    predict_fn,
    k: int = 1,
) -> Dict:
    """Evaluate a policy on a set of examples.

    predict_fn(example) -> List[str] (ranking of actions)
    Each example has 'legal_actions' (implicit from training) and 'oracle_action'.
    """
    recalls = []
    for ex in examples:
        oracle = ex.get("oracle_action") or ex.get("action")
        legal = ex.get("legal_actions", [])
        if not legal:
            continue
        ranking = predict_fn(ex)
        recalls.append(oracle_recall_at_k(ranking, oracle, k))
    return {
        "recall_at_1": float(np.mean(recalls)) if recalls else 0.0,
        "n_examples": len(recalls),
        "k": k,
    }


def search_efficiency(
    wqt_final_objective: float, baseline_final_objective: float,
    wqt_n_evals: int, baseline_n_evals: int,
) -> Dict:
    """Improvement per expensive evaluation."""
    improvement = baseline_final_objective - wqt_final_objective
    return {
        "improvement": float(improvement),
        "wqt_evals": wqt_n_evals,
        "baseline_evals": baseline_n_evals,
        "improvement_per_eval_wqt": float(improvement / max(wqt_n_evals, 1)),
        "eval_reduction": float(1 - wqt_n_evals / max(baseline_n_evals, 1)),
    }
