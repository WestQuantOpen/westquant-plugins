"""Baseline search policies (Part XIII).

random, fixed, greedy, frequency — and a WQT20-guided policy that uses the
model's action logits over legal actions.
"""
from __future__ import annotations

import random
from typing import List, Optional

from ecosystem.wqir import WQIRState


def select_action(policy: str, legal: List[str], state: WQIRState, seed: int = 42) -> str:
    """Select an action from the legal set according to a baseline policy."""
    if not legal:
        return "STOP"
    if policy == "random":
        rng = random.Random(seed + hash(state.state_id) % 10000)
        return rng.choice(legal)
    elif policy == "fixed":
        # fixed rule order: first legal action
        return legal[0]
    elif policy == "greedy":
        # greedy immediate metric improvement: pick the action with best
        # immediate value (uses the deterministic oracle when available)
        from wqt20.data.generators import action_value
        h = getattr(state, "_hamiltonian", None)
        if h is not None:
            return max(legal, key=lambda a: action_value(h, a)[2])
        return legal[0]
    elif policy == "frequency":
        # pick the most common action in a fixed frequency table
        freq_order = ["GROUP_PAULIS", "TROTTERIZE", "REORDER_TERMS",
                      "CHANGE_ENCODING", "TAPER_SYMMETRY", "CHANGE_BASIS",
                      "CANONICALIZE", "STOP"]
        for a in freq_order:
            if a in legal:
                return a
        return legal[0]
    elif policy == "WQT20":
        # handled by policy.py (model-guided); fallback to random here
        rng = random.Random(seed)
        return rng.choice(legal)
    return legal[0]


def run_search(policy: str, legal_fn, state, max_steps: int = 10, seed: int = 42) -> List[str]:
    """Run a search rollout, returning the action sequence."""
    actions = []
    for _ in range(max_steps):
        legal = legal_fn(state)
        if not legal or legal == ["STOP"]:
            break
        a = select_action(policy, legal, state, seed)
        if a == "STOP":
            break
        actions.append(a)
    return actions
