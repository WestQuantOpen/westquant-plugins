"""WQT20-guided search policy.

Uses the model's logits over legal action tokens to score and rank actions.
The search system (not the model) owns branching and rollback.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def score_actions(
    model,
    tokenizer,
    state_text: str,
    legal_text: str,
    legal_actions: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """Score each legal action using the model's next-token logits over
    the <A:...> action tokens.

    Returns {action_id: score} (softmax probability over legal actions).
    """
    import torch

    prompt = state_text + " " + legal_text + " <NEXT> "
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=2048, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]  # next-token logits

    # gather logits for each legal action token
    action_logits = []
    for a in legal_actions:
        tok = f"<A:{a}>"
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            action_logits.append(-1e9)
        else:
            action_logits.append(logits[tid].item())
    action_logits = np.array(action_logits)
    # softmax over legal actions only
    action_logits = action_logits - action_logits.max()
    probs = np.exp(action_logits)
    probs = probs / probs.sum()
    return {a: float(p) for a, p in zip(legal_actions, probs)}


def select_wqt20_action(
    model, tokenizer, state_text: str, legal_text: str,
    legal_actions: List[str], device: str = "cpu", top_k: int = 1,
) -> Tuple[str, Dict[str, float]]:
    """Select the top-k action(s) from the model. Returns (action, scores)."""
    scores = score_actions(model, tokenizer, state_text, legal_text,
                           legal_actions, device)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[0][0], scores
