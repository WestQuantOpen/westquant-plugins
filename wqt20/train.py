"""WQT20 training: smoke -> pilot -> 1.0.

The smoke stage (Part X) validates the architecture and data pipeline, and
tests H1: that a small specialized model can learn a representation-scheduling
policy that beats random selection.

Training task: next-action prediction. Given (state_text + legal_text), predict
the highest-value action token. The label is the oracle action (highest
deterministic value). Loss = cross-entropy over the action-token vocabulary
positions, but implemented as standard causal LM loss on the serialized
"<state> <legal> <NEXT> <A:ACTION>" sequence so the model learns to emit the
best action autoregressively.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import build_model, build_tokenizer_hf, count_parameters, save_model
from .data.trajectories import build_training_examples, generate_dataset, trajectory_to_examples
from .data.generators import oracle_action, action_value, serialize_state, serialize_legal_actions, serialize_action_value
from .eval.metrics import oracle_recall_at_k, mean_reciprocal_rank
from .eval.splits import make_all_splits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WQT20Dataset(Dataset):
    """Next-action prediction dataset.

    Each example is serialized as:
        <state_text> <legal_text> <NEXT> <A:oracle_action>
    and trained with causal LM loss. The model learns to predict the oracle
    action token given the state + legal context.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        # cache action token ids
        self.action_token_ids = {}
        for ex in examples:
            tok = ex["action_token"]
            if tok not in self.action_token_ids:
                self.action_token_ids[tok] = tokenizer.convert_tokens_to_ids(tok)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Build the prompt + target sequence
        prompt = ex["text"] + " <NEXT> "
        target = ex["action_token"]
        full = prompt + target + " <eos>"

        ids = self.tokenizer(full, return_tensors="pt", truncation=True,
                             max_length=self.max_length, add_special_tokens=False)["input_ids"][0]
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        # labels: mask the prompt, only train on the target tokens
        labels = ids.clone()
        labels[:prompt_len] = -100
        return {
            "input_ids": ids,
            "labels": labels,
        }


def collate_fn(batch, pad_id=0):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        a = torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        attn.append(a)
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_trajectories: int = 2000,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_length: int = 512,
    seed: int = 42,
    device: str = "auto",
    eval_split: float = 0.1,
    out_dir: str = "checkpoints/wqt20-smoke",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[wqt20] device: {device}")

    # 1. Tokenizer + model
    print("[wqt20] building tokenizer...")
    tokenizer = build_tokenizer_hf()
    vocab_size = len(tokenizer)  # includes added atomic tokens
    print(f"[wqt20] vocab size: {vocab_size}")

    print("[wqt20] building model...")
    model = build_model(vocab_size=vocab_size)
    info = count_parameters(model)
    print(f"[wqt20] params: {info['total_M']:.2f}M")
    model.to(device)

    # 2. Data
    print(f"[wqt20] generating {n_trajectories} trajectories...")
    trajs = generate_dataset(n_trajectories, seed=seed)
    examples = []
    for t in trajs:
        examples.extend(trajectory_to_examples(t))
    print(f"[wqt20] {len(examples)} training examples")

    # attach legal_actions + oracle for eval
    legal_by_state = {}
    for ex in examples:
        key = ex["text"]
        if key not in legal_by_state:
            legal_by_state[key] = []
        legal_by_state[key].append(ex["action"])
    for ex in examples:
        ex["legal_actions"] = legal_by_state[ex["text"]]
        # oracle = highest value action
        oracle = max(legal_by_state[ex["text"]],
                     key=lambda a: next(e["value"] for e in examples
                                       if e["text"] == ex["text"] and e["action"] == a))
        ex["oracle_action"] = oracle

    # train/eval split (random for smoke; structural splits available via eval.splits)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_split))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]
    print(f"[wqt20] train={len(train_examples)} eval={len(eval_examples)}")

    train_ds = WQT20Dataset(train_examples, tokenizer, max_length)
    eval_ds = WQT20Dataset(eval_examples, tokenizer, max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                         collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    # 3. Optimizer + cosine schedule with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))
    total_steps = len(train_dl) * epochs
    warmup_steps = min(50, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Train
    print("[wqt20] training...")
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")
        # eval loss
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                eval_losses.append(out.loss.item())
        el = np.mean(eval_losses) if eval_losses else 0.0
        print(f"[wqt20] epoch {epoch} eval_loss {el:.4f}")
        model.train()

    dt = time.time() - t0
    print(f"[wqt20] training done in {dt:.1f}s")

    # 5. Evaluate: recall@1 vs random
    print("[wqt20] evaluating policy vs random...")
    recall_wqt = evaluate_policy_recall(model, tokenizer, eval_examples, device, k=1)
    recall_random = evaluate_random_recall(eval_examples, k=1)
    print(f"[wqt20] WQT20 recall@1: {recall_wqt:.4f}")
    print(f"[wqt20] random  recall@1: {recall_random:.4f}")

    beats_random = recall_wqt > recall_random
    print(f"[wqt20] H1 gate (beats random): {'PASS' if beats_random else 'FAIL'}")

    # 6. Save
    out = Path(out_dir)
    save_model(model, tokenizer, out)
    print(f"[wqt20] saved to {out}")

    return {
        "params_M": info["total_M"],
        "train_loss": loss.item(),
        "eval_loss": el,
        "recall_at_1_wqt": recall_wqt,
        "recall_at_1_random": recall_random,
        "h1_gate_pass": beats_random,
        "train_time_s": dt,
    }


def evaluate_policy_recall(model, tokenizer, examples: List[Dict], device: str, k: int = 1) -> float:
    """Compute recall@k: does the model rank the oracle action in top-k?"""
    from .search.policy import score_actions
    model.eval()
    # group examples by state text (which already contains state + legal)
    states = {}
    for ex in examples:
        states.setdefault(ex["text"], {"legal": ex["legal_actions"], "oracle": ex["oracle_action"]})
    recalls = []
    with torch.no_grad():
        for state_text, info in states.items():
            legal = info["legal"]
            oracle = info["oracle"]
            # state_text already contains the legal block; pass empty legal_text
            # to avoid duplication. score_actions prepends legal_text after state.
            scores = score_actions(model, tokenizer, state_text, "", legal, device)
            ranking = sorted(scores.keys(), key=lambda a: -scores[a])
            recalls.append(oracle_recall_at_k(ranking, oracle, k))
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_random_recall(examples: List[Dict], k: int = 1) -> float:
    """Random baseline recall@k: 1/|legal| on average."""
    states = {}
    for ex in examples:
        states.setdefault(ex["text"], {"legal": ex["legal_actions"], "oracle": ex["oracle_action"]})
    recalls = []
    rng = random.Random(123)
    for _, info in states.items():
        legal = info["legal"]
        oracle = info["oracle"]
        ranking = legal.copy()
        rng.shuffle(ranking)
        recalls.append(oracle_recall_at_k(ranking, oracle, k))
    return float(np.mean(recalls)) if recalls else 0.0


def main():
    parser = argparse.ArgumentParser(description="WQT20 smoke training")
    parser.add_argument("--n", type=int, default=2000, help="number of trajectories")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints/wqt20-smoke")
    args = parser.parse_args()
    result = train(
        n_trajectories=args.n, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, out_dir=args.out,
    )
    print("\n=== WQT20 SMOKE RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
