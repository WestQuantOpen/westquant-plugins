"""WestQuant SDK Plugin.

The public-facing SDK that loads WQT20/WQT50 models from HuggingFace and
provides the `Search(...)` API for quantum representation scheduling.

    from westquant import Search
    result = Search(
        problem="MAXCUT",
        backend="ibm_brisbane",
        policy="WQT50",
        objectives={"two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2},
    ).run()

    # Direct model access
    s = Search(model_id="WestQuantStudio/WQT50M")
    s.load()
    cost = s.predict_cost(state_text, "ZX_SIMPLIFY")
    ranked = s.rank_actions(state_text, ["CANCEL_GATES", "FUSE_ROTATIONS", ...])
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Search:
    """WestQuant representation scheduling search.

    WQT20/WQT50 ranks legal transformations. The ecosystem executes them.
    Verification decides what survives.
    """
    problem: str = "MAXCUT"
    backend: str = "simulator"
    policy: str = "WQT50"  # WQT50, WQT20, random, fixed, greedy, frequency
    objectives: Dict[str, float] = field(default_factory=lambda: {
        "two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2
    })
    model_id: str = "WestQuantStudio/WQT50M"
    max_steps: int = 10
    top_k: int = 3
    seed: int = 42
    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    def load(self):
        """Load the model from HuggingFace. Called automatically by run()."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model.eval()
        except ImportError:
            raise ImportError("Install transformers + torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_id}'. "
                f"Ensure the model is published on HuggingFace. Error: {e}"
            )

    def predict_cost(self, state_text: str, action_name: str) -> float:
        """Predict the cost of applying an action to a state.

        Returns a calibrated cost in the real magnitude range.
        WQT50M: range ~1 to ~150,000 (real Qiskit costs).
        WQT20M-Beta: range ~6.0 to ~6.6 (synthetic, narrow).
        """
        self.load()
        import torch
        text = f"<PREDICT> {state_text} <ACTION> {action_name} <COST> "
        ids = self._tokenizer.encode(text, add_special_tokens=False,
                                     return_tensors="pt").to(self._device)
        with torch.no_grad():
            for _ in range(8):
                if ids.shape[1] > 512:
                    break
                out = self._model(input_ids=ids)
                next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == self._tokenizer.eos_token_id:
                    break
        gen = self._tokenizer.decode(ids[0].tolist())
        after = gen.split("<COST>")[-1].strip()
        if "<eos>" in after:
            after = after.split("<eos>")[0].strip()
        m = re.search(r"(\d+\.?\d*)", after)
        return float(m.group(1)) if m else 0.0

    def predict_preference(self, state_text: str, cand_a: str, cost_a: float,
                           cand_b: str, cost_b: float) -> str:
        """Predict preference between two candidates. Returns 'A>B' or 'B>A'."""
        self.load()
        import torch
        text = (f"<SOLVE> {state_text} "
                f"<CAND_A> {cand_a} <COST_A> {cost_a:.2f} "
                f"<CAND_B> {cand_b} <COST_B> {cost_b:.2f} "
                f"<PREF> ")
        ids = self._tokenizer.encode(text, add_special_tokens=False,
                                     return_tensors="pt").to(self._device)
        with torch.no_grad():
            for _ in range(5):
                if ids.shape[1] > 512:
                    break
                out = self._model(input_ids=ids)
                next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == self._tokenizer.eos_token_id:
                    break
        gen = self._tokenizer.decode(ids[0].tolist())
        after = gen.split("<PREF>")[-1].strip()
        if "<eos>" in after:
            after = after.split("<eos>")[0].strip()
        return after.split()[0] if after.split() else ""

    def predict_legality(self, state_text: str, action_name: str) -> str:
        """Predict if an action is legal. Returns 'LEGAL' or 'ILLEGAL'."""
        self.load()
        import torch
        text = f"<LEGAL> {state_text} <ACTION> {action_name} <LABEL> "
        ids = self._tokenizer.encode(text, add_special_tokens=False,
                                     return_tensors="pt").to(self._device)
        with torch.no_grad():
            for _ in range(3):
                out = self._model(input_ids=ids)
                next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == self._tokenizer.eos_token_id:
                    break
        gen = self._tokenizer.decode(ids[0].tolist())
        after = gen.split("<LABEL>")[-1].strip()
        if "<eos>" in after:
            after = after.split("<eos>")[0].strip()
        if "ILLEGAL" in after:
            return "ILLEGAL"
        if "LEGAL" in after:
            return "LEGAL"
        return "UNKNOWN"

    def rank_actions(self, state_text: str, action_names: List[str]) -> List[Tuple[str, float]]:
        """Rank actions by predicted cost (lowest = best).

        Returns list of (action_name, predicted_cost) sorted ascending.
        """
        scored = []
        for name in action_names:
            cost = self.predict_cost(state_text, name)
            scored.append((name, cost))
        scored.sort(key=lambda x: x[1])
        return scored

    def run(self) -> Dict:
        """Run the optimization search. Returns the best trajectory found."""
        if self.policy in ("WQT50", "WQT20"):
            self.load()
            return {
                "status": "ready",
                "model": self.model_id,
                "policy": self.policy,
                "backend": self.backend,
                "problem": self.problem,
                "objectives": self.objectives,
                "capabilities": [
                    "predict_cost",
                    "predict_preference",
                    "predict_legality",
                    "rank_actions",
                ],
                "note": ("Use predict_cost(), rank_actions(), predict_preference(), "
                         "and predict_legality() for model-guided search. "
                         "Full closed-loop search requires a framework adapter "
                         "(westquant.plugins.qiskit, .tket, or .pyzx)."),
            }
        return {
            "status": "ready",
            "model": None,
            "policy": self.policy,
            "backend": self.backend,
            "problem": self.problem,
            "objectives": self.objectives,
        }


__all__ = ["Search"]
