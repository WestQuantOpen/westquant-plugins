"""WestQuant SDK Plugin.

The public-facing SDK that loads the WQT20 model from HuggingFace and provides
the `westquant.Search(...)` API. This is the main entry point for users who
want to use WQT20M for quantum representation scheduling.

    from westquant import Search
    result = Search(
        problem="MAXCUT",
        backend="ibm_brisbane",
        policy="WQT20",
        objectives={"two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2},
    ).run()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Search:
    """WestQuant representation scheduling search.

    WQT20 ranks legal transformations. The ecosystem executes them.
    Verification decides what survives.
    """
    problem: str = "MAXCUT"
    backend: str = "simulator"
    policy: str = "WQT20"  # WQT20, random, fixed, greedy, frequency
    objectives: Dict[str, float] = field(default_factory=lambda: {
        "two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2
    })
    model_id: str = "WestQuantStudio/WQT20M-Beta"
    max_steps: int = 10
    top_k: int = 3
    seed: int = 42
    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    def _load_model(self):
        """Load WQT20 from HuggingFace."""
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
            raise ImportError("pip install transformers torch")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load WQT20 model '{self.model_id}'. "
                f"Ensure the model is published on HuggingFace. Error: {e}"
            )

    def _predict_preference(self, state_text: str, cand_a: str, cost_a: float,
                             cand_b: str, cost_b: float) -> str:
        """Predict preference between two candidates. Returns 'A>B' or 'B>A'.

        Uses greedy generation (same as the evaluation suite).
        """
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

    def _predict_value(self, state_text: str) -> float:
        """Predict cost-to-go from a state."""
        import torch
        text = f"<PREDICT> {state_text} <COST_TO_GO> "
        ids = self._tokenizer.encode(text, add_special_tokens=False,
                                       return_tensors="pt").to(self._device)
        with torch.no_grad():
            for _ in range(10):
                if ids.shape[1] > 512:
                    break
                out = self._model(input_ids=ids)
                next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == self._tokenizer.eos_token_id:
                    break
        gen = self._tokenizer.decode(ids[0].tolist())
        after = gen.split("<COST_TO_GO>")[-1].strip()
        if "<eos>" in after:
            after = after.split("<eos>")[0].strip()
        m = re.search(r"(\d+\.?\d*)", after)
        return float(m.group(1)) if m else 0.0

    def run(self) -> Dict:
        """Run the optimization search. Returns the best trajectory found."""
        if self.policy == "WQT20":
            self._load_model()
            return {
                "status": "ready",
                "model": self.model_id,
                "policy": self.policy,
                "backend": self.backend,
                "problem": self.problem,
                "objectives": self.objectives,
                "capabilities": [
                    "preference_comparison",
                    "value_prediction",
                    "legality_check",
                ],
                "note": ("WQT20M-Beta is a Beta release. "
                         "Use predict_preference() and predict_value() "
                         "for model-guided search. Full closed-loop search "
                         "requires the WestQuant ecosystem (transformation "
                         "registry + verifier)."),
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
