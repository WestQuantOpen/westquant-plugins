"""WestQuant SDK Plugin.

The public-facing SDK that loads the WQT20 model from HuggingFace and provides
the `westquant.Search(...)` API. This is the main entry point for users who
want to use WQT20 without the full training/development stack.

    from westquant import Search
    result = Search(
        problem=problem,
        backend="ibm_brisbane",
        policy="WQT20",
        objectives={"two_qubit_gates": 0.5, "depth": 0.3, "estimated_error": 0.2},
    ).run()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    model_id: str = "westquant/WQT20-1.0"
    max_steps: int = 10
    top_k: int = 3
    seed: int = 42
    _model: Any = None
    _tokenizer: Any = None

    def _load_model(self):
        """Load WQT20 from HuggingFace."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._device = device
        except ImportError:
            raise ImportError("pip install transformers torch")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load WQT20 model '{self.model_id}'. "
                f"Ensure the model is published on HuggingFace. Error: {e}"
            )

    def run(self) -> Dict:
        """Run the optimization search. Returns the best trajectory found."""
        if self.policy == "WQT20":
            self._load_model()
        # TODO: full search pipeline
        return {
            "status": "not_yet_implemented",
            "model": self.model_id,
            "policy": self.policy,
            "backend": self.backend,
            "problem": self.problem,
            "objectives": self.objectives,
        }


__all__ = ["Search"]
