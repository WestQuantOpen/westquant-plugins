"""WestQuant public SDK.

    from westquant import Search
    result = Search(problem=problem, backend=backend, policy="WQT20",
                    objectives={"G2": 0.5, "D": 0.3, "E": 0.2}).run()

The model ranks paths. The ecosystem executes them. Verification decides what
survives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .wqir import WQIRState, WQIRLevel, BackendSpec, ObjectiveSpec, ResourceVector
from .actions import TransformationRegistry, REGISTRY
from .repgraph import RepresentationGraph


@dataclass
class SearchConfig:
    problem: str = "MAXCUT"
    backend: BackendSpec = field(default_factory=BackendSpec)
    policy: str = "random"  # random, fixed, greedy, frequency, WQT20
    objectives: Dict[str, float] = field(default_factory=lambda: {"G2": 0.5, "D": 0.3, "E": 0.2})
    max_steps: int = 10
    top_k: int = 3
    seed: int = 42


@dataclass
class SearchResult:
    trajectory: List[str] = field(default_factory=list)
    final_state: Optional[WQIRState] = None
    final_objective: float = float("inf")
    n_evaluations: int = 0
    repgraph: Optional[RepresentationGraph] = None


class Search:
    def __init__(self, problem: str = "MAXCUT", backend: Optional[BackendSpec] = None,
                 policy: str = "random", objectives: Optional[Dict[str, float]] = None,
                 max_steps: int = 10, top_k: int = 3, seed: int = 42):
        self.config = SearchConfig(
            problem=problem,
            backend=backend or BackendSpec(),
            policy=policy,
            objectives=objectives or {"G2": 0.5, "D": 0.3, "E": 0.2},
            max_steps=max_steps, top_k=top_k, seed=seed,
        )
        self.registry = REGISTRY
        self.repgraph = RepresentationGraph()

    def run(self) -> SearchResult:
        from ..wqt20.search.baselines import select_action

        state = WQIRState(
            state_id="root", level=WQIRLevel.HAMILTONIAN, problem=self.config.problem,
            backend=self.config.backend,
            objective=ObjectiveSpec(self.config.objectives),
        )
        traj = [state.state_id]
        n_eval = 0
        for _ in range(self.config.max_steps):
            legal = self.registry.legal_actions(state)
            if not legal or legal == ["STOP"]:
                break
            action = select_action(self.config.policy, legal, state, self.config.seed)
            n_eval += 1
            if action == "STOP":
                break
            traj.append(action)
        return SearchResult(
            trajectory=traj, final_state=state, n_evaluations=n_eval,
            repgraph=self.repgraph,
        )
