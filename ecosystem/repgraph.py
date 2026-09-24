"""Representation Graph storage.

G = (V, E). Nodes are states; edges are transformations with delta metrics and
verification status. Both good and bad edges are stored — the model must
understand dead ends and costly transformations, not just winners.

Storage is JSONL (one record per edge) for append-only, streaming, and easy
deduplication by canonical state hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .wqir import WQIRState, ResourceVector
from .actions import TransformationRegistry, REGISTRY


def canonical_hash(state: WQIRState) -> str:
    """Stable hash of the state's structured content (not the python id)."""
    payload = {
        "level": int(state.level),
        "problem": state.problem,
        "payload": state.payload,
        "backend": state.backend.name,
        "objective": state.objective.weights,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class RepGraphEdge:
    source_state_id: str
    target_state_id: str
    action_id: str
    delta: Dict[str, float] = field(default_factory=dict)
    compute_cost: float = 0.0
    verification: str = "UNKNOWN"
    objective_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepGraphNode:
    state_id: str
    problem: str
    level: int
    resources: Dict[str, float] = field(default_factory=dict)
    verification: str = "UNKNOWN"


class RepresentationGraph:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.nodes: Dict[str, RepGraphNode] = {}
        self.edges: List[RepGraphEdge] = []

    def add_state(self, state: WQIRState) -> str:
        sid = state.state_id or canonical_hash(state)
        if sid not in self.nodes:
            self.nodes[sid] = RepGraphNode(
                state_id=sid, problem=state.problem, level=int(state.level),
                resources=state.resources.as_dict(), verification=state.verification,
            )
        return sid

    def add_edge(self, edge: RepGraphEdge) -> None:
        self.edges.append(edge)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(edge.to_dict()) + "\n")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self.edges = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.edges.append(RepGraphEdge(**json.loads(line)))

    def edges_from(self, state_id: str) -> List[RepGraphEdge]:
        return [e for e in self.edges if e.source_state_id == state_id]

    def best_edge(self, state_id: str, minimize: bool = True) -> Optional[RepGraphEdge]:
        out = self.edges_from(state_id)
        if not out:
            return None
        key = lambda e: e.objective_value
        return min(out, key=key) if minimize else max(out, key=key)
