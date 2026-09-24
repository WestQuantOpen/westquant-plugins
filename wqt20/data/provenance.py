"""Data provenance ledger (Part VIII).

Every record must include provenance. Stored as JSONL in DATA_PROVENANCE.jsonl.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Provenance:
    dataset_version: str = "wqt-smoke-0.1"
    generator_version: str = "wqt20.data.trajectories:0.1"
    tool: str = "wqt20"
    tool_version: str = "0.1.0"
    seed: int = 42
    source_family: str = "synthetic"
    license: str = "Apache-2.0"
    verification_status: str = "deterministic"
    backend_spec: str = "SIMULATOR"
    objective_spec: str = "G2:0.5,D:0.3,E:0.2"
    parent_state_id: str = ""
    action_id: str = ""
    child_state_id: str = ""


class ProvenanceLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: List[Dict[str, Any]] = []

    def add(self, prov: Provenance, extra: Dict[str, Any] = None) -> None:
        rec = asdict(prov)
        if extra:
            rec.update(extra)
        self.records.append(rec)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self.records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)
