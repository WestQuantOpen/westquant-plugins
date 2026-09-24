"""WQT20 structural held-out splits (Part XV).

Split by STRUCTURE, not random rows. Required held-out splits:
unseen instance, unseen graph, unseen circuit size, unseen algorithm parameter
range, unseen hardware topology, unseen backend gate set, unseen representation
combination, unseen transformation sequence, cross-tool transfer, long-horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SplitSpec:
    name: str
    description: str
    train_filter: dict   # attribute -> allowed values (or range tuple)
    test_filter: dict


SPLITS: List[SplitSpec] = [
    SplitSpec("unseen_instance", "Hold out specific graph instances (by seed range)",
              train_filter={"seed": (0, 800)}, test_filter={"seed": (800, 1000)}),
    SplitSpec("unseen_graph_family", "Train on ER/3regular/cycle, test on grid/path",
              train_filter={"family": ["ER", "3regular", "cycle"]},
              test_filter={"family": ["grid", "path"]}),
    SplitSpec("unseen_size", "Train on small circuits, test on larger",
              train_filter={"n_qubits": (4, 7)}, test_filter={"n_qubits": (8, 10)}),
    SplitSpec("unseen_problem", "Train on MAXCUT/MWIS, test on TFIM",
              train_filter={"problem": ["MAXCUT", "MWIS"]},
              test_filter={"problem": ["TFIM"]}),
    SplitSpec("unseen_backend", "Train on SIMULATOR, test on hardware backends",
              train_filter={"backend": ["SIMULATOR"]},
              test_filter={"backend": ["SUPERCONDUCTING", "TRAPPED_ION"]}),
]


def _match(ex: Dict, filt: Dict) -> bool:
    for key, val in filt.items():
        ev = ex.get(key)
        if isinstance(val, tuple) and len(val) == 2:
            if not (val[0] <= ev < val[1]):
                return False
        elif isinstance(val, list):
            if ev not in val:
                return False
    return True


def make_split(examples: List[Dict], split: SplitSpec) -> Tuple[List[Dict], List[Dict]]:
    train = [ex for ex in examples if _match(ex, split.train_filter)]
    test = [ex for ex in examples if _match(ex, split.test_filter)]
    return train, test


def make_all_splits(examples: List[Dict]) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    return {s.name: make_split(examples, s) for s in SPLITS}
