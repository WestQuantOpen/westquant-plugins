# WestQuant Open Source — MEGAPLAN

**Two parallel projects, one shared core.**

1. **WQT20** — the open, ~20M-parameter learned policy engine for quantum
   representation scheduling. Narrow by design: nearly all capacity goes to the
   mathematics and decisions needed for *representation, equivalence,
   scheduling, and cost prediction* — not general reasoning, chat, or code.
2. **WestQuant Open Ecosystem** — the distribution machine around the model:
   WQIR schema, transformation registry, deterministic verifier APIs,
   Representation Graph storage, SDK, adapters (Qiskit/TKET/PyZX/CUDA-Q),
   benchmarks, and datasets.

The fundamental design principle, preserved across both projects:

> **AI schedules. Deterministic mathematics executes. Independent verification certifies.**

WQT20 ranks which transformation is promising. The ecosystem executes it.
Verification decides what survives. That separation is non-negotiable.

---

## Why narrow, not broad

The earlier draft proposed a 350–500M quantum-native model with a 50/30/20 mix
of math / circuits / representations and room for general interaction. WQT20
deliberately rejects that. At 20M parameters there is no spare capacity for
breadth. The model must be a **compact learned policy** over a structured
representation space:

$$\pi_\theta(a_t \mid S_t, H, O)$$

Almost all parameters are spent on the math that supports representation
reasoning, equivalence, scheduling, and cost prediction. The 50/30/20 mix is
replaced by the staged curriculum (Part IX of the WQT20 spec): math
bootstrapping → representation specialization → policy training → search
distillation.

There is research support for the core idea: learned policies have searched
sequences of ZX rewrites (RL + GNN + tree search), and AlphaTensor-Quantum
shows learned search can discover new circuit-optimization solutions. WQT20's
hypothesis is narrower and testable: a 20M structured Transformer can learn a
transferable representation-scheduling policy.

---

## Repository layout

```
westquant/
├── wqt20/                 # the model
│   ├── tokenizer.py        # ~4096 BPE, quantum-native structured tokens
│   ├── model.py            # Llama-style ~20M, HF LlamaConfig
│   ├── data/               # generators, trajectories, provenance
│   ├── search/             # baselines + WQT20-guided search
│   ├── eval/               # recall@k, regret, pareto, structural splits
│   ├── train.py            # smoke → pilot → 1.0
│   └── tests/
├── ecosystem/             # the distribution machine
│   ├── wqir.py             # WestQuant IR schema (12 levels)
│   ├── actions.py          # transformation registry
│   ├── repgraph.py         # Representation Graph storage
│   ├── verify.py           # deterministic verifier API
│   └── sdk.py              # westquant.Search(...) public API
├── configs/                # smoke.yaml, pilot.yaml
├── scripts/                # build_tokenizer, run_smoke, run_eval
├── benchmarks/             # locked held-out splits
└── data/                   # generated datasets + DATA_PROVENANCE.jsonl
```

---

## Implementation order (Part XIX)

Do not train the full model first. Build in this order:

1. WQIR schema
2. Transformation registry
3. Exact deterministic transformation/verifier APIs
4. Representation Graph storage format
5. Benchmark harness
6. Dataset generators
7. Baseline search algorithms
8. WQT20 tokenizer
9. 2M–5M toy Transformer
10. Smoke dataset
11. **Verify that learned policy beats random on toy tasks** ← gate
12. Scale to 20M
13. Generate expensive oracle/search trajectories
14. Train WQT20-PILOT
15. Freeze evaluation methodology
16. Train WQT20-1.0
17. Integrate with Qiskit and TKET
18. Release open weights, code, data subset, benchmarks

No major compute is spent on WQT20 until steps 1–11 are proven.

---

## Research hypotheses (tested explicitly)

- **H1** Small specialized models learn quantum-representation policies
- **H2** Representation scheduling generalizes (size / instance / topology)
- **H3** Mathematical representation matters independently of gate-level opt
- **H4** Hardware changes the optimal representation path
- **H5** Search efficiency > action-imitation accuracy

---

## Release gates (Part XVI)

WQT20-1.0 may be described as an optimization policy only if it beats random
and fixed-heuristic search on locked held-out data, survives unseen-instance
evaluation, shows positive transfer on at least one OOD structural split, is
reproducible from public scripts, and equivalence-preserving transformations
are independently verified. If gates fail, release as an experimental research
checkpoint — do not claim practical optimization superiority.

---

## Status

- [x] Repo scaffold (wqt20 + ecosystem)
- [x] MEGAPLAN
- [x] WQIR schema (12 levels)
- [x] Transformation registry (31 actions, 12 families)
- [x] RepGraph format (JSONL)
- [x] Deterministic verifier (Pauli algebra, commutation, grouping, Trotter)
- [x] Tokenizer (~4561 tokens, quantum-native structured + BPE)
- [x] Model (19.06M, Llama-style, HF-compatible)
- [x] Data generators (MaxCut, MWIS, TFIM)
- [x] Baselines (random, fixed, greedy, frequency)
- [x] Eval harness (recall@k, regret, Pareto, 5 structural splits)
- [x] 24 tests passing
- [x] **Smoke training — H1 gate PASSED** (recall@1: 0.967 vs random 0.123)
- [ ] Generate oracle/search trajectories (Part XIX step 13)
- [ ] Train WQT20-PILOT (step 14)
- [ ] Freeze evaluation methodology (step 15)
- [ ] Train WQT20-1.0 (step 16)
- [ ] Integrate Qiskit + TKET (step 17)
- [ ] Open release (step 18)
