# WQT50M — Plan for 50M Parameter Model with 1B Records

## Vision

> "AI schedules. Deterministic mathematics executes. Independent verification certifies."

WQT50M is the next major scale-up after WQT20M. It targets:
- **50M parameters** (2.5x WQT20M)
- **1 Billion unique training records** (10x WQT20M)
- **10M+ problems** from 30 niche domains
- **Deep optimization rounds** from actively running WestQuant Studio

## Why 50M?

WQT5M ≈ WQT10M on the beta2 evaluation — the bottleneck was data, not capacity.
WQT20M will test whether 19M params + 100M records (with cost-visible preferences)
finally breaks through the anti-cheating and search-improvement gates.

If WQT20M passes:
- WQT50M scales up to handle harder problems and deeper search
- More capacity for cross-domain transfer between 30 niches
- Better long-horizon planning for multi-step optimization

If WQT20M fails:
- The failure mode tells us whether to invest in data quality (more domains,
  deeper trajectories) or architecture (attention patterns, memory)
- WQT50M plan pivots accordingly

## 30 Niche Domains

### Core quantum computing (15 domains — carried from WQT20M)

| # | Domain | Key actions | State properties |
|---|--------|-------------|-----------------|
| 1 | Graph optimization | MAXCUT, COLOR, MIS, TSP, SDP_RELAX | n_nodes, n_edges, density, chromatic |
| 2 | Quantum chemistry | UCCSD, ADAPT_VQE, GIVENS, FROZEN_CORE | n_electrons, n_orbitals, excitation_level |
| 3 | Error correction | SYNDROME, DECODE_SURFACE, MAGIC_DISTILL | code_distance, n_logical, error_rate |
| 4 | Quantum annealing | REVERSE_ANNEAL, MINOR_EMBED, HYBRID_SOLVE | n_variables, chain_strength, anneal_time |
| 5 | Variational | QAOA_P1/P2/P3, WARM_START, ADAPTIVE_LAYER | p_depth, n_params, mixer_type |
| 6 | Hamiltonian simulation | TROTTER, LCU, QUBITIZE, QDRIFT | n_terms, max_weight, sim_time |
| 7 | Circuit optimization | CANCEL, ZX_SIMPLIFY, FUSE_ROT, TEMPLATE | n_gates, n_2q, depth |
| 8 | Hardware mapping | SABRE_ROUTE, PULSE_OPT, NOISE_AWARE | n_logical, n_physical, topology |
| 9 | Neutral atom | SET_RYDBERG, PULSE_SHAPE, ATOM_REARRANGE | n_atoms, spacing, detuning |
| 10 | Photonic | KLM_CNOT, CLUSTER_STATE, FUSION_MEASURE | n_modes, n_photons, loss_rate |
| 11 | Topological | BRAID_ANYON, FIBONACCI_BRAID | n_anyons, braid_length |
| 12 | Quantum walk | COINED_WALK, GROVER_COIN, AMPLIFY | n_positions, walk_time, coin_dim |
| 13 | State preparation | MPS_PREP, TREE_TENSOR, COMPRESS_STATE | n_qubits, entanglement, bond_dim |
| 14 | Amplitude amplification | GROVER_ITER, FIXED_POINT, ITERATIVE_QPE | n_solutions, n_items, iterations |
| 15 | Quantum ML | IQP_KERNEL, DATA_REUPLOAD, FIDELITY_KERNEL | n_features, n_layers, kernel_type |

### New niche domains (15 domains — WQT50M expansion)

| # | Domain | Key actions | State properties | Why it matters |
|---|--------|-------------|-----------------|----------------|
| 16 | **Pulse-level optimization** | OPTIMIZE_AMPLITUDE, SHAPE_PULSE, CALIBRATE_PHASE, DRAG_CORRECT, DERIVATIVE_REMOVAL | pulse_duration, amplitude, frequency, phase, n_channels | Pulse-level control is the lowest abstraction layer. WQT50M should learn when to go to pulse level vs gate level. |
| 17 | **Quantum compilation** | DECOMPOSE_TOFFOLI, DECOMPOSE_MULTI_QUBIT, SYNTHESIZE_UNITARY, KAK_DECOMPOSE, CARTAN_DECOMPOSE | n_qubits, unitary_dim, entangling_power | Multi-qubit gate decomposition is the core of compilation. |
| 18 | **Quantum network routing** | ENTANGLEMENT_SWAP, PURIFY_BELL, ROUTE_PHOTON, CHAIN_SWAP, DISTRIBUTE_EPR | n_nodes, n_links, fidelity_threshold, distance | Distributed quantum computing requires network-aware routing. |
| 19 | **Quantum cryptography** | BB84_ENCODE, E91_PROTOCOL, QKD_KEY_DISTILL, PRIVACY_AMPLIFY, DECOY_STATE | key_length, error_rate, eavesdrop_detected | Security protocols have strict verification requirements. |
| 20 | **Quantum sensing** | RAMSEY_SEQUENCE, SPIN_ECHO, DYNAMICAL_DECOUPLING, GRADIOMETRY, ENTANGLE_SENSORS | sensor_count, sensitivity, coherence_time, noise_floor | Sensing optimization has different objectives (SNR, sensitivity). |
| 21 | **Quantum thermodynamics** | THERMALIZE, COOL_SYSTEM, WORK_EXTRACTION, ERGOTROPIC_GAP, CATALYST_COOLING | n_qubits, temperature, entropy, work_capacity | Thermodynamic protocols have unique constraints. |
| 22 | **Quantum metrology** | PARALLEL_STRATEGY, ADAPTIVE_PHASE, ENTANGLE_METER, BAYESIAN_UPDATE, HEISENBERG_LIMIT | n_probes, iterations, precision_target, phase_uncertainty | Metrology pushes precision limits. |
| 23 | **Quantum simulation (lattice)** | GAUGE_FIELD, WILSON_LOOP, LATTICE_GAUGE, CONTINUUM_LIMIT, TOPOLOGICAL_SECTOR | lattice_size, n_gauge_fields, beta_coupling, topology | Lattice gauge theory is a major HEP application. |
| 24 | **Quantum finance** | PORTFOLIO_OPT, RISK_MEASURE, OPTION_PRICE, MONTE_CARLO_Q, AMPLITUDE_EST_Q | n_assets, n_paths, confidence_level, variance | Finance has real-world QPU applications. |
| 25 | **Quantum optimization (classical)** | BRANCH_CUT, HEURISTIC_SEARCH, LAGRANGIAN_RELAX, COLUMN_GEN, PRIMAL_DUAL | n_constraints, n_variables, integrality_gap, duality_gap | Classical optimization enhanced by quantum subroutines. |
| 26 | **Quantum supremacy sampling** | RANDOM_CIRCUIT, BOSON_SAMPLE, IQP_SAMPLE, RAVEROS_SAMPLE, VERIFY_SAMPLE | n_qubits, depth, n_samples, verification_method | Sampling tasks have different optimization targets. |
| 27 | **Quantum error mitigation** | ZERO_NOISE_EXTRAP, PROBABILISTIC_CANCEL, MEASUREMENT_MITIGATE, CLIFFORD_DATA_REGRESS, VIRTUAL_DISTILL | n_shots, noise_level, mitigation_overhead, fidelity_gain | Error mitigation is critical for NISQ. |
| 28 | **Quantum memory** | STORE_QUBIT, RETRIEVE_QUBIT, COHERENCE_EXTEND, SWAP_TO_MEMORY, ERROR_CORRECT_MEMORY | storage_time, fidelity, n_memory_cells, error_rate | Quantum memory is the bottleneck for networks. |
| 29 | **Quantum channel coding** | ENCODE_CHANNEL, DECODE_CHANNEL, CAPACITY_ACHIEVE, POLAR_CODE, ENTANGLEMENT_ASSIST | channel_capacity, noise_model, n_uses, rate | Channel coding theory applies to quantum communication. |
| 30 | **Quantum algorithm design** | COMPOSE_ORACLE, NEST_LOOP, RECURSIVE_AMPLIFY, QUANTUM_WALK_SEARCH, ELEMENT_DISTINCT | oracle_depth, n_iterations, speedup_class, memory_bound | Meta-optimization of algorithm structure itself. |

## Deep Optimization Rounds

The key innovation for WQT50M is that data comes from **actively running WestQuant Studio**
on real optimization problems, not just synthetic generation.

### Round 1: Single-step optimization
- For each problem, generate all legal transformations
- Evaluate each transformation deterministically
- Record the best action, costs, and resource deltas
- This is what WQT20M already does

### Round 2: Multi-step trajectories
- Apply the best action from Round 1
- Generate the next state
- Repeat for 5-20 steps
- Record the full trajectory with branching
- Include dead ends and backtracking
- Compute long-term value V(S) from leaf costs

### Round 3: Multi-objective Pareto exploration
- For each state, explore with 5 different objectives:
  - Minimize depth
  - Minimize 2-qubit gates
  - Minimize error
  - Minimize QPU time
  - Maximize postselection probability
- Record which action is best for each objective
- Generate preference pairs that flip with objective

### Round 4: Hardware-conditioned optimization
- For each trajectory, evaluate on 8 backends:
  - Superconducting (heavy_hex, grid, linear)
  - Trapped ion (all_to_all)
  - Neutral atom (grid, all_to_all)
  - Photonic (linear)
  - Simulator (all_to_all)
- Record routing overhead, error accumulation, feasibility
- Generate hardware counterfactuals (same problem, different backend)

### Round 5: Packing and scheduling
- Pack multiple small algorithms into larger hardware:
  - 2x 10-qubit algorithms → 20-qubit hardware
  - 3x 10-qubit algorithms → 30-qubit hardware
  - 5x 10-qubit algorithms → 50-qubit hardware
- Optimize packing layout to minimize cross-talk
- Schedule gates to maximize parallelism
- Record packing efficiency and schedule quality

### Round 6: Hardware-specific optimization
- **Trapped ion (all-to-all)**: Leverage full connectivity
  - No routing needed — all gates are native
  - Optimize for gate count and fidelity
  - Use long-range entanglement for non-local operations
  
- **Neutral atom (2-by-2 connected)**: Leverage local connectivity
  - Group atoms into 2-qubit blocks
  - Optimize within-block operations
  - Use atom rearrangement for between-block connectivity
  - Minimize atom moves (slow operation)

- **Superconducting (heavy_hex)**: Leverage native topology
  - Use native gate set (CZ, SX, RZ)
  - Optimize for heavy-hex connectivity
  - Minimize swap overhead
  - Use dynamical decoupling on idle qubits

### Round 7: Postselection optimization
- Maximize postselection probability
- Identify which measurements to postselect on
- Optimize circuit structure for high acceptance
- Trade off postselection rate vs output quality
- Record postselection probability as a first-class metric

### Round 8: Preprocessing optimization
- Move as much computation as possible to classical preprocessing
- Identify classically computable subcircuits
- Replace quantum subroutines with classical equivalents
- Record classical/quantum split and total runtime

### Round 9: Cross-domain transfer
- Take a circuit optimized for one domain
- Try to optimize it for a different domain
- Record transfer success/failure
- Generate cross-domain preference pairs

### Round 10: Meta-optimization
- Use WQT20M to guide the optimization process itself
- Record which optimization strategies work best for which problems
- Generate meta-preference pairs (strategy A > strategy B for problem type X)
- This creates the training data for WQT50M to learn when to use which strategy

## Data Generation Plan

### Volume targets

| Metric | WQT20M | WQT50M | Scale |
|--------|--------|--------|-------|
| Problems | 1.2M | 10M+ | 8x |
| Records | 100M | 1B | 10x |
| Domains | 15 | 30 | 2x |
| Optimization rounds | 1 | 10 | 10x |
| Backends | 8 | 12+ | 1.5x |
| Objectives | 5 | 10 | 2x |

### Per-domain record distribution

Each domain gets ~33M records:
- Policy: ~8M (oracle + all-actions with costs)
- Preference: ~12M (with visible costs, all difficulty buckets)
- Value: ~2M (cost-to-go with ranking context)
- Legality: ~8M (valid + hard negatives + counterfactuals)
- Hardware: ~3M (8+ backends, feasibility + objective)

### Deep trajectory data

For Round 2-10, each problem generates:
- 5-20 steps per trajectory
- 3-5 branches per step
- Dead ends and backtracking recorded
- Long-term value computed from leaves
- Multi-objective labels per state
- Hardware-conditioned outcomes per backend

This means each problem generates ~100-500 records from deep rounds,
vs ~80 records from Round 1 alone.

### Storage estimate

- 1B records × ~200 bytes per record = ~200 GB
- Tokenized: 1B × ~400 bytes = ~400 GB
- With deduplication: ~150-250 GB unique

## Architecture Plan

### WQT50M architecture

| Parameter | WQT20M | WQT50M |
|-----------|--------|--------|
| Layers | 8 | 12 |
| Hidden size | 384 | 512 |
| Attention heads | 6 | 8 |
| KV heads | 2 | 4 |
| FFN dim | 1536 | 2048 |
| Vocab | 4561 | ~8000 (new domain tokens) |
| Parameters | 19M | ~50M |
| Context | 512 | 1024 |

### Tokenizer expansion

New tokens needed for 30 domains:
- Domain markers: `<DOMAIN:pulse_optimization>`, etc.
- New action names: ~150 new actions
- New state properties: ~50 new property tokens
- New backend types: photonic, topological, etc.

### Training plan

- 3 epochs over 1B records
- Batch size: 64 (larger model, more memory)
- Learning rate: 1e-4 (lower for larger model)
- Warmup: 1000 steps
- Cosine decay
- Device: MPS (Apple Silicon M5 Max)
- Estimated time: ~72 hours (3 days)

## Evaluation Plan

### 10x validation suite (from WQT20M suite)

- 2000 examples per task (vs 200)
- 1000 search problems (vs 100)
- 500 anti-cheating examples (vs 50)
- 200 metamorphic tests (vs 30)
- Bootstrap CIs with 10000 resamples
- Cross-domain evaluation (30 domains)
- Cross-backend evaluation (12+ backends)
- Deep search budgets up to 5000

### New WQT50M-specific tests

1. **Deep trajectory evaluation**: Test on 20-step trajectories, not just single-step
2. **Multi-objective Pareto**: Evaluate Pareto-front coverage
3. **Packing efficiency**: Test on packing multiple algorithms into hardware
4. **Hardware-specific**: Separate evaluations for trapped ion, neutral atom, superconducting
5. **Postselection**: Evaluate postselection probability prediction
6. **Preprocessing**: Evaluate classical/quantum split recommendations
7. **Cross-domain transfer**: Train on 25 domains, test on 5 held-out domains
8. **Meta-optimization**: Test strategy selection for problem types

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Data generation | 2-3 weeks | 1B records across 30 domains |
| 2. Tokenizer expansion | 2 days | ~8000 token vocabulary |
| 3. Architecture + training | 3-4 days | WQT50M checkpoint |
| 4. Evaluation | 1-2 days | Full validation suite results |
| 5. Analysis | 1 day | Scaling analysis + release decision |
| **Total** | **~4 weeks** | **WQT50M release candidate** |

## WestQuant Studio Integration

The deep optimization rounds (2-10) require WestQuant Studio to actively run
optimization on real problems. This means:

1. **Problem library**: 10M+ problems across 30 domains
2. **Transformation engine**: Deterministic execution of all actions
3. **Verifier**: Independent verification of all transformations
4. **RepGraph**: Persistent storage of all trajectories
5. **Multi-backend simulator**: Evaluate on 12+ hardware configurations
6. **Packing engine**: Pack multiple algorithms into hardware
7. **Pulse simulator**: For pulse-level optimization
8. **Classical preprocessor**: For preprocessing optimization

The Studio runs the optimization rounds and generates training data as a
byproduct. WQT50M then learns from these trajectories.

## Key Insight

The fundamental shift from WQT20M to WQT50M is:

> **WQT20M learns from generated examples.**
> **WQT50M learns from actual optimization experience.**

This means the data is not just "what is the best action for this state" but
"what is the best strategy for this class of problems, on this hardware,
with this objective, given this budget."

The 10 deep optimization rounds create the trajectory data that teaches
WQT50M about long-horizon planning, multi-objective trade-offs, and
hardware-specific strategies that cannot be learned from single-step examples.
