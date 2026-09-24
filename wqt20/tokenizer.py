"""WQT20 tokenizer: ~4096 BPE, quantum-native structured tokens.

Design (Part VI):
- Atomic structured tokens (special, WQIR levels, math ops, Pauli, gates,
  Hamiltonian, fermionic, graph, hardware, objective, resource, action,
  verifier-result, numeric) are added as un-splittable "added tokens".
- A byte-level BPE fallback (over printable ASCII) handles any remaining
  text/numerics so the tokenizer is HF- and GGUF-compatible.

The atomic vocab is ~600 tokens; the BPE merges fill the rest to ~4096.
This is lossless enough that deterministic engines can reconstruct structured
objects from the token stream.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor

# ---------------------------------------------------------------------------
# Atomic structured token vocab (~600 tokens)
# ---------------------------------------------------------------------------

SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>", "<mask>"]

# WQIR levels (12)
WQIR_LEVELS = [
    "<LEVEL:PROBLEM>", "<LEVEL:MATHEMATICAL_FORMULATION>", "<LEVEL:HAMILTONIAN>",
    "<LEVEL:ENCODING>", "<LEVEL:OPERATOR_REPRESENTATION>", "<LEVEL:ALGORITHM>",
    "<LEVEL:ANSATZ>", "<LEVEL:LOGICAL_CIRCUIT>", "<LEVEL:SYNTHESIZED_CIRCUIT>",
    "<LEVEL:ROUTED_CIRCUIT>", "<LEVEL:NATIVE_GATE_CIRCUIT>",
    "<LEVEL:HARDWARE_EXECUTION_STATE>",
]

# Equivalence classes (8)
EQUIVALENCE = [
    "<EQ:EXACT>", "<EQ:GLOBAL_PHASE>", "<EQ:SPECTRAL>", "<EQ:GROUND_STATE>",
    "<EQ:OBJECTIVE>", "<EQ:APPROXIMATE>", "<EQ:NOT_EQUIVALENT>", "<EQ:UNKNOWN>",
]

# Math operators & structure (~70)
MATH_OPS = [
    "+", "-", "*", "/", "^", "=", "!=", "<", ">", "<=", ">=", "%",
    "neg", "inv", "sqrt", "abs", "exp", "log", "ln", "sin", "cos", "tan",
    "det", "tr", "T", "adj", "conj", "re", "im", "d", "sum", "prod", "lim",
    "compose", "kron", "direct_sum", "trace", "rank", "eig", "svd", "expm",
    "(", ")", "[", "]", "{", "}", ",", ";", ":", ".",
    "row", "mat", "vec", "frac", "seq", "cf", "tensor", "outer", "inner",
    "bra", "ket", "dagger", "partial_tr", "schmidt", "entropy",
]
ARITY = ["@1", "@2", "@3", "@θ", "@γ", "@β", "@e", "@t"]

# Numeric encoding (~50)
DIGITS = [f"d{i}" for i in range(10)] + [str(i) for i in range(10)]
NUM_SIGN = ["#+", "#-", "#/", "#.", "#e"]
NUM_EXP = [f"e-{i}" for i in range(1, 9)] + [f"e{i}" for i in range(1, 9)]
NUM_MARKERS = ["<int>", "<flt>", "<rat>", "<cpx>"]

# Variables (~40)
VARS = (
    [f"x{i}" for i in range(16)]
    + ["n", "k", "i", "j", "m", "t", "s", "a", "b", "c", "g", "f", "h", "p", "q", "r"]
    + ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "N"]
    + ["pi", "e_const", "phi", "inf", "nan", "i_const"]
)

# Pauli (~20)
PAULI = ["I", "X", "Y", "Z", "<P:STRING>", "<P:WEIGHT>", "<P:GROUP>", "<P:COMMUTE>", "<P:ANTICOMMUTE>"]
PAULI_LETTERS = ["IX", "IY", "IZ", "XI", "YI", "ZI", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]

# Gates (~50)
GATES_1Q = ["<g:RX>", "<g:RY>", "<g:RZ>", "<g:H>", "<g:S>", "<g:T>", "<g:X>", "<g:Y>", "<g:Z>", "<g:U3>", "<g:SX>", "<g:PHASE>"]
GATES_2Q = ["<g:CNOT>", "<g:CX>", "<g:CZ>", "<g:RXX>", "<g:RYY>", "<g:RZZ>", "<g:SWAP>", "<g:iSWAP>", "<g:CR>", "<g:CPHASE>", "<g:XX>", "<g:YY>", "<g:ZZ>"]
GATES_META = ["<meas>", "<meas:Z>", "<meas:X>", "<meas:Y>", "<shots>", "<barrier>", "<reset>"]
QUBIT_REFS = [f"q{i}" for i in range(32)]

# Hamiltonian (~40)
HAMILTONIANS = [
    "<H:Ising>", "<H:TFIM>", "<H:Heisenberg>", "<H:XY>", "<H:MaxCut>", "<H:MWIS>",
    "<H:QUBO>", "<H:QAOA>", "<H:VQE>", "<H:fermionic>", "<H:lattice>", "<H:Pauli>",
    "<H:portfolio>", "<H:graph>",
]
ENCODINGS = ["<enc:onehot>", "<enc:unary>", "<enc:domain_wall>", "<enc:binary>", "<enc:slack>"]
MAPPINGS = ["<map:JW>", "<map:BK>", "<map:parity>", "<map:bk_superfast>"]
ANSATZ = ["<ansatz:UCCSD>", "<ansatz:HE>", "<ansatz:ADAPT>", "<ansatz:kUpCCGSD>", "<ansatz:QAOA_p>", "<ansatz:hardware_efficient>"]
PENALTY = ["<penalty:U>", "<penalty:slack>"]

# Fermionic (~30)
FERMIONIC = [
    "<f:creation>", "<f:annihilation>", "<f:number>", "<f:excitation>", "<f:UCC>",
    "<f:active_space>", "<f:frozen_core>", "<f:taper>", "<f:second_quant>",
]

# Graph (~80)
GRAPH = [
    "<GRAPH>", "<NODE>", "<EDGE>", "<NODES>", "<EDGES>", "<ADJ>", "<LAPLACIAN>",
    "<spec:compression>", "<spec:eigen>", "<spec:partition>", "<k:8>",
]
GRAPH_FAMILIES = [
    "<fam:ER>", "<fam:BA>", "<fam:WS>", "<fam:SBM>", "<fam:regular>", "<fam:geometric>",
    "<fam:grid>", "<fam:heavy_hex>", "<fam:3regular>", "<fam:cycle>", "<fam:path>",
    "<fam:star>", "<fam:complete>",
]
GRAPH_METRICS = [
    "<g:n>", "<g:m>", "<g:density>", "<g:degree>", "<g:diameter>", "<g:modularity>",
    "<g:clustering>", "<g:spectral_radius>", "<g:treewidth>", "<g:coloring>",
    "<g:cut>", "<g:independent_set>", "<g:clique>",
]
VERTEX = [f"<t:vertex{i}>" for i in range(32)]
SUBGRAPH = ["<sub:clique>", "<sub:path>", "<sub:cycle>", "<sub:star>", "<sub:grid>", "<sub:tree>"]

# Hardware (~40)
HARDWARE = [
    "<BACKEND:SUPERCONDUCTING>", "<BACKEND:TRAPPED_ION>", "<BACKEND:NEUTRAL_ATOM>",
    "<BACKEND:PHOTONIC>", "<BACKEND:SIMULATOR>", "<BACKEND:FT_IDEAL>",
]
TOPOLOGY = ["<topo:all_to_all>", "<topo:heavy_hex>", "<topo:grid>", "<topo:linear>", "<topo:ring>"]
NATIVE = ["<native_1q>", "<native_2q>", "<gate_err_1q>", "<gate_err_2q>", "<readout_err>", "<coherence>"]

# Objective & resource (~40)
OBJECTIVE = ["<OBJECTIVE>", "<W:G2>", "<W:D>", "<W:T>", "<W:E>", "<W:C>", "<W:M>", "<W:A>", "<target:minimize>", "<target:maximize>"]
RESOURCE = ["<RES>", "<r:n_q>", "<r:D>", "<r:G1>", "<r:G2>", "<r:T>", "<r:M>", "<r:A>", "<r:E>", "<r:C>", "<DELTA>", "<DELTA_2Q>", "<DELTA_DEPTH>", "<DELTA_T>", "<DELTA_NQ>", "<DELTA_E>", "<VALUE>", "<PARETO>", "<DOMINATED>", "<NONDOMINATED>", "<FRONTIER>"]

# Actions (~35) — mirrors ecosystem.actions
ACTIONS = [
    "<A:CANONICALIZE>", "<A:CHANGE_BASIS>", "<A:CHANGE_ENCODING>", "<A:MAP_FERMIONS>",
    "<A:TAPER_SYMMETRY>", "<A:GROUP_PAULIS>", "<A:REORDER_TERMS>", "<A:TROTTERIZE>",
    "<A:CHANGE_PRODUCT_FORMULA>", "<A:DECOMPOSE_UNITARY>", "<A:CHANGE_ANSATZ>",
    "<A:REDUCE_ACTIVE_SPACE>", "<A:CIRCUIT_REWRITE>", "<A:ZX_REWRITE>", "<A:COMMUTE>",
    "<A:CANCEL>", "<A:FUSE_ROTATIONS>", "<A:DECOMPOSE_GATE>", "<A:SELECT_NATIVE_GATESET>",
    "<A:SELECT_SYNTHESIS_ENGINE>", "<A:SELECT_ROUTING_ENGINE>", "<A:MAP_QUBITS>",
    "<A:INSERT_SWAP_NETWORK>", "<A:CALL_QISKIT>", "<A:CALL_TKET>", "<A:CALL_PYZX>",
    "<A:EVALUATE>", "<A:VERIFY>", "<A:BRANCH>", "<A:BACKTRACK>", "<A:STOP>",
]

# Verifier results (~15)
VERIFIER = [
    "<V:EXACT_LINEAR>", "<V:UNITARY_EQUIV>", "<V:GROUND_STATE_EQUIV>", "<V:SPECTRAL_EQUIV>",
    "<V:COMMUTATION_GRAPH>", "<V:PRODUCT_FORMULA>", "<V:ZX_CALCULUS>", "<V:SIMULATOR>",
    "<V:EXPRESSIVITY>", "<V:APPROXIMATE>", "<V:PASS>", "<V:FAIL>", "<V:INVALID>",
    "<V:CONNECTIVITY>", "<V:NUMERICAL>",
]

# State structure markers (~15)
STRUCTURE = [
    "<STATE>", "</STATE>", "<LEGAL>", "</LEGAL>", "<NEXT>", "<HISTORY>", "</HISTORY>",
    "<TERMS>", "</TERMS>", "<CANDIDATES>", "</CANDIDATES>", "<BEST>", "</BEST>",
    "<CIRCUIT>", "</CIRCUIT>", "<FEATURES>", "</FEATURES>", "<PROBLEM>", "</PROBLEM>",
    "<BACKEND>", "</BACKEND>", "<LEVEL>", "</LEVEL>", "<METRICS>", "</METRICS>",
]

# Intent (~6)
INTENT = ["<SOLVE>", "<REPRESENT>", "<VERIFY>", "<SCHEDULE>", "<EQUIV>", "<PREDICT>"]

# Clifford/stabilizer (~10)
CLIFFORD = ["<CLIFFORD>", "<STABILIZER>", "<TABLEAU>", "<GRAPH_STATE>", "<SYMPLECTIC>"]

# ZX (~10)
ZX = ["<ZX>", "<SPIDER>", "<PHASE>", "<PIVOT>", "<LOCAL_COMP>", "<PHASE_GADGET>"]

# Algorithm families (~15)
ALGORITHMS = ["<alg:Grover>", "<alg:QFT>", "<alg:QPE>", "<alg:Hamiltonian_sim>", "<alg:VQE>", "<alg:QAOA>", "<alg:walk>", "<alg:amplitude_est>", "<alg:LCU>", "<alg:block_encode>", "<alg:qubitization>", "<alg:state_prep>", "<alg:adder>", "<alg:multiplier>"]


def build_atomic_vocab() -> List[str]:
    """Return the ordered list of atomic (un-splittable) structured tokens."""
    groups = [
        SPECIAL, WQIR_LEVELS, EQUIVALENCE, MATH_OPS, ARITY, DIGITS, NUM_SIGN,
        NUM_EXP, NUM_MARKERS, VARS, PAULI, PAULI_LETTERS, GATES_1Q, GATES_2Q,
        GATES_META, QUBIT_REFS, HAMILTONIANS, ENCODINGS, MAPPINGS, ANSATZ,
        PENALTY, FERMIONIC, GRAPH, GRAPH_FAMILIES, GRAPH_METRICS, VERTEX,
        SUBGRAPH, HARDWARE, TOPOLOGY, NATIVE, OBJECTIVE, RESOURCE, ACTIONS,
        VERIFIER, STRUCTURE, INTENT, CLIFFORD, ZX, ALGORITHMS,
    ]
    seen = set()
    vocab = []
    for g in groups:
        for t in g:
            if t not in seen:
                vocab.append(t)
                seen.add(t)
    return vocab


ATOMIC_VOCAB = build_atomic_vocab()

# Special token ids
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
UNK_ID = 4


def build_tokenizer(vocab_size: int = 4096, save_dir: str | Path | None = None) -> Tokenizer:
    """Build the WQT20 BPE tokenizer with atomic structured tokens.

    The atomic tokens are added as un-splittable added tokens. A byte-level BPE
    over printable ASCII is trained to fill the remaining vocab slots so any
    text/numeric fallback can be encoded.
    """
    atomic = build_atomic_vocab()
    n_atomic = len(atomic)

    # Byte-level BPE for fallback; trainer fills up to vocab_size total.
    bpe = BPE(unk_token="<unk>")
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelPostProcessor(trim_offsets=False)

    # Train BPE on a small synthetic corpus of the structured serialization
    # format so merges are meaningful for our domain.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_:<>/=+*^()[]{}"),
    )

    # Synthetic training corpus: serialize a sample of atomic tokens + numerics
    # so the BPE learns useful merges (e.g. "vertex", "DEPTH", ":").
    corpus = _synthetic_corpus()

    # Train BPE (this assigns ids starting at 0 to special tokens, then merges)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Add the structured atomic tokens as added tokens (un-splittable),
    # appended after the BPE vocab so they get stable high ids.
    added = [(t, i) for i, t in enumerate(atomic, start=tokenizer.get_vocab_size())]
    # Actually use add_tokens API which appends
    for t in atomic:
        if t not in tokenizer.get_vocab():
            tokenizer.add_tokens([t])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_dir / "tokenizer.json"))
        # Save vocab manifest
        meta = {
            "vocab_size": tokenizer.get_vocab_size(),
            "n_atomic": len(atomic),
            "atomic_tokens": atomic,
            "special": {t: tokenizer.token_to_id(t) for t in SPECIAL},
        }
        (save_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2))

    return tokenizer


def _synthetic_corpus() -> List[str]:
    """Varied synthetic corpus of the WQT20 serialization format for BPE merges.

    Generates many randomized structured-state lines so the BPE learns useful
    merges over the ASCII fallback parts (numerics, metric names, fragments).
    """
    import random
    rng = random.Random(42)
    lines: List[str] = []

    problems = ["MAXCUT", "MWIS", "QUBO", "MAX3SAT", "PORTFOLIO", "H2", "LiH", "H4", "BeH2", "TFIM", "Heisenberg"]
    levels = ["HAMILTONIAN", "ENCODING", "OPERATOR_REPRESENTATION", "ANSATZ", "LOGICAL_CIRCUIT", "NATIVE_GATE_CIRCUIT"]
    h_types = ["Ising", "TFIM", "Heisenberg", "MaxCut", "MWIS", "QUBO", "fermionic", "Pauli"]
    backends = ["SUPERCONDUCTING", "TRAPPED_ION", "NEUTRAL_ATOM", "SIMULATOR", "FT_IDEAL"]
    topos = ["all_to_all", "heavy_hex", "grid", "linear", "ring"]
    actions = ["GROUP_PAULIS", "TROTTERIZE", "CHANGE_ENCODING", "REORDER_TERMS", "CHANGE_BASIS",
               "TAPER_SYMMETRY", "CIRCUIT_REWRITE", "ZX_REWRITE", "COMMUTE", "CANCEL",
               "FUSE_ROTATIONS", "DECOMPOSE_GATE", "SELECT_NATIVE_GATESET", "STOP"]
    gates = ["RY", "RZ", "RX", "H", "CNOT", "CZ", "RXX", "RYY", "RZZ", "SWAP", "iSWAP", "S", "T"]
    fams = ["ER", "BA", "WS", "SBM", "regular", "geometric", "grid", "heavy_hex", "3regular"]
    eqs = ["EXACT", "GROUND_STATE", "SPECTRAL", "OBJECTIVE", "APPROXIMATE", "NOT_EQUIVALENT"]
    ansatze = ["UCCSD", "HE", "ADAPT", "kUpCCGSD", "QAOA_p", "hardware_efficient"]
    maps_ = ["JW", "BK", "parity"]

    def rand_float(lo, hi):
        return round(rng.uniform(lo, hi), 4)

    def rand_int(lo, hi):
        return rng.randint(lo, hi)

    for _ in range(4000):
        kind = rng.randint(0, 5)
        if kind == 0:
            lines.append(f"<STATE> <LEVEL:{rng.choice(levels)}> <PROBLEM:{rng.choice(problems)}> "
                        f"<H:{rng.choice(h_types)}> terms={rand_int(2,40)} n_q={rand_int(2,20)} "
                        f"<BACKEND:{rng.choice(backends)}> <topo:{rng.choice(topos)}> "
                        f"gate_err_2q={rand_float(0,0.01)} readout_err={rand_float(0,0.05)}")
        elif kind == 1:
            lines.append(f"<OBJECTIVE> <W:G2>={rand_float(0,1)} <W:D>={rand_float(0,1)} "
                        f"<W:E>={rand_float(0,1)} <target:minimize> "
                        f"<RES> depth={rand_int(1,200)} gates_2q={rand_int(0,500)} t_count={rand_int(0,100)} "
                        f"measurements={rand_int(1,50)} cost={rand_float(0,100)} error={rand_float(0,0.1)}")
        elif kind == 2:
            n = rng.randint(2, 5)
            chosen = rng.sample(actions, n)
            lines.append("<LEGAL> " + " ".join(f"<A:{a}>" for a in chosen) + " </LEGAL> "
                        f"<NEXT> <A:{rng.choice(chosen)}> <DELTA_2Q>={rand_int(-20,5)} "
                        f"<DELTA_DEPTH>={rand_int(-10,5)} <VALUE>={rand_float(0,1)}")
        elif kind == 3:
            terms = []
            for _ in range(rand_int(3, 8)):
                terms.append("".join(rng.choice("IXYZ") for _ in range(rand_int(2, 6))))
            lines.append("<TERMS> " + " ".join(terms) + f" </TERMS> "
                        f"groups={rand_int(1,8)} weight={rand_int(1,6)} commute=yes")
        elif kind == 4:
            qs = " ".join(f"q{rand_int(0,15)}" for _ in range(rand_int(2, 5)))
            lines.append(f"<CIRCUIT> <g:{rng.choice(gates)}> {qs} @theta={rand_float(0,6.28)} "
                        f"<g:{rng.choice(gates)}> q{rand_int(0,15)} q{rand_int(0,15)} "
                        f"@gamma={rand_float(0,3)} <meas:Z> q{rand_int(0,15)} shots={rand_int(100,8000)} </CIRCUIT>")
        else:
            lines.append(f"<fam:{rng.choice(fams)}> nodes={rand_int(3,30)} edges={rand_int(2,60)} "
                        f"density={rand_float(0,1)} modularity={rand_float(0,1)} "
                        f"spectral_radius={rand_float(0,5)} <spec:compression> <k:8> "
                        f"<EQ:{rng.choice(eqs)}> <V:PASS> penalty U={rand_float(1,30)} "
                        f"<ansatz:{rng.choice(ansatze)}> <map:{rng.choice(maps_)}> feasible=yes")

    return lines


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


if __name__ == "__main__":
    tok = build_tokenizer(vocab_size=4096, save_dir=Path(__file__).parent / "tokenizer")
    print(f"vocab size: {tok.get_vocab_size()}")
    enc = tok.encode("<STATE> <LEVEL:HAMILTONIAN> <H:Ising> <A:GROUP_PAULIS> <DELTA_2Q>=-4")
    print("tokens:", enc.tokens)
    print("ids:", enc.ids)
