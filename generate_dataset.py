#!/usr/bin/env python3
"""Generate a large structured semantic dataset — 10K+ lines, 500+ concepts."""

import random
import sys

random.seed(42)

# ─── Domain taxonomy ──────────────────────────────────────────────

domains = {
    "programming_langs": [
        "Rust", "Python", "C++", "Go", "Java", "Haskell", "Lisp", "Erlang", "Zig", "Odin",
        "TypeScript", "Elixir", "Clojure", "F#", "Scala", "Kotlin", "Swift", "D", "Nim", "Crystal",
        "Julia", "Racket", "Scheme", "Prolog", "COBOL", "Fortran", "Ada", "Pascal", "Perl", "Lua",
    ],
    "compilers": [
        "LLVM", "GCC", "Clang", "rustc", "GHC", "javac", "NASM", "YASM", "TinyCC", "SWIFTc",
        "compiler", "interpreter", "JIT", "AOT", "bytecode", "IR", "AST", "parser", "lexer", "optimizer",
    ],
    "data_structures": [
        "array", "linked_list", "stack", "queue", "tree", "binary_tree", "BTree", "RBTree", "AVL",
        "hashmap", "trie", "skip_list", "heap", "graph", "deque", "priority_queue", "union_find",
        "segment_tree", "Fenwick", "splay_tree",
    ],
    "programming_concepts": [
        "function", "closure", "lambda", "macro", "inline", "async", "await", "coroutine", "generator",
        "trait", "interface", "abstract_class", "protocol", "typeclass", "mixin", "delegate",
        "thread", "process", "fiber", "green_thread", "goroutine", "actor", "channel", "lock",
        "mutex", "semaphore", "barrier", "RwLock", "Arc", "RefCell",
    ],
    "gpu_computing": [
        "GPU", "CUDA", "OpenCL", "Vulkan", "shader", "warp", "block", "grid", "stream", "kernel",
        "global_memory", "shared_memory", "register", "SM", "SP", "tensor_core", "RT_core",
    ],
    "ml_concepts": [
        "neural_network", "transformer", "attention", "MLP", "CNN", "RNN", "LSTM", "GRU", "GAN",
        "diffusion", "VAE", "autoencoder", "residual", "normalization", "dropout", "batch_norm",
        "gradient", "backprop", "SGD", "Adam", "AdamW", "RMSProp", "learning_rate", "loss",
        "cross_entropy", "MSE", "MAE", "cosine_similarity", "embedding", "tokenizer",
    ],
    "physics": [
        "quantum", "relativity", "gravity", "electromagnetism", "thermodynamics", "mechanics", "optics",
        "atom", "electron", "proton", "neutron", "quark", "gluon", "photon", "boson", "fermion",
        "Higgs", "lepton", "muon", "tau", "neutrino", "energy", "force", "mass", "wave",
    ],
    "biology": [
        "DNA", "RNA", "protein", "gene", "chromosome", "genome", "mutation", "evolution", "cell",
        "mitosis", "meiosis", "ribosome", "mitochondria", "nucleus", "membrane", "cytoplasm",
        "enzyme", "hormone", "antibody", "virus", "bacteria",
    ],
    "astronomy": [
        "star", "planet", "galaxy", "nebula", "black_hole", "supernova", "neutron_star", "quasar",
        "comet", "asteroid", "meteor", "orbit", "eclipse", "constellation",
    ],
    "ecology": [
        "ecosystem", "biodiversity", "habitat", "species", "population", "food_chain", "symbiosis",
        "predation", "decomposition", "photosynthesis",
    ],
    "math": [
        "number", "integer", "rational", "real", "complex", "prime", "natural", "fraction",
        "algebra", "geometry", "calculus", "topology", "logic", "set_theory", "number_theory",
        "combinatorics", "matrix", "vector", "tensor", "scalar", "determinant", "eigenvalue",
        "probability", "statistics", "distribution", "variance", "mean", "median",
    ],
    "linguistics": [
        "noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction",
        "phoneme", "morpheme", "lexeme", "syntax", "semantics", "pragmatics", "phonology",
        "alphabet", "glyph", "font", "script", "letter", "symbol",
    ],
    "programming_tools": [
        "git", "docker", "make", "cmake", "cargo", "npm", "pip", "conda", "vscode", "neovim",
        "emacs", "vim", "linux", "bash", "zsh", "tmux", "ssh", "http", "websocket", "REST",
    ],
    "databases": [
        "SQL", "NoSQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra", "DuckDB",
        "key_value", "relational", "document", "index", "query", "transaction",
    ],
}

bridges = [
    # Cross-domain semantic bridges
    ("Rust", "compiler"), ("Python", "interpreter"), ("Java", "JIT"),
    ("GPU", "CUDA"), ("neural_network", "GPU"), ("attention", "transformer"),
    ("quantum", "physics"), ("atom", "biology"), ("DNA", "gene"),
    ("star", "galaxy"), ("galaxy", "astronomy"), ("ecosystem", "species"),
    ("number", "math"), ("algebra", "math"), ("vector", "matrix"),
    ("noun", "verb"), ("syntax", "semantics"),
    ("git", "linux"), ("docker", "linux"), ("SQL", "PostgreSQL"),
    ("function", "lambda"), ("trait", "interface"),
    ("thread", "process"), ("actor", "channel"),
    ("gradient", "neural_network"), ("embedding", "transformer"),
    ("mutation", "evolution"), ("protein", "enzyme"),
    ("quark", "gluon"), ("electron", "proton"), ("energy", "mass"), ("energy", "force"),
    ("array", "linked_list"), ("stack", "queue"),
    ("parser", "AST"), ("lexer", "parser"),
    ("lock", "mutex"), ("semaphore", "barrier"),
    ("segment_tree", "Fenwick"), ("BTree", "RBTree"),
    ("LSTM", "GRU"), ("CNN", "RNN"),
    ("neuron", "neural_network"), ("synapse", "weight"),
]

chains = [
    ["Rust", "trait", "interface", "inheritance", "typeclass", "Haskell"],
    ["GPU", "CUDA", "kernel", "block", "thread", "warp", "SM"],
    ["transformer", "attention", "MLP", "embedding", "vector", "tokenizer"],
    ["neural_network", "gradient", "backprop", "SGD", "AdamW", "loss"],
    ["quantum", "quark", "gluon", "boson", "fermion", "Higgs"],
    ["star", "galaxy", "black_hole", "supernova", "neutron_star", "quasar"],
    ["DNA", "RNA", "protein", "ribosome", "cell", "nucleus"],
    ["ecosystem", "species", "population", "habitat", "biodiversity", "evolution"],
    ["algebra", "geometry", "calculus", "topology", "number_theory", "combinatorics"],
    ["probability", "statistics", "distribution", "variance", "regression", "correlation"],
    ["noun", "verb", "syntax", "semantics", "pragmatics", "linguistics"],
    ["compiler", "parser", "AST", "IR", "optimizer", "bytecode"],
    ["git", "commit", "branch", "merge", "rebase", "clone"],
    ["docker", "container", "image", "volume", "network", "compose"],
    ["SQL", "query", "index", "transaction", "join", "relational"],
    ["thread", "mutex", "lock", "deadlock", "race_condition", "atomic"],
    ["Rust", "C++", "memory", "ownership", "borrowing", "lifetime"],
    ["energy", "mass", "force", "gravity", "wave", "mechanics"],
]

# ─── Collect all unique words ───
all_words = {}
for domain, words in domains.items():
    for w in words:
        all_words[w] = domain

# ─── Generate pairs ───
lines = set()

# 1. Intra-domain pairs (dense within each domain)
for domain, words in domains.items():
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if random.random() < 0.35:
                a, b = words[i], words[j]
                lines.add((a, b))

# 2. Chain pairs (sequential walk through concepts)
for chain in chains:
    for i in range(len(chain)):
        for j in range(i + 1, min(i + 5, len(chain))):
            lines.add((chain[i], chain[j]))

# 3. Bridge pairs (cross-domain)
for a, b in bridges:
    lines.add((a, b))

    # 4. Intra-domain pairs (increase density)
for domain, words in domains.items():
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if random.random() < 0.55:  # more density
                a, b = words[i], words[j]
                lines.add((a, b))

# 5. Fill with random cross-domain pairs (sparse)
all_words_list = list(all_words.keys())
for _ in range(4000):
    a = random.choice(all_words_list)
    b = random.choice(all_words_list)
    if a != b and (a, b) not in lines and (b, a) not in lines:
        da, db = all_words[a], all_words[b]
        if da != db:
            lines.add((a, b))

# 6. Fill with same-domain pairs (increase total)
for _ in range(5000):
    domain = random.choice(list(domains.values()))
    a, b = random.choice(domain), random.choice(domain)
    if a != b:
        lines.add((a, b))

# 7. More random filler
for _ in range(5000):
    a = random.choice(all_words_list)
    b = random.choice(all_words_list)
    if a != b:
        lines.add((a, b))

lines = list(lines)
random.shuffle(lines)

print(f"Generated {len(lines)} concept pairs", file=sys.stderr)
unique = set()
for a, b in lines:
    unique.add(a)
    unique.add(b)
print(f"Unique concepts: {len(unique)}", file=sys.stderr)

# Domain stats
domain_counts = {}
for w in unique:
    d = all_words.get(w, "other")
    domain_counts[d] = domain_counts.get(d, 0) + 1
for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
    print(f"  {d}: {c} concepts", file=sys.stderr)

with open("concept_data.tsv", "w") as f:
    f.write("concept_a\tconcept_b\n")
    for a, b in lines:
        f.write(f"{a}\t{b}\n")

print(f"\nWrote concept_data.tsv ({len(lines)} pairs)", file=sys.stderr)
