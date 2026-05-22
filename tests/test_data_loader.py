"""Test binary data cache format matches what CUDA training expects."""

import struct
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SEQ_LEN = 512


def test_cache_exists():
    assert (DATA_DIR / "train_cache.bin").exists(), "train_cache.bin not found"
    assert (DATA_DIR / "train_labels.bin").exists(), "train_labels.bin not found"
    assert (DATA_DIR / "train_cache.meta").exists(), "train_cache.meta not found"


def test_cache_dimensions():
    tokens = np.fromfile(DATA_DIR / "train_cache.bin", dtype=np.int32)
    labels = np.fromfile(DATA_DIR / "train_labels.bin", dtype=np.int64)

    assert len(tokens) % SEQ_LEN == 0, \
        f"Tokens not divisible by {SEQ_LEN}: {len(tokens)}"
    assert len(labels) % SEQ_LEN == 0, \
        f"Labels not divisible by {SEQ_LEN}: {len(labels)}"
    assert len(tokens) == len(labels), \
        f"Token/label count mismatch: {len(tokens)} vs {len(labels)}"

    n_samples = len(tokens) // SEQ_LEN
    tokens_2d = tokens.reshape(n_samples, SEQ_LEN)
    labels_2d = labels.reshape(n_samples, SEQ_LEN)

    assert tokens_2d.shape == (n_samples, SEQ_LEN)
    assert labels_2d.shape == (n_samples, SEQ_LEN)


def test_token_ranges():
    tokens = np.fromfile(DATA_DIR / "train_cache.bin", dtype=np.int32)
    assert tokens.min() >= 2001, f"Min token < 2001: {tokens.min()}"
    assert tokens.max() <= 48000, f"Max token > 48000: {tokens.max()}"
    assert 2003 in tokens, "BOS token (2003) not found"
    assert 2005 in tokens, "SEP token (2005) not found"
    assert 2004 in tokens, "EOS token (2004) not found"


def test_labels_format():
    """Labels: -100 for question part, token ID for answer + EOS."""
    labels = np.fromfile(DATA_DIR / "train_labels.bin", dtype=np.int64)
    tokens = np.fromfile(DATA_DIR / "train_cache.bin", dtype=np.int32)
    n = len(tokens) // SEQ_LEN

    labels_2d = labels.reshape(n, SEQ_LEN)
    tokens_2d = tokens.reshape(n, SEQ_LEN)

    n_valid = 0

    for i in range(min(n, 100)):
        row_labels = labels_2d[i]
        row_tokens = tokens_2d[i]

        sep_idx = np.where(row_tokens == 2005)[0]
        eos_idx = np.where(row_tokens == 2004)[0]
        if len(sep_idx) == 0 or len(eos_idx) == 0:
            continue

        si, ei = sep_idx[0], eos_idx[0]

        # Up to and including SEP: all labels must be -100
        assert np.all(row_labels[:si + 1] == -100), f"Sample {i}: non -100 before SEP"

        # From after-SEP up to first PAD: labels must be >= 0 (answer + EOS)
        answer_slice = row_labels[si + 1:]
        answer_tokens = row_tokens[si + 1:]
        for j in range(len(answer_slice)):
            if answer_tokens[j] == 2001:
                break
            assert answer_slice[j] >= 0, f"Sample {i}: -100 at answer pos {j}"
        n_valid += 1

    assert n_valid > 0, "No samples with valid SEP found"


def test_cpp_compatible_load():
    """Simulate the C++ PhotoData::load_cache logic."""
    path = DATA_DIR / "train_cache.bin"
    with open(path, "rb") as f:
        data = f.read()

    # C++ loads as int32, same way
    tokens = struct.unpack(f"{len(data)//4}i", data)
    num_samples = len(tokens) // SEQ_LEN

    assert num_samples > 0, "No samples loaded"
    assert len(tokens) == num_samples * SEQ_LEN

    # C++ label loading
    lb_path = DATA_DIR / "train_labels.bin"
    with open(lb_path, "rb") as f:
        lb_data = f.read()
    labels = struct.unpack(f"{len(lb_data)//8}q", lb_data)
    assert len(labels) == len(tokens)

    # C++ batch creation simulation
    B, S = 4, SEQ_LEN
    for batch_start in range(0, min(num_samples, 16), B):
        for b in range(B):
            idx = (batch_start + b) % num_samples
            src = tokens[idx * S:(idx + 1) * S]
            inp = src[:-1]  # input: [0..S-2]
            # labels: normally src[1:] but can use precomputed labels
            lbl = labels[idx * S + 1:(idx + 1) * S]
            assert len(inp) == S - 1
            assert len(lbl) == S - 1


def test_bpe_tokenizer_exists():
    tokenizer_path = DATA_DIR / "tokenizer" / "bpe_tokenizer.json"
    assert tokenizer_path.exists(), "BPE tokenizer not found"

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert tok.get_vocab_size() > 0, "Empty tokenizer vocab"
    assert tok.token_to_id("<pad>") is not None, "Missing <pad>"
    assert tok.token_to_id("<unk>") is not None, "Missing <unk>"
