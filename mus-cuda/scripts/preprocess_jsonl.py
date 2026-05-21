#!/usr/bin/env python3
"""Preprocess dataset.jsonl → CUDA binary cache for text QA training."""

import sys, os, json, time, random
import numpy as np
from pathlib import Path

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT)

from mus.config import MUSConfig
from mus.tokenizer.bpe_tokenizer import BPETokenizer

BPE_PATH = os.path.join(PROJECT, "scripts", "models", "mus_bpe")

SEQ_LEN = 512
QA_ONLY = True  # only relevance=1 samples
MAX_SAMPLES = 50000  # limit for CUDA training
SHUFFLE = True

def main():
    config = MUSConfig()

    # Load tokenizer
    tok = BPETokenizer.load(BPE_PATH)
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    sep = tok._str_to_id.get("<sep>", 11)
    print(f"BOS={bos} EOS={eos} SEP={sep}")

    # Read JSONL
    jsonl_path = os.path.join(PROJECT, "data", "dataset.jsonl")
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            if not q or not a:
                continue
            if QA_ONLY and obj.get("relevance", 1) != 1:
                continue
            samples.append((q, a))
            if len(samples) >= MAX_SAMPLES:
                break

    print(f"Loaded {len(samples)} QA pairs from {jsonl_path}")
    if SHUFFLE:
        random.shuffle(samples)

    # Tokenize
    all_tokens = []
    all_labels = []
    num_skipped = 0

    t0 = time.time()
    for i, (q, a) in enumerate(samples):
        q_ids = tok.encode(q, add_special_tokens=False)
        a_ids = tok.encode(a, add_special_tokens=False)

        # format: [bos, q..., sep, a..., eos] (unpadded)
        seq = [bos] + q_ids + [sep] + a_ids + [eos]
        if len(seq) > SEQ_LEN:
            num_skipped += 1
            continue

        # labels: -100 for question part, token ID for answer + eos part
        q_end = 1 + len(q_ids) + 1  # bos + q_ids + sep
        labels = [-100] * q_end + seq[q_end:]  # answer + eos (no padding yet)
        # pad both to SEQ_LEN (padding positions → -100 in labels)
        seq += [0] * (SEQ_LEN - len(seq))
        labels += [-100] * (SEQ_LEN - len(labels))

        all_tokens.append(seq)
        all_labels.append(labels)

        if (i + 1) % 10000 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.1f}s")

    print(f"Tokenized {len(all_tokens)} samples, skipped {num_skipped} (too long)")
    print(f"  Sample question: {samples[0][0][:50]}...")
    print(f"  Tokens: {all_tokens[0][:15]}...")
    print(f"  Labels: {all_labels[0][:15]}...")

    # Save binary
    out_dir = os.path.join(PROJECT, "data")
    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(os.path.join(out_dir, "train_cache.bin"))
    print(f"Saved {out_dir}/train_cache.bin: {arr.shape} = {arr.nbytes//1024//1024} MB")

    lb = np.array(all_labels, dtype=np.int64)
    lb.tofile(os.path.join(out_dir, "train_labels.bin"))
    print(f"Saved {out_dir}/train_labels.bin: {lb.shape} = {lb.nbytes//1024//1024} MB")

    meta_path = os.path.join(out_dir, "train_cache.meta")
    with open(meta_path, "w") as f:
        f.write(f"num_samples={len(all_tokens)}\n")
        f.write(f"seq_len={SEQ_LEN}\n")
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"hidden_dim={config.hidden_dim}\n")
        f.write(f"num_layers={config.num_layers}\n")
        f.write(f"num_heads={config.num_heads}\n")
        f.write(f"src=dataset.jsonl\n")
        f.write(f"qa_only={QA_ONLY}\n")
    print(f"Saved {meta_path}")

if __name__ == "__main__":
    main()
