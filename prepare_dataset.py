#!/usr/bin/env python3
"""
Prepare Russian + English dataset for Uragan 1.0 training.
- Downloads OpenAssistant oasst1 (RU + EN messages)
- Trains BPE tokenizer  
- Converts conversations to binary .bin cache for CUDA training

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --langs ru en --max-samples 50000
"""

import os, sys, json, struct, time, random, argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path(__file__).parent / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
CACHE_DIR = DATA_DIR

# Special token IDs for Uragan 1.0 model
PAD_TOKEN = 2001
UNK_TOKEN = 2002
BOS_TOKEN = 2003
EOS_TOKEN = 2004
SEP_TOKEN = 2005  # <sep> between question and answer

# BPE tokens start at this offset
BPE_OFFSET = 2201

SEQ_LEN = 512

def download_dataset(langs, max_samples):
    print("=== Loading OpenAssistant oasst1 ===")
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    
    print(f"Total messages: {len(ds)}")
    filtered = [ex for ex in ds if ex["lang"] in langs]
    print(f"Filtered ({', '.join(langs)}): {len(filtered)}")
    
    if max_samples and len(filtered) > max_samples:
        filtered = random.sample(filtered, max_samples)
        print(f"Sampled: {len(filtered)}")
    
    return filtered

def reconstruct_conversations(messages):
    msg_by_id = {m["message_id"]: m for m in messages}
    pairs = []
    
    for m in messages:
        if m["role"] == "assistant" and m["parent_id"] in msg_by_id:
            parent = msg_by_id[m["parent_id"]]
            if parent["role"] == "prompter":
                pairs.append((parent["text"], m["text"]))
    
    print(f"Reconstructed {len(pairs)} prompt/response pairs")
    return pairs

def train_bpe_tokenizer(texts, vocab_size=40000):
    print(f"\n=== Training BPE tokenizer (vocab_size={vocab_size}) ===")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"],
        min_frequency=2,
        show_progress=True,
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    path = str(TOKENIZER_DIR / "bpe_tokenizer.json")
    tokenizer.save(path)
    print(f"Tokenizer saved to {path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

def tokenize_pairs(pairs, tokenizer):
    print(f"\n=== Tokenizing {len(pairs)} pairs ===")
    all_tokens = []
    all_labels = []
    skipped = 0
    
    t0 = time.time()
    for i, (q, a) in enumerate(pairs):
        q_ids = tokenizer.encode(q).ids
        a_ids = tokenizer.encode(a).ids
        
        q_ids = [t + BPE_OFFSET for t in q_ids]
        a_ids = [t + BPE_OFFSET for t in a_ids]
        
        seq = [BOS_TOKEN] + q_ids + [SEP_TOKEN] + a_ids + [EOS_TOKEN]
        if len(seq) > SEQ_LEN:
            skipped += 1
            continue
        
        q_end = 1 + len(q_ids) + 1
        labels = [-100] * q_end + seq[q_end:]
        
        seq += [PAD_TOKEN] * (SEQ_LEN - len(seq))
        labels += [-100] * (SEQ_LEN - len(labels))
        
        all_tokens.append(seq)
        all_labels.append(labels)
        
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(pairs)}] {time.time()-t0:.1f}s")
    
    print(f"Tokenized {len(all_tokens)} samples, skipped {skipped} (too long)")
    return all_tokens, all_labels

def save_binary(tokens, labels, prefix="train"):
    print(f"\n=== Saving to {CACHE_DIR} ===")
    
    cache_path = CACHE_DIR / f"{prefix}_cache.bin"
    labels_path = CACHE_DIR / f"{prefix}_labels.bin"
    meta_path = CACHE_DIR / f"{prefix}_cache.meta"
    
    import numpy as np
    arr = np.array(tokens, dtype=np.int32)
    arr.tofile(cache_path)
    print(f"  {cache_path}: {arr.shape} = {arr.nbytes // 1024 // 1024} MB")
    
    lb = np.array(labels, dtype=np.int64)
    lb.tofile(labels_path)
    print(f"  {labels_path}: {lb.shape} = {lb.nbytes // 1024 // 1024} MB")
    
    with open(meta_path, "w") as f:
        f.write(f"num_samples={len(tokens)}\n")
        f.write(f"seq_len={SEQ_LEN}\n")
        f.write(f"vocab_size={BPE_OFFSET + tokenizer.get_vocab_size()}\n")
        f.write(f"bpe_offset={BPE_OFFSET}\n")
        f.write(f"bpe_vocab_size={tokenizer.get_vocab_size()}\n")
        f.write(f"src=OpenAssistant/oasst1\n")
        f.write(f"langs={args.langs}\n")
    print(f"  {meta_path}")
    
    print(f"\n=== Sample ===")
    print(f"  Tokens (first 20): {tokens[0][:20]}")
    print(f"  Labels (first 20): {labels[0][:20]}")

def verify_cache(cache_path):
    print(f"\n=== Verifying {cache_path} ===")
    import numpy as np
    arr = np.fromfile(cache_path, dtype=np.int32)
    ns = len(arr) // SEQ_LEN
    print(f"  Samples: {ns}, tokens: {len(arr)}")
    
    sample = arr[:SEQ_LEN]
    min_t, max_t = sample.min(), sample.max()
    print(f"  Token range: [{min_t}, {max_t}]")
    
    aer = sum(1 for t in sample if t < 2001)
    cpp = sum(1 for t in sample if 2001 <= t <= 2100)
    tag = sum(1 for t in sample if 2101 <= t <= 2200)
    txt = sum(1 for t in sample if t > 2200)
    print(f"  Distribution: AER={aer} CPP={cpp} TAG={tag} TEXT={txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Uragan 1.0")
    parser.add_argument("--langs", nargs="+", default=["ru", "en"], help="Languages to include")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples (0 = all)")
    parser.add_argument("--vocab-size", type=int, default=40000, help="BPE vocab size")
    parser.add_argument("--skip-download", action="store_true", help="Skip download (use cached)")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    parser.add_argument("--verify", action="store_true", help="Only verify existing cache")
    args = parser.parse_args()
    
    if args.verify:
        verify_cache(CACHE_DIR / "train_cache.bin")
        sys.exit(0)
    
    messages = download_dataset(args.langs, args.max_samples)
    
    pairs = reconstruct_conversations(messages)
    
    all_texts = [q for q, a in pairs] + [a for q, a in pairs]
    print(f"Total text samples for tokenizer: {len(all_texts)}")
    
    tokenizer = train_bpe_tokenizer(all_texts, args.vocab_size)
    
    tokens, labels = tokenize_pairs(pairs, tokenizer)
    
    save_binary(tokens, labels)
    
    verify_cache(CACHE_DIR / "train_cache.bin")
    
    print("\n=== Done ===")
    print(f"Dataset ready for CUDA training at {CACHE_DIR}/")
    print(f"  train_cache.bin  — int32 tokens")
    print(f"  train_labels.bin — int64 labels (-100 = ignore)")
    print(f"  train_cache.meta — metadata")
