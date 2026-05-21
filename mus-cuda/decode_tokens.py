#!/usr/bin/env python3
"""
Token ID Decoder / Demangler for Uragan 1.0 vocabulary
 Converts raw token IDs to human-readable format

Usage:
    python decode_tokens.py <train_cache.bin>
    python decode_tokens.py <train_cache.bin> <train_labels.bin>
"""

import struct
import sys
from pathlib import Path

CPP_START = 2001
CPP_END = 2100
MULTIMODAL_START = 2101
MULTIMODAL_END = 2200

CPP_CHARS = {
    2001: '<PAD>',   2002: '<UNK>',   2003: '<BOS>',   2004: '<EOS>',
    2005: '<NL>',    2006: '<SPACE>', 2007: '<TAB>',
}
for i, code in enumerate(range(32, 127)):
    CPP_CHARS[2008 + i] = chr(code)

def get_token_type(token_id: int) -> str:
    if token_id < CPP_START:
        return "AER"
    elif token_id <= CPP_END:
        return "CPP"
    elif token_id >= MULTIMODAL_START and token_id <= MULTIMODAL_END:
        return "TAG"
    else:
        return "TEXT"

def token_to_str(token_id: int) -> str:
    if token_id in CPP_CHARS:
        return CPP_CHARS[token_id]
    elif token_id < CPP_START:
        return f"<AER:{token_id}>"
    elif token_id <= CPP_END:
        return f"<CPP:{token_id}>"
    elif token_id >= MULTIMODAL_START and token_id <= MULTIMODAL_END:
        return f"<TAG:{token_id}>"
    else:
        return f"<TXT:{token_id}>"

def load_tokens(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    num_tokens = len(data) // 4
    tokens = struct.unpack(f'{num_tokens}i', data)
    return list(tokens)

def load_labels(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    num_labels = len(data) // 8
    labels = struct.unpack(f'{num_labels}q', data)
    return list(labels)

def analyze_tokens(tokens: list):
    stats = {'total': len(tokens), 'aer': 0, 'cpp': 0, 'tag': 0, 'text': 0, 'unique': set()}
    for t in tokens:
        tt = get_token_type(t).lower()
        stats[tt] += 1
        stats['unique'].add(t)
    stats['unique'] = len(stats['unique'])
    return stats

def print_sample(tokens: list, labels: list = None, start: int = 0, count: int = 50):
    print(f"\n=== Token Sample (positions {start} to {start+count-1}) ===")
    print(f"{'Pos':>4} {'Token':>8} {'Type':>6} {'Label':>8} | Text")
    print("-" * 60)
    for i in range(start, min(start + count, len(tokens))):
        t = tokens[i]
        label = labels[i] if labels and i < len(labels) else -100
        tt = get_token_type(t)
        text = token_to_str(t)
        l_str = f"{label}" if label != -100 else "-"
        print(f"{i:4d} {t:8d} {tt:6s} {l_str:>8} | {text}")

def print_sequence_detail(tokens: list, sample_idx: int = 0):
    seq_len = 512
    start = sample_idx * seq_len
    seq = tokens[start:start+seq_len]
    
    print(f"\n=== Sequence {sample_idx} Detail ===")
    print(f"Length: {len(seq)}")
    print(f"First 10: {[token_to_str(t) for t in seq[:10]]}")
    
    # Count non-padding
    non_pad = sum(1 for t in seq if t != 0)
    print(f"Non-padding tokens: {non_pad}")
    
    # Group by type
    type_counts = {}
    for t in seq:
        tt = get_token_type(t)
        type_counts[tt] = type_counts.get(tt, 0) + 1
    print(f"Type breakdown: {type_counts}")

def main():
    if len(sys.argv) < 2:
        print("Usage: decode_tokens.py <train_cache.bin> [labels file]")
        sys.exit(1)
    
    cache_path = sys.argv[1]
    labels_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Loading: {cache_path}")
    tokens = load_tokens(cache_path)
    print(f"Loaded {len(tokens)} tokens ({len(tokens)//512} sequences)")
    
    stats = analyze_tokens(tokens)
    print(f"\n=== Token Distribution ===")
    print(f"  Total:    {stats['total']:,}")
    print(f"  Unique:   {stats['unique']}")
    print(f"  AER:      {stats['aer']:,} ({100*stats['aer']/stats['total']:.1f}%)")
    print(f"  CPP:      {stats['cpp']:,} ({100*stats['cpp']/stats['total']:.1f}%)")
    print(f"  TAG:      {stats['tag']:,} ({100*stats['tag']/stats['total']:.1f}%)")
    print(f"  TEXT:     {stats['text']:,} ({100*stats['text']/stats['total']:.1f}%)")
    
    labels = None
    if labels_path:
        print(f"\nLoading labels: {labels_path}")
        labels = load_labels(labels_path)
        print(f"Loaded {len(labels)} labels")
    
    print_sample(tokens, labels, 0, 30)
    print_sample(tokens, labels, 256, 30)
    print_sample(tokens, labels, 512, 30)
    
    # Show first few sequences
    for i in range(3):
        print_sequence_detail(tokens, i)

if __name__ == '__main__':
    main()