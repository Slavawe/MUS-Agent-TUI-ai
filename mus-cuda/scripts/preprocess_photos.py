#!/usr/bin/env python3
"""Препроцессинг фото → ASCII-токены → бинарный кэш для C++/CUDA обучения."""

import sys, os, struct, time, random
import numpy as np
from PIL import Image
from pathlib import Path

# Project root = two levels up from scripts/
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT)

from mus.config import MUSConfig
from mus.vision.ascii_vision import ASCIIVisionEncoder

PHOTO_DIR = os.path.join(PROJECT, "data", "photos")
CACHE_FILE = os.path.join(PHOTO_DIR, "train_cache.bin")
META_FILE = os.path.join(PHOTO_DIR, "train_cache.meta")

def main():
    config = MUSConfig()
    encoder = ASCIIVisionEncoder(config)

    seq_len = 512
    vision_start = config.aer_vision_start_id
    vision_end = config.aer_vision_end_id
    pad_id = 0

    photo_dir = Path(PHOTO_DIR)
    files = sorted(photo_dir.glob("*.jpg")) + sorted(photo_dir.glob("*.png"))
    print(f"Found {len(files)} images in {PHOTO_DIR}")

    samples = []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            img = Image.open(f).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            tokens = encoder.encode(arr)  # [vision_start, ascii_tokens..., vision_end]
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
            continue

        # Crop/pad to seq_len
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        tokens = tokens + [pad_id] * (seq_len - len(tokens))
        samples.append(tokens)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")

    print(f"Encoded {len(samples)} samples in {time.time()-t0:.1f}s")

    # Save binary: [num_samples, seq_len] int32
    arr = np.array(samples, dtype=np.int32)
    arr.tofile(CACHE_FILE)
    print(f"Saved {CACHE_FILE}: {arr.shape} = {arr.nbytes // 1024 // 1024} MB")

    # Save metadata
    with open(META_FILE, "w") as f:
        f.write(f"num_samples={len(samples)}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"hidden_dim={config.hidden_dim}\n")
        f.write(f"num_layers={config.num_layers}\n")
        f.write(f"num_heads={config.num_heads}\n")
        f.write(f"head_dim={config.head_dim}\n")
    print(f"Saved {META_FILE}")

if __name__ == "__main__":
    main()
