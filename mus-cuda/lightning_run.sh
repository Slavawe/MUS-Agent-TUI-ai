#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  МУС (Modular Smart System) — Lightning AI Studio Launch Script
#  Запуск:  bash lightning_run.sh [400m|700m|test]
#  По умолчанию: авто-выбор под VRAM (400M для 4GB, 700M для 6GB+)
#  На H100 (80GB): берите 700m или test для быстрой проверки
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_DIR/mus-cuda/build"
DATA_DIR="$REPO_DIR/data"
CHECKPOINT_DIR="$REPO_DIR/checkpoints"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  МУС — Lightning AI Studio                                ║"
echo "║  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" 
echo "╚══════════════════════════════════════════════════════════════╝"

# ─── Проверка CUDA ────────────────────────────────────────────────────────
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing..."
    conda install -y -c nvidia cuda-toolkit 2>/dev/null || \
    apt-get update && apt-get install -y cuda-toolkit-12-8 2>/dev/null || {
        echo "Please install CUDA Toolkit manually"
        echo "  conda install -c nvidia cuda-toolkit"
        exit 1
    }
fi

echo "CUDA: $(nvcc --version | tail -1)"

# ─── Зависимости ───────────────────────────────────────────────────────────
pip install -q datasets tokenizers 2>/dev/null || true

# ─── Сборка ────────────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR" "$CHECKPOINT_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake (Release)..."
cmake "$REPO_DIR/mus-cuda" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="90" `# H100 = sm_90` \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Building..."
make -j$(nproc)

echo "Build complete."

# ─── Данные ────────────────────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/train_cache.bin" ]; then
    echo "No training data found. Generating from OpenAssistant..."
    cd "$REPO_DIR"
    python3 -c "
from datasets import load_dataset
ds = load_dataset('OpenAssistant/oasst1', split='train')
import pickle, random, json
random.seed(42)

# Загружаем токенизатор
from tokenizers import Tokenizer
tok = Tokenizer.from_file('data/tokenizer/bpe_tokenizer.json')

samples = []
for item in ds:
    text = item['text']
    if not text: continue
    tokens = tok.encode(text).ids
    for i in range(0, len(tokens), 512):
        chunk = tokens[i:i+512]
        if len(chunk) >= 64:
            chunk = chunk[:512] + [0]*(512-len(chunk))
            samples.extend(chunk)
            if len(samples) >= 11010 * 512: break
    if len(samples) >= 11010 * 512: break

import numpy as np
data = np.array(samples[:11010*512], dtype=np.int32)
data.tofile('data/train_cache.bin')
print(f'Saved {len(data)//512} samples')
"
fi

# ─── Запуск обучения ──────────────────────────────────────────────────────
cd "$BUILD_DIR"
CONFIG="${1:-auto}"

echo ""
echo "═══ Starting training (config=$CONFIG) ═══"

case "$CONFIG" in
    400m)
        echo "400M model (4GB VRAM)"
        echo "Config: D=1024 L=24 H=16 S=256 V=48000"
        ./mus_train_4_6gb 400m
        ;;
    700m|700M)
        echo "700M model (6GB+ VRAM) — recommended for H100"
        echo "Config: D=1280 L=28 H=20 S=256 V=48000"
        ./mus_train_4_6gb 700m
        ;;
    test)
        echo "Test run — FP32 12-layer (quick architecture validation)"
        echo "Config: D=768 L=12 H=12 S=512 B=4 V=42201"
        ./mus_train "$DATA_DIR/train_cache.bin"
        ;;
    h100)
        echo "H100 optimized — auto-config based on VRAM"
        echo "Config: D=1280 L=22 H=20 V=42201 S=512 B=8 (for 80GB)"
        ./mus_h100
        ;;
    auto|*)
        echo "Auto-config based on VRAM"
        if [ -f "$DATA_DIR/train_cache.bin" ]; then
            ./mus_h100
        else
            ./mus_train_4_6gb
        fi
        ;;
esac

echo ""
echo "═══ Training complete ═══"
echo "Checkpoints: $CHECKPOINT_DIR"
ls -lh "$CHECKPOINT_DIR/" 2>/dev/null || echo "(none)"
