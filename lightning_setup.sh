#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  MUS-Core — Lightning AI Studio Setup Script
#  Установка Rust + сборка CUDA + запуск обучения
#  Запуск:  bash lightning_setup.sh
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$REPO_DIR/data"
CKPT_DIR="${MUS_CKPT_DIR:-$REPO_DIR/checkpoints}"
RUST_DIR="$REPO_DIR/mus-core-rust"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MUS-Core — Lightning AI                                   ║"
echo "║  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "║  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) MB"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── 1. Установка Rust ────────────────────────────────────────────────────
if ! command -v rustc &> /dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  Rust $(rustc --version)"
else
    echo "[1/5] Rust already installed: $(rustc --version)"
fi

export PATH="$HOME/.cargo/bin:$PATH"

# ─── 2. Проверка CUDA ─────────────────────────────────────────────────────
echo "[2/5] Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "  CUDA not found. Install via:"
    echo "    conda install -c nvidia cuda-toolkit"
    echo "  Or Lightning AI should have it pre-installed."
    exit 1
fi
echo "  CUDA: $(nvcc --version | tail -1)"

# ─── 3. GPU архитектура ──────────────────────────────────────────────────
echo "[3/5] Detecting GPU architecture..."
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$GPU_ARCH" ]; then
    echo "  Failed to detect, defaulting to sm_75"
    GPU_ARCH="sm_75"
else
    MAJOR=$(echo "$GPU_ARCH" | cut -d. -f1)
    MINOR=$(echo "$GPU_ARCH" | cut -d. -f2)
    GPU_ARCH="sm_${MAJOR}${MINOR}"
fi
export MUS_CUDA_ARCH="$GPU_ARCH"
echo "  Target: $MUS_CUDA_ARCH"

# ─── 4. Подготовка данных ─────────────────────────────────────────────────
echo "[4/5] Preparing data..."
mkdir -p "$DATA_DIR" "$CKPT_DIR"

if [ ! -f "$DATA_DIR/train_cache.bin" ]; then
    echo "  Generating dataset from OpenAssistant..."
    pip install -q datasets tokenizers 2>/dev/null || true
    cd "$REPO_DIR"
    python prepare_dataset.py --max-samples 25000
    echo "  Data ready: $(ls -lh $DATA_DIR/train_cache.bin | awk '{print $5}')"
else
    echo "  Using existing data: $(ls -lh $DATA_DIR/train_cache.bin | awk '{print $5}')"
fi

# ─── 5. Сборка ────────────────────────────────────────────────────────────
echo "[5/5] Building mus-core-rust..."
cd "$RUST_DIR"
export MUS_CKPT_DIR="$CKPT_DIR"
cargo build --release 2>&1
echo "  Build complete: $(ls -lh target/release/mus-core-rust | awk '{print $5}')"

# ─── Запуск ────────────────────────────────────────────────────────────────
echo ""
echo "═══ Starting training ═══"
echo "  Checkpoints: $CKPT_DIR"
echo "  Data: $DATA_DIR/train_cache.bin"
echo ""

cd "$RUST_DIR"
export MUS_CKPT_DIR="$CKPT_DIR"
exec ./target/release/mus-core-rust "$DATA_DIR/train_cache.bin"
