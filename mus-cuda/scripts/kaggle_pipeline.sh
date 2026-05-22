#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  MUS-эфир 1.0 — Kaggle Training Pipeline
#  Первое обучение: FP32, 12 слоёв, 121M параметров
#  Usage: bash kaggle_pipeline.sh <HF_TOKEN>
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: bash kaggle_pipeline.sh <HUGGINGFACE_WRITE_TOKEN>"
    exit 1
fi

echo "===== MUS-эфир 1.0 — First Training ====="

cd /kaggle/working
REPO_DIR="/kaggle/working/mus-cuda"

# 1. Clone repo
echo "[1/6] Cloning repo..."
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/Slavawe/MUS-Agent-TUI-ai.git mus-cuda
fi
cd "$REPO_DIR"

# 2. Install Python deps for data preprocessing
echo "[2/6] Installing Python dependencies..."
pip install -q datasets tokenizers 2>/dev/null || true

# 3. Prepare dataset (download OpenAssistant, train BPE, create cache)
echo "[3/6] Preparing dataset..."
python3 prepare_dataset.py --max-samples 30000

# 4. Build
echo "[4/6] Building mus_train..."
mkdir -p build && cd build
cmake "$REPO_DIR" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) mus_train
echo "  Build complete."

# 5. Train
echo "[5/6] Training MUS-эфир 1.0 (FP32, 12 layers)..."
mkdir -p checkpoints
./mus_train "$REPO_DIR/data/train_cache.bin" 2>&1

# 6. Upload weights to Hugging Face
echo "[6/6] Uploading weights..."
python3 "$REPO_DIR/scripts/upload_to_hf.py" \
    --token "$HF_TOKEN" \
    --model "Shuteira/mus-ether-1.0-weights" \
    --file "$(ls -t checkpoints/*.bin 2>/dev/null | head -1)" \
    --commit "MUS-эфир 1.0 first training: $(date +%Y-%m-%d_%H-%M)"

echo "===== Done ====="
