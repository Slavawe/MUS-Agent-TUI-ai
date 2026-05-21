#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  МУС Kaggle Pipeline — C++ CUDA training on Tesla T4/P100
#  Usage: bash kaggle_pipeline.sh <HF_TOKEN>
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: bash kaggle_pipeline.sh <HUGGINGFACE_WRITE_TOKEN>"
    exit 1
fi

echo "===== МУС Kaggle Pipeline v2.0 ====="

# 1. Detect GPU
echo "[1/6] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# 2. Install deps (Kaggle has CUDA pre-installed)
echo "[2/6] Installing build tools..."
apt-get update -qq && apt-get install -y -qq cmake build-essential bc 2>/dev/null || true

# 3. Clone repo
REPO_DIR="/kaggle/working/mus-cuda"
echo "[3/6] Setting up in ${REPO_DIR}..."
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/Shuteira/mus-uran-weights.git "$REPO_DIR" 2>/dev/null || {
        mkdir -p "$REPO_DIR"
        cp -r /kaggle/input/mus-source/* "$REPO_DIR"/ 2>/dev/null || true
    }
fi
cd "$REPO_DIR"

# 4. Build
echo "[4/6] Building..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMUS_BUILD_CLOUD=ON
make -j$(nproc) mus_train_kaggle

# 5. Train (auto-detect config by VRAM)
echo "[5/6] Training..."
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{print $1/1024}')
echo "  VRAM: ${VRAM_GB} GB"
if (( $(echo "$VRAM_GB > 12" | bc -l) )); then
    CONFIG="700m"
elif (( $(echo "$VRAM_GB > 6" | bc -l) )); then
    CONFIG="700m"
else
    CONFIG="400m"
fi
echo "  Config: ${CONFIG}"
./mus_train_kaggle "${CONFIG}"

# 6. Upload to Hugging Face
echo "[6/6] Uploading..."
BIN_FILE=$(ls -t *.bin 2>/dev/null | head -1)
if [ -n "$BIN_FILE" ]; then
    python3 /kaggle/working/mus-cuda/scripts/upload_to_hf.py \
        --token "$HF_TOKEN" \
        --model "Shuteira/mus-uran-weights" \
        --file "$BIN_FILE" \
        --commit "Auto: $(date +%Y-%m-%d_%H-%M) ${CONFIG}"
    echo "Uploaded: ${BIN_FILE}"
else
    echo "No .bin found to upload"
fi

echo "===== Done ====="
