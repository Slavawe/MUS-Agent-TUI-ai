#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  МУС Colab Pipeline — backup training on Google Colab GPUs
#  Usage:  !bash colab_pipeline.sh <HF_TOKEN>
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: bash colab_pipeline.sh <HUGGINGFACE_WRITE_TOKEN>"
    exit 1
fi

echo "===== МУС Colab Pipeline v2.0 ====="

# 1. Mount Google Drive
echo "[1/7] Mounting Google Drive..."
mkdir -p /content/drive
if mountpoint -q /content/drive; then
    echo "  Drive already mounted"
else
    # Colab handles Google Drive mounting via its runtime UI
    echo "  Assuming drive is mounted at /content/drive/MyDrive"
fi

DRIVE_DIR="/content/drive/MyDrive/mus-cuda"

# 2. Install dependencies
echo "[2/7] Installing dependencies..."
apt-get update -qq && apt-get install -y -qq cmake build-essential bc 2>/dev/null || true
pip install huggingface_hub -q 2>/dev/null || true

# 3. Clone or copy repo
echo "[3/7] Setting up МУС..."
REPO_DIR="/content/mus-cuda"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
elif [ -d "$DRIVE_DIR" ]; then
    echo "  Using cached from Google Drive"
    cp -r "$DRIVE_DIR" "$REPO_DIR"
else
    git clone https://github.com/Shuteira/mus-uran-weights.git "$REPO_DIR" 2>/dev/null || {
        echo "  Git clone failed, creating empty..."
        mkdir -p "$REPO_DIR"
    }
fi
cd "$REPO_DIR"

# 4. Copy cached .bin from Drive (resume training)
echo "[4/7] Checking for cached weights..."
mkdir -p build && cd build
cp "$DRIVE_DIR"/*.bin . 2>/dev/null || true

# 5. Build
echo "[5/7] Building..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DMUS_BUILD_CLOUD=ON
make -j$(nproc) mus_train_kaggle

# 6. Train
echo "[6/7] Training..."
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

# 7. Upload to Hugging Face + cache to Drive
echo "[7/7] Uploading..."
BIN_FILE=$(ls -t *.bin 2>/dev/null | head -1)
if [ -n "$BIN_FILE" ]; then
    python3 /content/mus-cuda/scripts/upload_to_hf.py \
        --token "$HF_TOKEN" \
        --model "Shuteira/mus-uran-weights" \
        --file "$BIN_FILE" \
        --commit "Colab: $(date +%Y-%m-%d_%H-%M) ${CONFIG}"

    # Cache to Drive
    cp "$BIN_FILE" "$DRIVE_DIR/" 2>/dev/null || true
    echo "Uploaded + cached: ${BIN_FILE}"
else
    echo "No .bin found to upload"
fi

echo "===== Done ====="
