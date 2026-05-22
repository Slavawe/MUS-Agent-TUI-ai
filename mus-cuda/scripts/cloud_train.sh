#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  МУС Unified Cloud Trainer — auto-detect environment, train, upload
#  Usage: bash cloud_train.sh <HF_TOKEN> [config]
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: bash cloud_train.sh <HUGGINGFACE_WRITE_TOKEN> [400m|700m]"
    exit 1
fi
CONFIG="${2:-auto}"

echo "===== MUS-эфир 1.0 — Cloud Trainer ====="

# Auto-detect platform
if [ -d "/kaggle" ]; then
    echo "Platform: Kaggle"
    exec bash "$(dirname "$0")/kaggle_pipeline.sh" "$HF_TOKEN" "$CONFIG"
elif [ -d "/content/drive" ]; then
    echo "Platform: Google Colab"
    exec bash "$(dirname "$0")/colab_pipeline.sh" "$HF_TOKEN" "$CONFIG"
elif command -v nvidia-smi &> /dev/null; then
    echo "Platform: Local GPU"
    REPO_DIR="$(dirname "$0")/.."
    cd "$REPO_DIR"
    mkdir -p build && cd build
    cmake "$REPO_DIR" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) mus_train
    echo "Build complete."

    # Prepare dataset if not already done
    if [ ! -f "$REPO_DIR/data/train_cache.bin" ]; then
        cd "$REPO_DIR"
        pip install -q datasets tokenizers 2>/dev/null || true
        python3 prepare_dataset.py --max-samples 30000
        cd build
    fi

    # Train
    mkdir -p checkpoints
    echo "Training MUS-эфир 1.0..."
    ./mus_train "$REPO_DIR/data/train_cache.bin"

    # Upload result
    BIN_FILE=$(ls -t checkpoints/*.bin 2>/dev/null | head -1)
    if [ -n "$BIN_FILE" ]; then
        python3 "$(dirname "$0")/upload_to_hf.py" \
            --token "$HF_TOKEN" \
            --model "Shuteira/mus-ether-1.0-weights" \
            --file "$BIN_FILE" \
            --commit "Local: $(date +%Y-%m-%d_%H-%M) FP32-12L"
    else
        echo "No checkpoint .bin found in checkpoints/"
    fi
else
    echo "ERROR: No supported platform detected"
    exit 1
fi
