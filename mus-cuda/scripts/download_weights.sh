#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  МУС Weight Downloader — fetch trained .bin from Hugging Face
#  Usage: bash download_weights.sh [MODULE] [FILE]
#  Modules: coding, analytics, graphics, sound, all
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

HF_REPO="Shuteira/mus-ether-1.0-weights"
WEIGHTS_DIR="$(dirname "$0")/../weights"
mkdir -p "$WEIGHTS_DIR"

download() {
    local remote_path="$1"
    local local_name="$2"
    local url="https://huggingface.co/${HF_REPO}/resolve/main/${remote_path}"

    echo "Downloading ${local_name}..."
    if curl -f -L -o "${WEIGHTS_DIR}/${local_name}" "$url" 2>/dev/null; then
        echo "  → ${WEIGHTS_DIR}/${local_name} ($(du -h "${WEIGHTS_DIR}/${local_name}" | cut -f1))"
        return 0
    else
        echo "  ✗ Failed: ${url}"
        return 1
    fi
}

MODULE="${1:-all}"
case "$MODULE" in
    coding|code)
        echo "=== Module: CODING ==="
        download "weights/mus_coding_700m.bin" "coding_700m.bin"
        ;;
    analytics|analysis)
        echo "=== Module: ANALYTICS ==="
        download "weights/mus_analytics_700m.bin" "analytics_700m.bin"
        ;;
    graphics|vision)
        echo "=== Module: GRAPHICS ==="
        download "weights/mus_graphics_700m.bin" "graphics_700m.bin"
        ;;
    sound|audio)
        echo "=== Module: SOUND ==="
        download "weights/mus_sound_700m.bin" "sound_700m.bin"
        ;;
    all|*)
        echo "=== All Modules ==="
        download "weights/mus_coding_700m.bin" "coding_700m.bin"
        download "weights/mus_analytics_700m.bin" "analytics_700m.bin"
        download "weights/mus_graphics_700m.bin" "graphics_700m.bin"
        download "weights/mus_sound_700m.bin" "sound_700m.bin"
        ;;
esac

echo "Done. Weights in: ${WEIGHTS_DIR}"
