# МУС — Modular Smart System

**Version:** 2.0.0  
**License:** MIT / Apache 2.0 (Dual)  
**Stack:** C++17 / CUDA → Hugging Face Hub

A production-grade CUDA transformer with **4-config memory auto-detection** (4GB→8GB→1B),
**multimodal vision/audio**, and a **fully automated cloud pipeline** (Kaggle → HF).

## Quick Start

```bash
# Build all tools
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Auto-train (detects your VRAM, picks the right model)
./mus_train_4_6gb
```

## Available Configs

| Config  | Params | VRAM | GPU                    |
|---------|--------|------|------------------------|
| 400M    | 376M   | 3.4GB| GTX 1060, 1660 Ti     |
| 700M    | 685M   | 5.8GB| RTX 3060, 4060, T4    |
| 500M    | 498M   | 4.1GB| RTX 3050, 4050        |
| 1B      | 1.0B   | 7.8GB| RTX 4090, A100, L4    |

## Multi-Module Switching

Load different weights for different tasks:

```bash
bash scripts/download_weights.sh coding     # Code generation
bash scripts/download_weights.sh analytics  # Document parsing
bash scripts/download_weights.sh graphics   # Diagram generation
bash scripts/download_weights.sh sound      # Audio quantization
```

## Cloud Pipeline

```bash
# Run on Kaggle (Tesla T4/P100)
bash scripts/kaggle_pipeline.sh <HF_TOKEN>

# Run on Colab (backup GPU)
bash scripts/colab_pipeline.sh <HF_TOKEN>

# Auto-detect platform
bash scripts/cloud_train.sh <HF_TOKEN>
```

The final `.bin` weights are automatically uploaded to:
**https://huggingface.co/Shuteira/mus-uran-weights**

## Project Structure

```
mus-cuda/
├── include/mus_cuda.h      # Unified header (configs, kernels, API)
├── src/
│   ├── kernels.cu          # FP32 CUDA kernels
│   ├── kernels_f16.cu      # FP16 CUDA kernels + multimodal
│   ├── train_4_6gb.cu      # Auto-detecting trainer (4-6GB)
│   ├── train_f16.cu        # 500M training
│   ├── memory_analysis.cu  # VRAM analysis tool
│   └── data_loader.cu      # Multimodal dataset loader
├── scripts/
│   ├── kaggle_pipeline.sh  # Kaggle auto-trainer
│   ├── colab_pipeline.sh   # Colab auto-trainer
│   ├── cloud_train.sh      # Unified launcher
│   ├── upload_to_hf.py     # Hugging Face uploader
│   └── download_weights.sh # Weight downloader
└── CMakeLists.txt          # Universal build system
```

## Build Options

```bash
cmake .. -DMUS_BUILD_CLOUD=ON     # Build for Kaggle/Colab
cmake .. -DMUS_BUILD_MINIMAL=ON   # Minimal tools only
cmake .. -DMUS_ENABLE_DEBUG=ON    # NaN tracing debug
```
