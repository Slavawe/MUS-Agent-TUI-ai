# Uragan 1.0 — C++ CUDA Transformer

![CI Build](https://github.com/Slavawe/MUS-AI_platform/actions/workflows/build.yml/badge.svg)
[![Hugging Face](https://img.shields.io/badge/🤗_Weights-Shuteira/uragan--1.0--weights-blue)](https://huggingface.co/Shuteira/uragan-1.0-weights)

**Version:** 1.0.0  
**License:** MIT / Apache 2.0 (Dual)  
**Stack:** C++17 / CUDA → Hugging Face Hub

Production-grade CUDA transformer с **автоопределением VRAM** (4GB→8GB→1B),
**мультимодальным C++ vision/audio** и **полным облачным пайплайном** (Kaggle → HF).

## Quick Start

```bash
# Build all tools
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Auto-train (detects your VRAM, picks the right model)
./uragan_train
```

## Available Configs

| Config  | Params | VRAM | GPU                    |
|---------|--------|------|------------------------|
| 400M    | 376M   | 3.4GB| GTX 1060, 1660 Ti     |
| 700M    | 685M   | 5.8GB| RTX 3060, 4060, T4    |
| 500M    | 498M   | 4.1GB| RTX 3050, 4050        |
| 1B      | 1.0B   | 7.8GB| RTX 4090, A100, L4    |

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
**https://huggingface.co/Shuteira/uragan-1.0-weights**
