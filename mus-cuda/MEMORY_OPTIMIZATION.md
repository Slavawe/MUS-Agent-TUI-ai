# MUS Memory Optimization Guide

## Overview
This document describes memory optimization strategies for MUS models to fit within 8GB VRAM constraints.

## Memory Analysis

### 1B Model - Full Configuration (9.2GB)
```
Weights:            2.0GB (FP16)
Gradients:          2.0GB (FP16)  
Adam m/v:           4.0GB (FP32)  ← Major memory consumer!
Workspace:          1.0GB
Activations:        0.2GB
Multimodal:         0.0GB
Total:              9.2GB
```

### 1B Model - Optimized (7.8GB)
```
Weights:            2.0GB (FP16)
Gradients:          2.0GB (FP16)
Adam m/v:           2.0GB (FP16)  ← FP16 saves 2GB!
Workspace:          0.5GB (reduced)
Activations:        0.2GB
Multimodal:         0.1GB (minimal)
Total:              7.8GB
```

## Optimization Strategies

### 1. Precision Optimization
- **Adam states**: FP32 → FP16 (saves 2GB)
- **Weights**: FP16 (already optimal)
- **Gradients**: FP16 (already optimal)
- **RMSNorm**: FP32 (keep for numerical stability)

### 2. Feature Reduction
- **Vision features**: 1024 → 512 (-0.5GB)
- **Audio features**: 768 → 384 (-0.3GB)

### 3. Feature Selection
- **Audio**: Disable (-1.0GB)
- **Cross-attention**: Disable (-0.5GB)
- **Vision**: Keep (0.1GB)

### 4. Memory Management
- **Workspace**: 512MB → 256MB (-0.5GB)
- **Batch size**: B=1 (minimal)
- **Sequence length**: S=256 (reasonable)

## Available Configurations

### 1B Optimized (7.8GB)
```cpp
MUSConfig cfg = get_1b_optimized_config();
// D=1536, L=32, H=24, Vision: enabled, Audio: disabled
```

### 800M Optimized (6.2GB)
```cpp
MUSConfig cfg = get_800m_config();
// D=1280, L=28, H=20, Vision: enabled, Audio: disabled
```

### 500M Optimized (4.1GB)
```cpp
MUSConfig cfg = get_500m_config();
// D=1280, L=22, H=20, Vision: disabled, Audio: disabled
```

## Usage Examples

### Memory Analysis
```bash
# Analyze memory usage
./memory_analysis 1b_opt
./memory_analysis 800m  
./memory_analysis 500m

# Compare all configs
./memory_analysis
```

### Training with Memory Optimization
```bash
# 1B optimized training (7.8GB)
./mus_train_optimized_8gb text_data.bin vision_data.bin --model 1b_opt

# 800M training (6.2GB)  
./mus_train_optimized_8gb text_data.bin --model 800m

# 500M training (4.1GB)
./mus_train_optimized_8gb text_data.bin --model 500m
```

## Dynamic Memory Management

### On-Demand Allocation
```cpp
half* workspace;
size_t workspace_size;
mus_allocate_on_demand_f16(ctx, cfg, B, S, &workspace, &workspace_size);
```

### Memory Optimization
```cpp
size_t current_vram, target_vram = 8 * 1024 * 1024 * 1024; // 8GB
mus_optimize_memory_usage_f16(ctx, cfg, &current_vram, target_vram);
```

## Performance Impact

### Memory Optimizations
- **FP16 Adam**: 2GB savings, minimal accuracy impact
- **Feature reduction**: 0.8GB savings, minor accuracy impact
- **Feature selection**: 1.5GB savings, significant accuracy impact

### Training Optimizations  
- **Gradient checkpointing**: 0.3GB savings, recomputation overhead
- **Reduced workspace**: 0.5GB savings, minimal impact
- **B=1**: Minimal memory impact, reasonable throughput

## Recommended Setup

### For 8GB VRAM (RTX 3060, 4060, 3070)
```
Model: 1B Optimized
Memory: 7.8GB
Features: Vision only
Precision: FP16 weights/gradients, FP32 Adam m/v
```

### For 6GB VRAM (RTX 3060, 4050)
```
Model: 800M Optimized  
Memory: 6.2GB
Features: Vision only
Precision: FP16 weights/gradients, FP32 Adam m/v
```

### For 4GB VRAM (GTX 1660 Ti, RTX 3050)
```
Model: 500M Optimized
Memory: 4.1GB  
Features: Text only
Precision: FP16 weights/gradients, FP32 Adam m/v
```

## Memory Monitoring

```bash
# Monitor VRAM during training
nvidia-smi -l 1

# Check CUDA memory usage
python3 -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB')"
```

## Future Optimizations

1. **Mixed Precision Adam**: FP32 compute with FP16 storage
2. **Gradient Accumulation**: B=4 with accumulation every 4 steps
3. **CPU Offloading**: Move less frequently used weights to CPU
4. **Quantization**: INT8 weights for inference
5. **Pruning**: Sparse attention and MLP layers

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size (B=1)
2. Disable multimodal features
3. Use smaller model configuration
4. Enable gradient checkpointing

### Performance Issues
1. Monitor VRAM usage during training
2. Check for memory fragmentation
3. Verify feature dimensions
4. Test different workspace sizes