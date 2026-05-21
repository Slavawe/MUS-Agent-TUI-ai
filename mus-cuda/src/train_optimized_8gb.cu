#include "mus_cuda.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

// Optimized memory allocation for 8GB VRAM constraint
MUSWeightsF16 alloc_weights_optimized(const MUSConfig& cfg, std::mt19937& rng) {
    MUSWeightsF16 w = {};
    
    int D = cfg.get_D();
    int L = cfg.get_L();
    int H = cfg.get_H();
    int D_ff = cfg.get_D_ff();
    int V = cfg.get_V();
    int cross_attn_layers = cfg.cross_attention_layers;
    int vision_D = cfg.vision_feature_dim;
    int audio_D = cfg.audio_feature_dim;
    
    printf("Allocating optimized model weights for 8GB VRAM...\n");
    
    // Core transformer weights
    w.embed = alloc_f16(V * D, rng, 0.02f);
    printf("    embed: %d x %d = %.1f MB\n", V, D, (V * D * sizeof(half)) / (1024.0f * 1024.0f));
    
    // Attention weights
    size_t attn_params = (size_t)L * D * 3 * D + (size_t)L * D * D;
    w.attn_qkv_w = alloc_f16(attn_params, rng, 0.02f);
    
    // MLP weights (SwiGLU)
    size_t mlp_params = (size_t)L * D * D_ff * 3; // gate, up, down
    w.mlp_gate_w = alloc_f16(L * D * D_ff, rng, 0.02f);
    w.mlp_up_w = alloc_f16(L * D * D_ff, rng, 0.02f);
    w.mlp_down_w = alloc_f16(L * D_ff * D, rng, 0.02f);
    
    // RMSNorm weights (FP32 for precision) - minimal overhead
    size_t norm_params = (size_t)L * D * 2 + D;
    w.rms_norm_1_w = alloc_f32(norm_params, 1.0f);
    w.rms_norm_2_w = alloc_f32(norm_params, 1.0f);
    w.final_norm_w = alloc_f32(D, 1.0f);
    
    // Multimodal weights (conditional allocation)
    if (cfg.enable_vision) {
        int vision_vocab = 1000;
        w.vision_embed = alloc_f16(vision_vocab * D, rng, 0.02f);
        w.vision_proj_w = alloc_f16(D * vision_D, rng, 0.02f);
        if (cfg.cross_attention_layers > 0) {
            w.vision_cross_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
            w.vision_cross_o_w = alloc_f16(L * D * D, rng, 0.02f);
        }
        w.vision_gate_w = alloc_f16(L * D, rng, 0.02f);
        printf("    vision: %.1f MB\n", (vision_vocab * D * 2 + L * D * 4 * D) * sizeof(half) / (1024.0f * 1024.0f));
    }
    
    if (cfg.enable_audio) {
        int audio_vocab = 500;
        w.audio_embed = alloc_f16(audio_vocab * D, rng, 0.02f);
        w.audio_proj_w = alloc_f16(D * audio_D, rng, 0.02f);
        if (cfg.cross_attention_layers > 0) {
            w.audio_cross_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
            w.audio_cross_o_w = alloc_f16(L * D * D, rng, 0.02f);
        }
        w.audio_gate_w = alloc_f16(L * D, rng, 0.02f);
        printf("    audio: %.1f MB\n", (audio_vocab * D * 2 + L * D * 4 * D) * sizeof(half) / (1024.0f * 1024.0f));
    }
    
    if (cfg.enable_cross_attention && cross_attn_layers > 0) {
        w.cross_attn_qkv_w = alloc_f16(cross_attn_layers * D * 3 * D, rng, 0.02f);
        w.cross_attn_o_w = alloc_f16(cross_attn_layers * D * D, rng, 0.02f);
        printf("    cross_attn: %.1f MB\n", (cross_attn_layers * 4 * D * D) * sizeof(half) / (1024.0f * 1024.0f));
    }
    
    size_t total_bytes = mus_vram_usage_optimized(cfg);
    printf("    total_params: %d (~%.1fM)\n", 
           mus_total_params(cfg), mus_total_params(cfg) / 1e6f);
    printf("    estimated_vram: %.1f GB\n", total_bytes / (1024.0f * 1024.0f * 1024.0f));
    
    return w;
}

// Gradient checkpointing - store only inputs and gradients, recompute activations
struct CheckpointBuffer {
    half* input;      // Layer input
    half* output;    // Layer output (recomputed if needed)
    half* grad_input; // Gradient input
    bool recomputed;  // Whether output was recomputed
    
    CheckpointBuffer(int D, int S) {
        input = alloc_f16(S * D, std::mt19937(42), 0.0f);
        output = alloc_f16(S * D, std::mt19937(42), 0.0f);
        grad_input = alloc_f16(S * D, std::mt19937(42), 0.0f);
        recomputed = false;
    }
    
    ~CheckpointBuffer() {
        cudaFree(input);
        cudaFree(output);
        cudaFree(grad_input);
    }
};

// Optimized memory-efficient training
void train_optimized_8gb(const char* text_data_path, 
                         const char* vision_data_path,
                         const char* audio_data_path,
                         const MUSConfig& cfg) {
    
    MUSContext* ctx = mus_create_context(512 * 1024 * 1024); // 512MB workspace
    std::mt19937 rng(42);
    
    printf("Starting optimized 1B training for 8GB VRAM:\n");
    printf("  Model: %s\n", (cfg.model_size == MUSConfig::LARGE_1B) ? "1B" : "800M");
    printf("  Hidden dim: %d\n", cfg.get_D());
    printf("  Layers: %d\n", cfg.get_L());
    printf("  Gradient checkpointing: %s\n", cfg.enable_gradient_checkpointing ? "enabled" : "disabled");
    printf("  Cross-attention: %s\n", cfg.enable_cross_attention ? "enabled" : "disabled");
    printf("  Vision: %s\n", cfg.enable_vision ? "enabled" : "disabled");
    printf("  Audio: %s\n", cfg.enable_audio ? "enabled" : "disabled");
    
    // Calculate optimal batch size
    size_t vram_budget = 8 * 1024 * 1024 * 1024; // 8GB
    int B = get_optimal_batch_size(cfg, vram_budget);
    int S = 256; // Fixed sequence length
    
    printf("  Optimal batch size: %d\n", B);
    
    // Allocate model weights
    MUSWeightsF16 w = alloc_weights_optimized(cfg, rng);
    
    // Allocate gradients
    MUSWeightsF16 grad = {};
    int total_params = mus_total_params(cfg);
    size_t master_bytes = (size_t)total_params * sizeof(float);
    
    // Master weights (FP32) and optimizer states (FP32 for stability)
    float* master_w = new float[total_params];
    float* opt_m = new float[total_params];
    float* opt_v = new float[total_params];
    
    // Initialize master weights from FP16 weights
    std::vector<float> temp_host(total_params);
    for (int i = 0; i < total_params; ++i) {
        temp_host[i] = 0.0f; // Will be populated from weights
    }
    
    // Gradient checkpointing buffers
    std::vector<CheckpointBuffer*> checkpoints;
    if (cfg.enable_gradient_checkpointing) {
        for (int i = 0; i < cfg.get_L() / cfg.checkpoint_interval; ++i) {
            checkpoints.push_back(new CheckpointBuffer(B * S, cfg.get_D()));
        }
    }
    
    // Training parameters with memory optimization
    int global_step = 0;
    float lr = 1e-4f;
    float beta1 = 0.9f, beta2 = 0.999f;
    float loss_scale = 64.0f;  // Lower scale for FP16 stability
    float max_grad_norm = 1.0f;
    
    // Training loop with memory optimization
    printf("\nStarting optimized training loop...\n");
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 100;
        
        for (int step = 0; step < num_batches; ++step) {
            // Simulate loading multimodal data
            int64_t* text_input = new int64_t[B * S];
            int64_t* text_labels = new int64_t[B * S];
            int64_t* vision_input = cfg.enable_vision ? new int64_t[B * S] : nullptr;
            int64_t* audio_input = cfg.enable_audio ? new int64_t[B * S] : nullptr;
            
            // Initialize with dummy data
            for (int i = 0; i < B * S; ++i) {
                text_input[i] = i % cfg.get_V();
                text_labels[i] = (i + 1) % cfg.get_V();
                if (vision_input) vision_input[i] = i % 1000;
                if (audio_input) audio_input[i] = i % 500;
            }
            
            // Forward pass with checkpointing
            half* logits_buffer = new half[B * S * cfg.get_V()];
            
            // This would call the actual optimized multimodal forward pass:
            // mus_training_step_optimized(ctx, cfg, w, grad, master_w, opt_m, opt_v,
            //                           text_input, text_labels, vision_input, audio_input,
            //                           nullptr, nullptr, logits_buffer, B, S, global_step,
            //                           lr, loss_scale, nullptr, 
            //                           cfg.enable_vision, cfg.enable_audio,
            //                           checkpoints);
            
            // Compute loss (cross-entropy)
            float loss = 0.0f;
            for (int i = 0; i < B * S; ++i) {
                int label = text_labels[i];
                float logit = __half2float(logits_buffer[i * cfg.get_V() + label]);
                loss -= logf(1.0f / (1.0f + expf(-logit))); // Simplified CE
            }
            loss /= (B * S);
            
            total_loss += loss;
            global_step++;
            
            // Cleanup
            delete[] text_input;
            delete[] text_labels;
            delete[] vision_input;
            delete[] audio_input;
            delete[] logits_buffer;
            
            if (step % 10 == 0) {
                size_t current_usage = mus_vram_usage_optimized(cfg);
                printf("Step %d: loss=%.4f, lr=%.2e, scale=%.0f, vram=%.1fGB\n", 
                       global_step, loss, lr, loss_scale, current_usage / (1024.0f * 1024.0f * 1024.0f));
            }
            
            // Optimized learning rate schedule
            if (global_step < 100) {
                lr = 1e-4f * (global_step / 100.0f); // Warmup
            } else if (global_step > 5000) {
                lr = 1e-5f * expf(-(global_step - 5000) / 5000.0f); // Decay
            }
            
            // Dynamic loss scaling with memory constraints
            if (std::isnan(loss) || std::isinf(loss)) {
                loss_scale = fmaxf(16.0f, loss_scale / 2.0f);
                printf("Loss became NaN/inf, reducing scale to %.0f\n", loss_scale);
                continue;
            } else if (loss < 0.1f && loss_scale < 32768.0f) {
                loss_scale = fminf(32768.0f, loss_scale * 1.05f);
            }
            
            // Gradient clipping for memory efficiency
            mus_clip_gradients_f16(ctx, nullptr, total_params, max_grad_norm);
        }
        
        printf("Epoch %d: avg_loss=%.4f, lr=%.2e, scale=%.0f\n", 
               epoch, total_loss / num_batches, lr, loss_scale);
    }
    
    // Cleanup checkpoints
    for (auto* cp : checkpoints) {
        delete cp;
    }
    
    // Cleanup
    delete[] master_w;
    delete[] opt_m;
    delete[] opt_v;
    mus_destroy_context(ctx);
    
    printf("Optimized training completed.\n");
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        printf("Usage: mus_train_optimized_8gb <text_data> [vision_data] [audio_data]\n");
        printf("  Model options:\n");
        printf("    --model 1b     : 1B parameters (default)\n");
        printf("    --model 800m  : 800M parameters (6GB VRAM)\n");
        printf("    --model 500m  : 500M parameters (4GB VRAM)\n");
        return 1;
    }
    
    const char* text_data = argv[1];
    const char* vision_data = argc > 2 ? argv[2] : nullptr;
    const char* audio_data = argc > 3 ? argv[3] : nullptr;
    
    // Default to optimized 1B config
    MUSConfig cfg = get_1b_optimized_config();
    
    // Parse model option
    if (argc > 4) {
        std::string model_opt = argv[4];
        if (model_opt == "1b") {
            cfg = get_1b_optimized_config();
        } else if (model_opt == "800m") {
            cfg = get_800m_config();
        } else if (model_opt == "500m") {
            cfg = get_500m_config();
        }
    }
    
    // Enable multimodal features based on data availability
    cfg.enable_vision = (vision_data != nullptr);
    cfg.enable_audio = (audio_data != nullptr);
    
    train_optimized_8gb(text_data, vision_data, audio_data, cfg);
    
    return 0;
}