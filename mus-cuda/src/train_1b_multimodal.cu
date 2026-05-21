#include "mus_cuda.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

// Utility functions for 1B multimodal training
MUSWeightsF16 alloc_weights_1b(const MUSConfig& cfg, std::mt19937& rng) {
    MUSWeightsF16 w = {};
    
    int D = cfg.get_D();
    int L = cfg.get_L();
    int H = cfg.get_H();
    int D_ff = cfg.get_D_ff();
    int V = cfg.get_V();
    int cross_attn_layers = cfg.cross_attention_layers;
    int vision_D = cfg.vision_feature_dim;
    int audio_D = cfg.audio_feature_dim;
    
    printf("Allocating 1B multimodal model weights (FP16)...\n");
    
    // Core transformer weights
    w.embed = alloc_f16(V * D, rng, 0.02f);
    printf("    embed: %d x %d\n", V, D);
    
    w.attn_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
    w.attn_o_w = alloc_f16(L * D * D, rng, 0.02f);
    w.mlp_gate_w = alloc_f16(L * D * D_ff, rng, 0.02f);
    w.mlp_up_w = alloc_f16(L * D * D_ff, rng, 0.02f);
    w.mlp_down_w = alloc_f16(L * D_ff * D, rng, 0.02f);
    
    // RMSNorm weights (FP32 for precision)
    w.rms_norm_1_w = alloc_f32(L * D, 1.0f);
    w.rms_norm_2_w = alloc_f32(L * D, 1.0f);
    w.final_norm_w = alloc_f32(D, 1.0f);
    
    // Multimodal weights
    if (cfg.enable_vision) {
        int vision_vocab = 1000; // Vision vocabulary size
        w.vision_embed = alloc_f16(vision_vocab * D, rng, 0.02f);
        w.vision_proj_w = alloc_f16(D * vision_D, rng, 0.02f);
        w.vision_cross_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
        w.vision_cross_o_w = alloc_f16(L * D * D, rng, 0.02f);
        w.vision_gate_w = alloc_f16(L * D, rng, 0.02f);
        printf("    vision: embed=%d×%d, proj=%d×%d, cross=%d\n", 
               vision_vocab, D, D, vision_D, L);
    }
    
    if (cfg.enable_audio) {
        int audio_vocab = 500; // Audio vocabulary size
        w.audio_embed = alloc_f16(audio_vocab * D, rng, 0.02f);
        w.audio_proj_w = alloc_f16(D * audio_D, rng, 0.02f);
        w.audio_cross_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
        w.audio_cross_o_w = alloc_f16(L * D * D, rng, 0.02f);
        w.audio_gate_w = alloc_f16(L * D, rng, 0.02f);
        printf("    audio: embed=%d×%d, proj=%d×%d, cross=%d\n", 
               audio_vocab, D, D, audio_D, L);
    }
    
    if (cfg.enable_cross_attention) {
        w.cross_attn_qkv_w = alloc_f16(cross_attn_layers * D * 3 * D, rng, 0.02f);
        w.cross_attn_o_w = alloc_f16(cross_attn_layers * D * D, rng, 0.02f);
        printf("    cross_attn: layers=%d\n", cross_attn_layers);
    }
    
    printf("    total_params: %d (~%.1fM)\n", 
           mus_total_params(cfg), mus_total_params(cfg) / 1e6f);
    
    return w;
}

// Main training function for 1B multimodal model
void train_1b_multimodal(const char* text_data_path, 
                        const char* vision_data_path,
                        const char* audio_data_path,
                        const MUSConfig& cfg) {
    
    MUSContext* ctx = mus_create_context(1024 * 1024 * 1024); // 1GB workspace
    std::mt19937 rng(42);
    
    printf("Starting 1B multimodal training with:\n");
    printf("  Model: %s\n", (cfg.model_size == MUSConfig::LARGE_1B) ? "1B" : "500M");
    printf("  Hidden dim: %d\n", cfg.get_D());
    printf("  Layers: %d\n", cfg.get_L());
    printf("  Heads: %d\n", cfg.get_H());
    printf("  FF dim: %d\n", cfg.get_D_ff());
    printf("  Vision: %s\n", cfg.enable_vision ? "enabled" : "disabled");
    printf("  Audio: %s\n", cfg.enable_audio ? "enabled" : "disabled");
    printf("  Cross-attention: %s\n", cfg.enable_cross_attention ? "enabled" : "disabled");
    
    // Allocate model weights
    MUSWeightsF16 w = alloc_weights_1b(cfg, rng);
    
    // Allocate gradients
    MUSWeightsF16 grad = {};
    int total_params = mus_total_params(cfg);
    size_t master_bytes = (size_t)total_params * sizeof(float);
    
    // Master weights (FP32) and optimizer states (on CPU)
    float* master_w = new float[total_params];
    float* opt_m = new float[total_params];
    float* opt_v = new float[total_params];
    
    // Initialize master weights from FP16 weights
    std::vector<float> temp_host(total_params);
    for (int i = 0; i < total_params; ++i) {
        temp_host[i] = 0.0f; // Will be populated from weights
    }
    
    // Setup training parameters
    int B = 1; // Batch size 1 for memory efficiency
    int S = 256; // Sequence length
    int global_step = 0;
    float lr = 1e-4f;
    float beta1 = 0.9f, beta2 = 0.999f;
    float loss_scale = 128.0f;
    float max_grad_norm = 1.0f;
    
    // Training loop (simplified for demonstration)
    printf("\nStarting training loop...\n");
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 100;
        
        for (int step = 0; step < num_batches; ++step) {
            // Simulate loading multimodal data
            // In practice: load text, vision features, audio features
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
            
            // Forward pass through multimodal model
            half* logits_buffer = new half[B * S * cfg.get_V()];
            
            // This would call the actual multimodal forward pass:
            // mus_training_step_f16(ctx, cfg, w, grad, master_w, opt_m, opt_v,
            //                       text_input, text_labels, vision_input, audio_input,
            //                       nullptr, nullptr, logits_buffer, B, S, global_step,
            //                       lr, loss_scale, nullptr, cfg.enable_vision, cfg.enable_audio);
            
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
                printf("Step %d: loss=%.4f, lr=%.2e, scale=%.0f\n", 
                       global_step, loss, lr, loss_scale);
            }
            
            // Learning rate schedule
            if (global_step < 200) {
                lr = 1e-4f * (global_step / 200.0f); // Warmup
            } else if (global_step > 10000) {
                lr = 1e-5f * expf(-(global_step - 10000) / 10000.0f); // Decay
            }
            
            // Dynamic loss scaling
            if (std::isnan(loss) || std::isinf(loss)) {
                loss_scale = fmaxf(1.0f, loss_scale / 2.0f);
                printf("Loss became NaN/inf, reducing scale to %.0f\n", loss_scale);
                continue;
            } else if (loss < 0.1f && loss_scale < 65536.0f) {
                loss_scale = fminf(65536.0f, loss_scale * 1.1f);
            }
        }
        
        printf("Epoch %d: avg_loss=%.4f, lr=%.2e, scale=%.0f\n", 
               epoch, total_loss / num_batches, lr, loss_scale);
    }
    
    // Cleanup
    delete[] master_w;
    delete[] opt_m;
    delete[] opt_v;
    mus_destroy_context(ctx);
    
    printf("Training completed.\n");
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        printf("Usage: mus_train_1b_multimodal <text_data> [vision_data] [audio_data]\n");
        return 1;
    }
    
    const char* text_data = argv[1];
    const char* vision_data = argc > 2 ? argv[2] : nullptr;
    const char* audio_data = argc > 3 ? argv[3] : nullptr;
    
    // Use 1B configuration by default
    MUSConfig cfg = get_1b_config();
    
    // Override with command line if needed
    if (argc > 4) {
        cfg.model_size = MUSConfig::CUSTOM;
        cfg.hidden_dim = std::stoi(argv[4]);
        cfg.num_layers = std::stoi(argv[5]);
        cfg.num_heads = std::stoi(argv[6]);
    }
    
    // Enable multimodal features based on data availability
    cfg.enable_vision = (vision_data != nullptr);
    cfg.enable_audio = (audio_data != nullptr);
    
    train_1b_multimodal(text_data, vision_data, audio_data, cfg);
    
    return 0;
}