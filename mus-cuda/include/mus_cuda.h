#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// ─── Config ────────────────────────────────────────────────────────────
struct MUSConfig {
    int vocab_size = 48000;
    int hidden_dim = 1280;      // D (embedding dimension)
    int num_layers = 22;        // L (number of transformer blocks)
    int num_heads = 20;         // H (number of attention heads)
    int head_dim = 64;          // D/H (head dimension)
    int max_seq_len = 8192;
    int ffn_dim = 0;  // if 0, get_D_ff returns hidden_dim * 4

    // Model size presets
    enum ModelSize { SMALL_500M, LARGE_1B, CUSTOM };
    ModelSize model_size = LARGE_1B;

    // Memory optimization flags
    bool enable_gradient_checkpointing = false;
    int checkpoint_interval = 4;  // Checkpoint every N layers
    bool enable_offloading = false;  // CPU offloading for memory

    int get_D() const { return hidden_dim; }
    int get_L() const { return num_layers; }
    int get_H() const { return num_heads; }
    int get_D_ff() const { return ffn_dim > 0 ? ffn_dim : hidden_dim * 4; }
    int get_V() const { return vocab_size; }

    // ASCII токены (символы палитры)
    int ascii_tokens_start = 2001;
    int ascii_tokens_end = 2077;

    // Режимные префиксы для мультимодального выравнивания
    int vision_photo = 2104;   // <vision_photo> — фото pipeline
    int vision_graph = 2105;   // <vision_graph> — граф pipeline

    // Мультимодальные теги
    int multimodal_tags_start = 2101;
    int multimodal_tags_end = 2300;

    float loss_weight_aer = 15.0f;
    float loss_weight_ascii = 5.0f;
    float loss_weight_text = 1.0f;
    
    // Multimodal support flags
    bool enable_vision = true;
    bool enable_audio = false;
    bool enable_cross_attention = true;
    
    // Vision model config
    int vision_feature_dim = 1024;      // Vision feature dimension
    int vision_patch_size = 16;        // Vision patch size
    int vision_grid_size = 64;         // Vision grid size (64x64 for 1024 patches)
    
    // Audio model config  
    int audio_feature_dim = 768;        // Audio feature dimension
    int audio_frame_rate = 16000;       // Audio sample rate
    int audio_context_window = 1024;    // Audio context frames
    
    // Cross-attention config
    int cross_attention_layers = 4;     // Number of cross-attention layers
    int cross_attention_heads = 8;      // Cross-attention heads
};

// Predefined configurations
inline MUSConfig get_500m_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::SMALL_500M;
    cfg.hidden_dim = 1280;
    cfg.num_layers = 22;
    cfg.num_heads = 20;
    cfg.head_dim = 64;
    cfg.enable_vision = true;
    cfg.enable_audio = false;
    cfg.enable_cross_attention = true;
    cfg.cross_attention_layers = 4;
    return cfg;
}

inline MUSConfig get_1b_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::LARGE_1B;
    cfg.hidden_dim = 1536;
    cfg.num_layers = 32;
    cfg.num_heads = 24;
    cfg.head_dim = 64;
    cfg.enable_vision = true;
    cfg.enable_audio = true;
    cfg.enable_cross_attention = true;
    cfg.cross_attention_layers = 8;
    cfg.vision_feature_dim = 1024;
    cfg.audio_feature_dim = 768;
    return cfg;
}

// 400M config for 4GB VRAM
inline MUSConfig get_400m_4gb_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::CUSTOM;
    cfg.hidden_dim = 1024;
    cfg.num_layers = 24;
    cfg.num_heads = 16;
    cfg.head_dim = 64;
    cfg.ffn_dim = 3072;     // 3x hidden_dim (saves ~25% vs 4x)
    cfg.max_seq_len = 256;
    cfg.enable_gradient_checkpointing = true;
    cfg.checkpoint_interval = 4;
    cfg.enable_vision = false;
    cfg.enable_audio = false;
    cfg.enable_cross_attention = false;
    cfg.vision_feature_dim = 0;
    cfg.audio_feature_dim = 0;
    return cfg;
}

// 700M config for 6GB VRAM
inline MUSConfig get_700m_6gb_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::CUSTOM;
    cfg.hidden_dim = 1280;
    cfg.num_layers = 28;
    cfg.num_heads = 20;
    cfg.head_dim = 64;
    cfg.ffn_dim = 4096;     // ~3.2x hidden_dim
    cfg.max_seq_len = 256;
    cfg.enable_gradient_checkpointing = true;
    cfg.checkpoint_interval = 4;
    cfg.enable_vision = true;
    cfg.enable_audio = false;
    cfg.enable_cross_attention = false;
    cfg.vision_feature_dim = 256;  // Reduced for memory
    cfg.audio_feature_dim = 0;
    return cfg;
}

// Optimized 1B config for 8GB VRAM with gradient checkpointing
inline MUSConfig get_1b_optimized_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::LARGE_1B;
    cfg.hidden_dim = 1536;
    cfg.num_layers = 32;
    cfg.num_heads = 24;
    cfg.head_dim = 64;
    cfg.enable_vision = true;
    cfg.enable_audio = true;
    cfg.enable_cross_attention = false;  // Disable for memory savings
    cfg.cross_attention_layers = 0;
    cfg.vision_feature_dim = 1024;
    cfg.audio_feature_dim = 768;
    
    // Enable gradient checkpointing for even less memory
    cfg.enable_gradient_checkpointing = true;
    cfg.checkpoint_interval = 4; // Checkpoint every 4 layers
    
    // Reduced feature dimensions for vision/audio
    cfg.vision_feature_dim = 512;   // Reduced from 1024
    cfg.audio_feature_dim = 384;    // Reduced from 768
    
    return cfg;
}

// Memory-efficient 800M config for 6GB VRAM
inline MUSConfig get_800m_config() {
    MUSConfig cfg;
    cfg.model_size = MUSConfig::CUSTOM;
    cfg.hidden_dim = 1280;    // Between 500M and 1B
    cfg.num_layers = 28;      // Fewer layers than 1B
    cfg.num_heads = 20;
    cfg.head_dim = 64;
    cfg.enable_vision = true;
    cfg.enable_audio = false;  // Disable audio for 6GB
    cfg.enable_cross_attention = false;
    cfg.cross_attention_layers = 0;
    cfg.vision_feature_dim = 512;
    return cfg;
}

// ─── Modular config switching ──────────────────────────────────────────
enum MUSModule {
    MUS_MODULE_CODING,     // Code generation + CUDA
    MUS_MODULE_ANALYTICS,  // Document parsing + Markdown
    MUS_MODULE_GRAPHICS,   // Diagram generation (Mermaid/Graphviz)
    MUS_MODULE_SOUND       // Discrete audio quantization
};

// Auto-select config by VRAM
inline MUSConfig mus_select_config(size_t vram_bytes, MUSModule module = MUS_MODULE_CODING) {
    float vram_gb = vram_bytes / (1024.0f * 1024.0f * 1024.0f);
    MUSConfig cfg;

    if (vram_gb >= 7.5f) cfg = get_700m_6gb_config();
    else                  cfg = get_400m_4gb_config();

    // Tune for module
    switch (module) {
        case MUS_MODULE_CODING:
            cfg.enable_vision = false;
            break;
        case MUS_MODULE_ANALYTICS:
            cfg.enable_vision = true;
            cfg.vision_feature_dim = std::min(cfg.vision_feature_dim, 256);
            break;
        case MUS_MODULE_GRAPHICS:
            cfg.enable_vision = true;
            cfg.vision_feature_dim = std::min(cfg.vision_feature_dim, 512);
            break;
        case MUS_MODULE_SOUND:
            cfg.enable_audio = true;
            cfg.enable_vision = false;
            cfg.audio_feature_dim = 384;
            break;
    }
    return cfg;
}

// Get parameter count for a configuration
inline int mus_total_params(const MUSConfig& cfg) {
    int D = cfg.get_D();
    int L = cfg.get_L();
    int H = cfg.get_H();
    int D_ff = cfg.get_D_ff();
    int V = cfg.get_V();

    int embed_params = V * D;
    int per_block = 3 * D * D +  // qkv matrix
                    D * D +      // output matrix
                    3 * D * D_ff; // SwiGLU: gate, up, down
    int norm_params = 2 * D * L + D; // layer norms + final norm
    int cross_attn_params = cfg.enable_cross_attention ?
                           cfg.cross_attention_layers * (D * D * 3 + D * D) : 0;

    int total = embed_params + L * per_block + norm_params + cross_attn_params;
    return total;
}

// Estimate VRAM usage for model training (FP16)
inline size_t mus_vram_usage(const MUSConfig& cfg) {
    int total_params = mus_total_params(cfg);
    size_t weights_bytes = (size_t)total_params * sizeof(half);
    size_t gradients_bytes = (size_t)total_params * sizeof(half);
    size_t adam_m_bytes = (size_t)total_params * sizeof(half);
    size_t adam_v_bytes = (size_t)total_params * sizeof(half);
    size_t workspace_bytes = 512 * 1024 * 1024;
    size_t misc_bytes = 50 * 1024 * 1024;

    return weights_bytes + gradients_bytes + adam_m_bytes + adam_v_bytes + workspace_bytes + misc_bytes;
}

// Memory-optimized VRAM estimate (FP16 Adam, reduced workspace, checkpointing)
inline size_t mus_vram_usage_optimized(const MUSConfig& cfg) {
    int total_params = mus_total_params(cfg);

    size_t weights_bytes = (size_t)total_params * sizeof(half);
    size_t gradients_bytes = (size_t)total_params * sizeof(half);
    size_t adam_m_bytes = (size_t)total_params * sizeof(half);  // FP16 Adam
    size_t adam_v_bytes = (size_t)total_params * sizeof(half);
    size_t workspace_bytes = 256 * 1024 * 1024;  // Reduced 256MB
    int D = cfg.get_D(), L = cfg.get_L();
    int S = std::min(256, cfg.max_seq_len);
    size_t activation_bytes = (size_t)L * S * D * sizeof(half);  // checkpointed grad
    size_t multimodal_bytes = cfg.enable_vision || cfg.enable_audio ? 64 * 1024 * 1024 : 0;

    return weights_bytes + gradients_bytes + adam_m_bytes + adam_v_bytes +
           workspace_bytes + activation_bytes + multimodal_bytes;
}

// Optimal batch size for given VRAM budget
inline int get_optimal_batch_size(const MUSConfig& cfg, size_t vram_budget_bytes) {
    int B = 1;
    size_t usage = mus_vram_usage_optimized(cfg);
    if (usage * 2 <= vram_budget_bytes) B = 2;
    return B;
}

MUSContext* mus_create_context(size_t workspace_bytes = 512 * 1024 * 1024);
void mus_destroy_context(MUSContext* ctx);

// ─── Weight table ──────────────────────────────────────────────────────
void build_weight_table(const MUSConfig& cfg, float* weights, int vocab_size);

// ─── FP32 operations (legacy, used by mus_inference/mus_train) ────────
float mus_weighted_ce_forward(MUSContext* ctx, const float* logits, const int64_t* labels, const float* weights, float* loss, int B, int S, int V);
void mus_weighted_ce_backward(MUSContext* ctx, const float* logits, const int64_t* labels, const float* weights, float scale, float* dldx, int B, int S, int V);
void mus_rmsnorm_forward(MUSContext* ctx, const float* x, const float* w, float* y, int rows, int cols, float eps = 1e-6f);
void mus_rmsnorm_backward(MUSContext* ctx, const float* x, const float* w, const float* dy, float* dx, float* dw, int rows, int cols, float eps = 1e-6f);
void mus_rope_forward(MUSContext* ctx, float* q, float* k, const int64_t* pos, int B, int H, int S, int D);
void mus_rope_backward(MUSContext* ctx, const int64_t* pos, float* dq, float* dk, int B, int H, int S, int D);
void mus_add_vectors(MUSContext* ctx, float* a, const float* b, int rows, int D);
void mus_swiglu_forward(MUSContext* ctx, const float* x, const float* gate_w, const float* up_w, const float* down_w, float* out, int rows, int dim);
void mus_swiglu_backward(MUSContext* ctx, const float* x, const float* gate_w, const float* up_w, const float* down_w, const float* d_out, float* d_x, float* d_gate_w, float* d_up_w, float* d_down_w, int rows, int dim);
void mus_attention_forward(MUSContext* ctx, const float* x, const float* qkv_w, const float* o_w, const int64_t* pos, float* out, int B, int S, int D, int H);
void mus_attention_backward(MUSContext* ctx, const float* x, const float* qkv_w, const float* o_w, const int64_t* pos, const float* d_out, float* d_x, float* d_qkv_w, float* d_o_w, int B, int S, int D, int H);
void mus_transformer_block_forward(MUSContext* ctx, const float* x_in, float* x_out, const float* attn_qkv_w, const float* attn_o_w, const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w, const float* rms_norm_1_w, const float* rms_norm_2_w, const int64_t* pos, int B, int S, int D, int H);
void mus_transformer_block_backward(MUSContext* ctx, const float* x_in, const float* d_out, const float* attn_qkv_w, const float* attn_o_w, const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w, const float* rms_norm_1_w, const float* rms_norm_2_w, const int64_t* pos, float* d_x_in, float* d_attn_qkv_w, float* d_attn_o_w, float* d_mlp_gate_w, float* d_mlp_up_w, float* d_mlp_down_w, float* d_rms_norm_1_w, float* d_rms_norm_2_w, int B, int S, int D, int H);
void mus_clip_gradients(MUSContext* ctx, float* g, int n, float max_norm);
float mus_check_tensor(const float* d_data, const float* h_ref, int n, const char* name, MUSContext* ctx = nullptr);

// ─── FP16 ↔ FP32 conversion ───────────────────────────────────────────
void mus_convert_f16_to_f32(MUSContext* ctx, const half* src, float* dst, int n);
void mus_convert_f32_to_f16(MUSContext* ctx, const float* src, half* dst, int n);

// ─── FP16 cuBLAS GEMM wrappers ────────────────────────────────────────
// A, B are FP16; C is FP16; compute in FP32
void mus_gemm_f16(MUSContext* ctx,
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const half* A, int lda,
    const half* B, int ldb,
    half* C, int ldc,
    float alpha = 1.0f, float beta = 0.0f);

// ─── Fused Weighted Cross-Entropy Forward (FP16) ──────────────────────
float mus_weighted_ce_forward_f16(
    MUSContext* ctx,
    const half* logits,
    const int64_t* labels,
    const float* weights,
    float* loss,
    int B, int S, int V
);

// ─── Fused Weighted Cross-Entropy Backward (FP16) ─────────────────────
void mus_weighted_ce_backward_f16(
    MUSContext* ctx,
    const half* logits,
    const int64_t* labels,
    const float* weights,
    float grad_loss_scale,
    half* dldx_out,
    int B, int S, int V
);

// ─── Fused AdamW Step (FP32 weights) ──────────────────────────────────
void mus_adamw_step(
    MUSContext* ctx,
    float* params, float* exp_avg, float* exp_avg_sq,
    const float* grads, int numel,
    float lr, float beta1, float beta2,
    float eps, float weight_decay, int step
);

// ─── Fused AdamW Step (FP16 weights, FP32 internal compute) ──────────
void mus_adamw_step_f16(
    MUSContext* ctx,
    half* params, half* exp_avg, half* exp_avg_sq,
    const half* grads, int numel,
    float lr, float beta1, float beta2,
    float eps, float weight_decay, int step
);

// ─── Fused RMSNorm Forward (FP16) ─────────────────────────────────────
void mus_rmsnorm_forward_f16(
    MUSContext* ctx,
    const half* x, const float* weight, half* y,
    int rows, int cols, float eps = 1e-6f
);

// ─── Fused RMSNorm Backward (FP16) ────────────────────────────────────
void mus_rmsnorm_backward_f16(
    MUSContext* ctx,
    const half* x, const float* weight, const half* dldy,
    half* dldx, float* dldw,
    int rows, int cols, float eps = 1e-6f
);

// ─── Fused RoPE Forward (FP16, in-place on Q and K) ───────────────────
void mus_rope_forward_f16(
    MUSContext* ctx,
    half* q, half* k, const int64_t* pos,
    int B, int H, int S, int D
);

// ─── RoPE Backward (FP16) ─────────────────────────────────────────────
void mus_rope_backward_f16(
    MUSContext* ctx,
    const int64_t* pos,
    half* dq, half* dk,
    int B, int H, int S, int D
);

// ─── Vector add (for residual connections, FP16) ──────────────────────
void mus_add_vectors_f16(MUSContext* ctx, half* a, const half* b, int rows, int D);

// ─── Fused SwiGLU Forward (FP16) ──────────────────────────────────────
// dim = hidden_dim, d_ff = intermediate FFN dimension
void mus_swiglu_forward_f16(
    MUSContext* ctx,
    const half* x,
    const half* gate_w, const half* up_w, const half* down_w,
    half* out,
    int rows, int dim, int d_ff
);

// ─── Fused SwiGLU Backward (FP16) ─────────────────────────────────────
void mus_swiglu_backward_f16(
    MUSContext* ctx,
    const half* x,
    const half* gate_w, const half* up_w, const half* down_w,
    const half* d_out,
    half* d_x,
    half* d_gate_w, half* d_up_w, half* d_down_w,
    int rows, int dim, int d_ff
);

// ─── Attention Forward (FP16) ─────────────────────────────────────────
void mus_attention_forward_f16(
    MUSContext* ctx,
    const half* x,
    const half* qkv_w, const half* o_w,
    const int64_t* pos,
    half* out,
    int B, int S, int D, int H
);

// ─── Attention Backward (FP16) ────────────────────────────────────────
void mus_attention_backward_f16(
    MUSContext* ctx,
    const half* x,
    const half* qkv_w, const half* o_w,
    const int64_t* pos,
    const half* d_out,
    half* d_x,
    half* d_qkv_w, half* d_o_w,
    int B, int S, int D, int H
);

// ─── Single Transformer Block Forward (FP16) ──────────────────────────
void mus_transformer_block_forward_f16(
    MUSContext* ctx,
    const half* x_in, half* x_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos,
    int B, int S, int D, int H, int d_ff
);

// ─── Transformer Block Backward (FP16) ────────────────────────────────
void mus_transformer_block_backward_f16(
    MUSContext* ctx,
    const half* x_in,
    const half* d_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos,
    half* d_x_in,
    half* d_attn_qkv_w, half* d_attn_o_w,
    half* d_mlp_gate_w, half* d_mlp_up_w, half* d_mlp_down_w,
    float* d_rms_norm_1_w, float* d_rms_norm_2_w,
    int B, int S, int D, int H, int d_ff
);

// ─── Weight struct (FP16 storage) ─────────────────────────────────────
struct MUSWeightsF16 {
    half* embed;           // [V, D]
    half* attn_qkv_w;     // [L, D, 3*D]
    half* attn_o_w;       // [L, D, D]
    half* mlp_gate_w;     // [L, D, D_ff]
    half* mlp_up_w;       // [L, D, D_ff]
    half* mlp_down_w;     // [L, D_ff, D]
    float* rms_norm_1_w;  // [L, D]  (RMSNorm stays FP32 for precision)
    float* rms_norm_2_w;  // [L, D]
    float* final_norm_w;  // [D]
    
    // Multimodal weights
    half* vision_embed;    // [vision_vocab, D] — vision token embeddings
    half* audio_embed;    // [audio_vocab, D] — audio token embeddings
    half* cross_attn_qkv_w; // [cross_attn_layers, D, 3*D] — cross-attention weights
    half* cross_attn_o_w;   // [cross_attn_layers, D, D]
    half* vision_proj_w;     // [D, vision_feature_dim] — vision feature projection
    half* audio_proj_w;     // [D, audio_feature_dim] — audio feature projection
    half* vision_cross_qkv_w; // [L, D, 3*D] — vision cross-attention
    half* vision_cross_o_w;   // [L, D, D]
    half* audio_cross_qkv_w;   // [L, D, 3*D] — audio cross-attention
    half* audio_cross_o_w;     // [L, D, D]
    
    // Multimodal gating weights
    half* vision_gate_w;      // [L, D] — vision gating weights
    half* audio_gate_w;      // [L, D] — audio gating weights
    half* modality_fuse_w;   // [L, D, 3*D] — modality fusion weights
};

// ─── Memory Management ─────────────────────────────────────────────────
void mus_optimize_memory_usage(MUSContext* ctx, const MUSConfig& cfg, 
                              size_t* current_vram, size_t target_vram);
void mus_allocate_on_demand(MUSContext* ctx, const MUSConfig& cfg, 
                           int batch_size, int sequence_length);
void mus_free_unused_buffers(MUSContext* ctx);

// ─── Memory Management Wrappers ───────────────────────────────────────
void mus_optimize_memory_usage_f16(
    MUSContext* ctx,
    const MUSConfig& cfg,
    size_t* current_vram,
    size_t target_vram
);

void mus_allocate_on_demand_f16(
    MUSContext* ctx,
    const MUSConfig& cfg,
    int batch_size,
    int sequence_length,
    half** workspace,
    size_t* workspace_size
);

// ─── Multimodal processing (FP16) ───────────────────────────────────────
void mus_vision_encoder_forward_f16(
    MUSContext* ctx,
    const half* vision_features,
    const half* vision_embed,
    half* vision_tokens,
    int B, int N, int D, int vision_D
);

void mus_audio_encoder_forward_f16(
    MUSContext* ctx,
    const half* audio_features,
    const half* audio_embed,
    half* audio_tokens,
    int B, int T, int D, int audio_D
);

void mus_cross_attention_forward_f16(
    MUSContext* ctx,
    const half* query,
    const half* key,
    const half* value,
    const half* cross_qkv_w,
    const half* cross_o_w,
    half* output,
    int B, int S_q, int S_k, int D, int H
);

void mus_modality_gate_forward_f16(
    MUSContext* ctx,
    const half* text_hidden,
    const half* vision_hidden,
    const half* audio_hidden,
    const half* vision_gate_w,
    const half* audio_gate_w,
    half* fused_hidden,
    int B, int S, int D
);

// ─── Complete Training Step (FP16) ────────────────────────────────────
void mus_training_step_f16(
    MUSContext* ctx,
    const MUSConfig& cfg,
    const MUSWeightsF16& w,
    MUSWeightsF16& grad,         // FP16 gradients
    float* master_w,             // [total_params] FP32 master weights
    float* opt_m,                // [total_params] FP32 Adam momentum (on CPU)
    float* opt_v,                // [total_params] FP32 Adam variance (on CPU)
    const int64_t* text_input_ids,
    const int64_t* text_labels,
    const int64_t* vision_input_ids,    // Optional vision tokens
    const int64_t* audio_input_ids,    // Optional audio tokens
    const int64_t* pos,
    const float* token_weights,
    half* logits_buffer,
    int B, int S,
    int global_step,
    float lr,
    float loss_scale,
    float* f32_scratch,         // [max_layer_params] FP32 scratch for Adam/layer
    bool has_vision = false,
    bool has_audio = false
);

// ─── Embedding kernels (FP32, legacy) ─────────────────────────────────
__global__ void embed_fwd_kernel(const float* __restrict__ table,
    const int* __restrict__ input, float* __restrict__ out, int B, int S, int D, int V);
__global__ void embed_bwd_kernel(const float* __restrict__ out_grad,
    const int* __restrict__ input, float* __restrict__ table_grad, int B, int S, int D, int V);

// ─── Embedding kernels (FP16) ─────────────────────────────────────────
__global__ void embed_fwd_kernel_f16(const half* __restrict__ table,
    const int* __restrict__ input, half* __restrict__ out, int B, int S, int D, int V);
__global__ void embed_bwd_kernel_f16(const half* __restrict__ out_grad,
    const int* __restrict__ input, float* __restrict__ table_grad_f32, int B, int S, int D, int V);
__global__ void f32_to_f16_grad_kernel(const float* __restrict__ src, half* __restrict__ dst, int n);

// ─── Gradient Clipping (FP16) ─────────────────────────────────────────
void mus_clip_gradients_f16(MUSContext* ctx, half* g, int n, float max_norm);

// ─── Debug: find first NaN in FP16 buffer ─────────────────────────────
int mus_check_nan_f16(MUSContext* ctx, const half* d_data, int n, const char* name);

// ─── Loss scaling ─────────────────────────────────────────────────────
void mus_scale_gradients_f16(MUSContext* ctx, half* g, int n, float scale);
void mus_unscale_gradients_f16(MUSContext* ctx, half* g, int n, float inv_scale);
void mus_unscale_f32(MUSContext* ctx, float* g, int n, float inv_scale);
