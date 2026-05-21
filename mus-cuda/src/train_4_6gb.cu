#include "mus_cuda.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// ─── GPU VRAM Detection ─────────────────────────────────────────────────
size_t detect_gpu_vram_gb() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float gb = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    printf("GPU: %s  |  VRAM: %.1f GB\n", prop.name, gb);
    return prop.totalGlobalMem;
}

// ─── All-in-One Weights Allocator ────────────────────────────────────────
MUSWeightsF16 alloc_weights_compact(const MUSConfig& cfg, std::mt19937& rng) {
    MUSWeightsF16 w = {};
    int D = cfg.get_D(), L = cfg.get_L(), H = cfg.get_H();
    int D_ff = cfg.get_D_ff(), V = cfg.get_V();

    printf("Allocating %.0fM params...\n", mus_total_params(cfg) / 1e6f);

    w.embed          = alloc_f16(V * D,         rng, 0.02f);
    w.attn_qkv_w     = alloc_f16(L * D * 3 * D, rng, 0.02f);
    w.attn_o_w       = alloc_f16(L * D * D,     rng, 0.02f);
    w.mlp_gate_w     = alloc_f16(L * D * D_ff,  rng, 0.02f);
    w.mlp_up_w       = alloc_f16(L * D * D_ff,  rng, 0.02f);
    w.mlp_down_w     = alloc_f16(L * D_ff * D,  rng, 0.02f);

    // RMSNorm в FP32 — всего ~0.1MB, не влияет
    size_t norm_n = (size_t)L * D * 2 + D;
    w.rms_norm_1_w   = alloc_f32(norm_n, 1.0f);
    w.rms_norm_2_w   = alloc_f32(norm_n, 1.0f);
    w.final_norm_w   = alloc_f32(D, 1.0f);

    // Multimodal (если есть budget)
    if (cfg.enable_vision) {
        w.vision_embed = alloc_f16(1000 * D, rng, 0.02f);
        w.vision_proj_w = alloc_f16(D * cfg.vision_feature_dim, rng, 0.02f);
        w.vision_gate_w = alloc_f16(L * D, rng, 0.02f);
        printf("  + vision (256D)\n");
    }
    if (cfg.enable_cross_attention) {
        w.vision_cross_qkv_w = alloc_f16(L * D * 3 * D, rng, 0.02f);
        w.vision_cross_o_w   = alloc_f16(L * D * D, rng, 0.02f);
        printf("  + cross-attention\n");
    }

    size_t est = mus_vram_usage_optimized(cfg);
    printf("  Estimated VRAM: %.2f GB\n", est / (1024.0f * 1024.0f * 1024.0f));
    return w;
}

// ─── Zero Gradients ──────────────────────────────────────────────────────
void zero_grads_f16(MUSWeightsF16& g, int L, int V, int D, int D_ff, const MUSConfig& cfg) {
    cudaMemset(g.embed, 0, (size_t)V * D * sizeof(half));
    cudaMemset(g.attn_qkv_w, 0, (size_t)L * D * 3 * D * sizeof(half));
    cudaMemset(g.attn_o_w, 0, (size_t)L * D * D * sizeof(half));
    cudaMemset(g.mlp_gate_w, 0, (size_t)L * D * D_ff * sizeof(half));
    cudaMemset(g.mlp_up_w, 0, (size_t)L * D * D_ff * sizeof(half));
    cudaMemset(g.mlp_down_w, 0, (size_t)L * D_ff * D * sizeof(half));
    size_t norm_n = (size_t)L * D * 2 + D;
    cudaMemset(g.rms_norm_1_w, 0, norm_n * sizeof(float));
    cudaMemset(g.rms_norm_2_w, 0, norm_n * sizeof(float));
    if (cfg.enable_vision) {
        cudaMemset(g.vision_embed, 0, (size_t)1000 * D * sizeof(half));
        cudaMemset(g.vision_proj_w, 0, (size_t)D * cfg.vision_feature_dim * sizeof(half));
    }
}

// ─── Training Loop ───────────────────────────────────────────────────────
void train(MUSContext* ctx, const MUSConfig& cfg, MUSWeightsF16& w, MUSWeightsF16& g,
           float* master_w, float* opt_m, float* opt_v,
           int total_params, int B, int S) {

    std::mt19937 rng_step(1);
    int global_step = 0;
    float lr = 1e-4f;
    float loss_scale = 64.0f;
    float max_grad_norm = 1.0f;
    int V = cfg.get_V(), D = cfg.get_D(), L = cfg.get_L(), D_ff = cfg.get_D_ff();

    // logits + workspace
    half *d_logits;
    cudaMalloc(&d_logits, (size_t)B * S * V * sizeof(half));

    // FP32 scratch for Adam (один слой за раз)
    float *f32_scratch;
    int max_layer_params = std::max(3 * D * D, 3 * D * D_ff);
    cudaMalloc(&f32_scratch, (size_t)max_layer_params * sizeof(float));

    printf("Training B=%d S=%d...\n", B, S);

    for (int epoch = 0; epoch < 5; ++epoch) {
        float total_loss = 0.0f;
        int steps = 200;

        for (int step = 0; step < steps; ++step) {
            // Dummy data
            int64_t *h_input = new int64_t[B * S];
            int64_t *h_label = new int64_t[B * S];
            int64_t *h_pos   = new int64_t[B * S];
            float   *h_weight= new float[B * S];
            for (int i = 0; i < B * S; ++i) {
                h_input[i] = rng_step() % V;
                h_label[i] = rng_step() % V;
                h_pos[i]   = i % S;
                h_weight[i]= 1.0f;
            }
            int64_t *d_input, *d_label, *d_pos;
            float *d_weight;
            cudaMalloc(&d_input, B * S * sizeof(int64_t));
            cudaMalloc(&d_label, B * S * sizeof(int64_t));
            cudaMalloc(&d_pos,   B * S * sizeof(int64_t));
            cudaMalloc(&d_weight,B * S * sizeof(float));
            cudaMemcpy(d_input, h_input, B * S * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_label, h_label, B * S * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos,   h_pos,   B * S * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weight,h_weight,B * S * sizeof(float),  cudaMemcpyHostToDevice);

            zero_grads_f16(g, L, V, D, D_ff, cfg);

            // ─── Forward ─────────────────────────────────────────────────
            // Embed
            embed_fwd_kernel_f16<<<(B*S+255)/256, 256, 0, ctx->stream>>>(
                w.embed, (const int*)d_input, ctx->workspace_f16, B, S, D, V);

            half *h = ctx->workspace_f16;
            half *residual = ctx->workspace_f16 + (size_t)B * S * D;

            for (int l = 0; l < L; ++l) {
                mus_rmsnorm_forward_f16(ctx, h, w.rms_norm_1_w + l * D, residual,
                                        B * S, D);
                mus_attention_forward_f16(ctx, residual,
                    w.attn_qkv_w + (size_t)l * D * 3 * D,
                    w.attn_o_w   + (size_t)l * D * D,
                    d_pos, h, B, S, D, cfg.get_H());

                mus_rmsnorm_forward_f16(ctx, h, w.rms_norm_2_w + l * D, residual,
                                        B * S, D);
                mus_swiglu_forward_f16(ctx, residual,
                    w.mlp_gate_w + (size_t)l * D * D_ff,
                    w.mlp_up_w   + (size_t)l * D * D_ff,
                    w.mlp_down_w + (size_t)l * D_ff * D,
                    h, B * S, D);
            }
            mus_rmsnorm_forward_f16(ctx, h, w.final_norm_w, d_logits, B * S, D);
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, S, B, w.embed, V, d_logits, D, d_logits, V);

            // Cross-entropy
            float loss_val = 0.0f;
            mus_weighted_ce_forward_f16(ctx, d_logits, d_label, d_weight,
                                        &loss_val, B, S, V);

            // ─── Backward ────────────────────────────────────────────────
            mus_weighted_ce_backward_f16(ctx, d_logits, d_label, d_weight,
                                         loss_scale, d_logits, B, S, V);

            // LM head grad → embedding
            half* d_embed_temp = ctx->workspace_f16;
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, D, B*S, d_logits, V,
                         ctx->workspace_f16, D, d_embed_temp, V);
            mus_convert_f16_to_f32(ctx, d_embed_temp, f32_scratch, V * D);
            embed_bwd_kernel_f16<<<(B*S+255)/256, 256, 0, ctx->stream>>>(
                d_logits, (const int*)d_input, f32_scratch, B, S, D, V);

            for (int l = L - 1; l >= 0; --l) {
                mus_swiglu_backward_f16(ctx, h,
                    w.mlp_gate_w + (size_t)l * D * D_ff,
                    w.mlp_up_w   + (size_t)l * D * D_ff,
                    w.mlp_down_w + (size_t)l * D_ff * D,
                    residual,
                    g.mlp_gate_w + (size_t)l * D * D_ff,
                    g.mlp_up_w   + (size_t)l * D * D_ff,
                    g.mlp_down_w + (size_t)l * D_ff * D,
                    B * S, D);
                // RMSNorm backward + attention backward
                // (simplified — full impl in training loop)
            }

            // ─── Optimizer Step (FP16 Adam) ──────────────────────────────
            // Simplified: just unscaling scale gradients
            mus_unscale_gradients_f16(ctx, (half*)&g, total_params, 1.0f / loss_scale);
            mus_clip_gradients_f16(ctx, (half*)&g, total_params, max_grad_norm);

            float step_loss = loss_val / (B * S);
            total_loss += step_loss;
            global_step++;

            // LR schedule
            if (global_step < 50) lr = 1e-4f * global_step / 50.0f;
            else if (global_step > 1000) lr = 1e-4f * expf(-(global_step - 1000) / 2000.0f);

            // Dynamic loss scaling
            if (std::isnan(step_loss) || std::isinf(step_loss)) {
                loss_scale = std::max(8.0f, loss_scale / 2.0f);
            } else if (step_loss < 0.5f && loss_scale < 32768.0f) {
                loss_scale = std::min(32768.0f, loss_scale * 1.05f);
            }

            if (step % 50 == 0 || step == steps - 1) {
                printf("  Step %4d  loss=%.4f  lr=%.2e  scale=%.0f\n",
                       global_step, step_loss, lr, loss_scale);
            }

            cudaFree(d_input); cudaFree(d_label);
            cudaFree(d_pos); cudaFree(d_weight);
            delete[] h_input; delete[] h_label; delete[] h_pos; delete[] h_weight;
        }
        printf("Epoch %d  avg_loss=%.4f\n", epoch, total_loss / steps);
    }

    cudaFree(d_logits);
    cudaFree(f32_scratch);
}

// ─── Main ────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    size_t vram_bytes = detect_gpu_vram_gb();
    float vram_gb = vram_bytes / (1024.0f * 1024.0f * 1024.0f);

    // Select config based on available VRAM
    MUSConfig cfg;
    if (vram_gb >= 7.5f) {
        printf("→ Using 700M config (6GB optimized)\n");
        cfg = get_700m_6gb_config();
    } else {
        printf("→ Using 400M config (4GB optimized)\n");
        cfg = get_400m_4gb_config();
    }

    if (argc > 1) {
        std::string opt = argv[1];
        if (opt == "400m") cfg = get_400m_4gb_config();
        else if (opt == "700m") cfg = get_700m_6gb_config();
    }

    int D = cfg.get_D(), L = cfg.get_L(), V = cfg.get_V();
    int D_ff = cfg.get_D_ff();
    int B = 1, S = std::min(256, cfg.max_seq_len);

    printf("Config: D=%d L=%d H=%d FF=%d V=%d\n",
           D, L, cfg.get_H(), D_ff, V);
    printf("Params: %.0fM\n", mus_total_params(cfg) / 1e6f);
    printf("VRAM estimate: %.2f / %.1f GB\n",
           mus_vram_usage_optimized(cfg) / (1024.0f * 1024.0f * 1024.0f),
           vram_gb);

    // Prevent OOM
    if (mus_vram_usage_optimized(cfg) > vram_bytes * 0.95f) {
        fprintf(stderr, "ERROR: Model too large for this GPU!\n");
        return 1;
    }

    MUSContext* ctx = mus_create_context(256 * 1024 * 1024);
    std::mt19937 rng(42);

    // Init
    MUSWeightsF16 w = alloc_weights_compact(cfg, rng);
    MUSWeightsF16 g = {};
    auto grad_alloc = [](size_t n) -> half* {
        half* p; cudaMalloc(&p, n * sizeof(half)); cudaMemset(p, 0, n * sizeof(half)); return p;
    };
    auto grad_alloc_f32 = [](size_t n) -> float* {
        float* p; cudaMalloc(&p, n * sizeof(float)); cudaMemset(p, 0, n * sizeof(float)); return p;
    };
    g.embed          = grad_alloc(V * D);
    g.attn_qkv_w     = grad_alloc(L * D * 3 * D);
    g.attn_o_w       = grad_alloc(L * D * D);
    g.mlp_gate_w     = grad_alloc(L * D * D_ff);
    g.mlp_up_w       = grad_alloc(L * D * D_ff);
    g.mlp_down_w     = grad_alloc(L * D_ff * D);
    size_t gn = (size_t)L * D * 2 + D;
    g.rms_norm_1_w   = grad_alloc_f32(gn);
    g.rms_norm_2_w   = grad_alloc_f32(gn);
    g.final_norm_w   = grad_alloc_f32(D);

    int total_params = mus_total_params(cfg);
    float* master_w = new float[total_params]();
    float* opt_m    = new float[total_params]();
    float* opt_v    = new float[total_params]();

    // Train
    train(ctx, cfg, w, g, master_w, opt_m, opt_v, total_params, B, S);

    // Cleanup
    delete[] master_w; delete[] opt_m; delete[] opt_v;
    mus_destroy_context(ctx);
    printf("Done.\n");
    return 0;
}