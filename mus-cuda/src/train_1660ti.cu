#include "mus_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <fstream>
#include <algorithm>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = call;                                        \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); }               \
} while(0)

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point s;
    Timer() : s(clock::now()) {}
    float ms() { return std::chrono::duration<float,std::milli>(clock::now()-s).count(); }
};

// ─── Binary data loader ─────────────────────────────────────────────
struct TrainData {
    std::vector<int> data;
    int num_samples, seq_len;
};

TrainData load_binary_cache(const std::string& path) {
    TrainData td;
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    td.data.resize(bytes / sizeof(int));
    f.read((char*)td.data.data(), bytes);
    td.seq_len = 256;
    td.num_samples = (int)td.data.size() / td.seq_len;
    printf("  Loaded %d samples x %d tokens (%zu MB)\n",
           td.num_samples, td.seq_len, bytes/1024/1024);
    return td;
}

// GPU weight initialization kernel (fast, no CPU RNG bottleneck)
__global__ void init_weights_kernel(half* w, int n, float stddev, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Simple deterministic pseudo-random using LCG (avoids CPU RNG bottleneck)
    unsigned int rng = seed + i * 2654435761u;
    rng = (rng * 1103515245u + 12345u) & 0x7fffffff;
    float u1 = (float)rng / 2147483648.0f;
    rng = (rng * 1103515245u + 12345u) & 0x7fffffff;
    float u2 = (float)rng / 2147483648.0f;
    // Box-Muller transform for normal distribution
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1e-10f)));
    float theta = 6.2831853f * u2;
    w[i] = __float2half(r * cosf(theta) * stddev);
}

// ─── FP16 weight buffer ────────────────────────────────────────────
struct WeightBuf {
    half* w;
    half* g;
    half* m;
    half* v;
    int n;
};

static WeightBuf alloc_f16(int n, std::mt19937& rng, float stddev, bool scale_out=false) {
    WeightBuf buf;
    buf.n = n;
    CUDA_CHECK(cudaMalloc(&buf.w, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.g, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.m, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.v, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.g, 0, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.m, 0, (size_t)n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.v, 0, (size_t)n * sizeof(half)));
    // Init weights using GPU kernel for speed
    int bs = 256, grid = (n + bs - 1) / bs;
    int seed = (int)(stddev * 10000) ^ (int)(n);
    init_weights_kernel<<<grid, bs, 0, 0>>>(buf.w, n, stddev, seed);
    CUDA_CHECK(cudaGetLastError());
    return buf;
}

// ─── GPU detection ─────────────────────────────────────────────────
static const char* gpu_name() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
}

static size_t gpu_vram_bytes() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.totalGlobalMem;
}

static int gpu_compute_capability() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major * 10 + prop.minor;
}

// ─── Main ──────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("\n");
    printf("  ╔══════════════════════════════════════════╗\n");
    printf("  ║  MUS-CUDA — GTX 1660 Ti Edition         ║\n");
    printf("  ║  500M params  |  FP16 Training           ║\n");
    printf("  ╚══════════════════════════════════════════╝\n");
    printf("\n");
    printf("  GPU:           %s\n", prop.name);
    printf("  VRAM:          %.0f MB\n", prop.totalGlobalMem / 1e6f);
    printf("  Compute Cap:   %d.%d\n", prop.major, prop.minor);

    bool is_turing = (prop.major == 7 && prop.minor == 5);
    if (!is_turing) {
        printf("  WARNING: Not a Turing GPU — optimizations may not apply\n");
    }
    if (prop.totalGlobalMem < 5.5e9) {
        fprintf(stderr, "  ERROR: Need >=6GB VRAM (have %.0f MB)\n",
                prop.totalGlobalMem / 1e6f);
        return 1;
    }

    // ─── Config ────────────────────────────────────────────────────
    MUSConfig cfg;
    if (argc > 2 && strcmp(argv[2], "800m") == 0) {
        cfg = get_800m_gtx1660ti_config();
        printf("  Mode: 800M (D=%d L=%d V=%d)\n", cfg.hidden_dim, cfg.num_layers, cfg.vocab_size);
    } else {
        cfg = get_500m_gtx1660ti_config();
    }
    int D = cfg.get_D(), V = cfg.get_V(), L = cfg.get_L();
    int H = cfg.get_H(), d_ff = cfg.get_D_ff();
    int B = 1, S = 256;
    int rows = B * S;
    int max_seq_len = cfg.max_seq_len;

    printf("\n  ─── Model ──────────────────────────────────\n");
    printf("  Architecture:  500M (D=%d L=%d H=%d FF=%d V=%d)\n",
           D, L, H, d_ff, V);
    printf("  Batch/Seq:     B=%d S=%d\n", B, S);
    printf("  Checkpoint:    every %d layers\n", cfg.checkpoint_interval);
    printf("  Vision:        disabled (all memory for text)\n");

    size_t vram_est = mus_vram_usage_optimized(cfg);
    printf("  Estimated VRAM: %.2f / %.0f GB\n",
           vram_est / 1e9, prop.totalGlobalMem / 1e9);
    if (vram_est > prop.totalGlobalMem * 0.95) {
        fprintf(stderr, "  ERROR: Model too large for this GPU!\n");
        return 1;
    }

    // ─── Data ──────────────────────────────────────────────────────
    std::string cache_path = "data/train_cache.bin";
    if (argc > 1) cache_path = argv[1];
    printf("\n  ─── Data ───────────────────────────────────\n");
    TrainData td = load_binary_cache(cache_path);
    int N = td.num_samples;

    // ─── RNG ───────────────────────────────────────────────────────
    std::mt19937 rng(42);

    // ─── Context ───────────────────────────────────────────────────
    // 256 MB workspace (FP16) — достаточно для 500M с checkpointing
    size_t ws_size = 64 * 1024 * 1024;
    printf("\n  ─── Init ───────────────────────────────────\n");
    printf("  Workspace: %zu MB\n", ws_size / 1024 / 1024);
    MUSContext* ctx = mus_create_context(ws_size);
    printf("  Context created\n");

    // ─── Allocate weights ──────────────────────────────────────────
    printf("  Allocating weights...\n");

    WeightBuf w_embed = alloc_f16(V * D, rng, 0.02f);
    printf("    embed: %d x %d\n", V, D);

    auto alloc_f32 = [&](int n, float val=1.0f) {
        float* d;
        CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(float)));
        std::vector<float> h(n, val);
        CUDA_CHECK(cudaMemcpy(d, h.data(), (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice));
        return d;
    };

    std::vector<WeightBuf> w_qkv(L), w_o(L), w_gate(L), w_up(L), w_down(L);
    std::vector<float*> w_rn1(L), w_rn2(L);
    std::vector<float*> g_rn1(L), g_rn2(L);

    // Embedding grad buffer (FP32 accumulation) — alloc early to avoid fragmentation
    float *d_embed_grad_f32;
    CUDA_CHECK(cudaMalloc(&d_embed_grad_f32, (size_t)V * D * sizeof(float)));
    printf("    embed_grad_f32: %zu MB\n", (size_t)V * D * sizeof(float) / 1024 / 1024);

    for (int l = 0; l < L; l++) {
        if (l % 4 == 0) { printf("    layer %d/%d...\n", l+1, L); fflush(stdout); }

        w_rn1[l] = alloc_f32(D, 1.0f);
        w_rn2[l] = alloc_f32(D, 1.0f);
        CUDA_CHECK(cudaMalloc(&g_rn1[l], (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_rn2[l], (size_t)D * sizeof(float)));

        w_qkv[l]   = alloc_f16(D * 3 * D,   rng, 0.02f);
        w_o[l]     = alloc_f16(D * D,       rng, 0.01f);
        w_gate[l]  = alloc_f16(D * d_ff,    rng, 0.02f);
        w_up[l]    = alloc_f16(D * d_ff,    rng, 0.02f);
        w_down[l]  = alloc_f16(d_ff * D,    rng, 0.01f);
    }

    float *fn_w = alloc_f32(D, 1.0f);
    float *fn_g;
    CUDA_CHECK(cudaMalloc(&fn_g, (size_t)D * sizeof(float)));

    size_t f16_params = (size_t)V * D + L * ((size_t)D * 3 * D + (size_t)D * D + 3 * (size_t)D * d_ff);
    size_t f32_params = (size_t)L * 2 * D + D;
    double total_params = (double)(f16_params + f32_params);
    double mem_weights = (double)f16_params * sizeof(half) + (double)f32_params * sizeof(float);
    double mem_grads  = (double)f16_params * sizeof(half) + (double)L * 2 * D * sizeof(float) + D * sizeof(float);
    double mem_optim  = (double)f16_params * sizeof(half) * 2;
    printf("\n    Total: %.0fM params\n", total_params / 1e6);
    printf("    Memory: W=%.0fMB G=%.0fMB O=%.0fMB\n",
           mem_weights/1e6, mem_grads/1e6, mem_optim/1e6);

    // ─── Loss accumulator ──────────────────────────────────────────
    float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    // ─── Token weights ─────────────────────────────────────────────
    float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)V * sizeof(float)));
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                          (size_t)V * sizeof(float), cudaMemcpyHostToDevice));

    // ─── Buffers ───────────────────────────────────────────────────
    std::vector<int> h_input((size_t)B * S);
    std::vector<int64_t> h_labels64((size_t)B * S);
    std::vector<int64_t> h_pos((size_t)B * S);
    for (int i = 0; i < B * S; i++) h_pos[i] = (int64_t)(i % S);

    int *d_input_ids;
    int64_t *d_labels64, *d_pos;
    half *d_logits, *d_trace, *d_fn_out;

    // Gradient checkpointing
    int ckpt = cfg.checkpoint_interval;
    int num_ckpt = L / ckpt + 2;  // +2: need one extra so ckpt_slot(L) doesn't collide with ckpt_slot(L-1)
    size_t trace_ckpt_sz = (size_t)num_ckpt * rows * D * sizeof(half);
    size_t trace_full_sz = (size_t)(L + 1) * rows * D * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_input_ids, (size_t)B * S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels64, (size_t)B * S * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * V * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_ckpt_sz));  // checkpoint slots
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)B * S * sizeof(int64_t)));

    // RMSNorm FP32 Adam states
    std::vector<float*> rn1_m(L), rn1_v(L), rn2_m(L), rn2_v(L);
    float *fn_m, *fn_v;
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMalloc(&rn1_m[l], (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn1_v[l], (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn2_m[l], (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn2_v[l], (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn1_m[l], 0, (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn1_v[l], 0, (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn2_m[l], 0, (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn2_v[l], 0, (size_t)D * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&fn_m, (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fn_v, (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMemset(fn_m, 0, (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMemset(fn_v, 0, (size_t)D * sizeof(float)));

    // ─── Training loop ─────────────────────────────────────────────
    int num_epochs = 3;
    int steps_per_epoch = N / B;
    int global_step = 0;

    float base_lr = 1e-4f;
    int warmup_steps = 200;
    float loss_scale = 128.0f;
    float weight_decay = 0.01f;

    printf("\n  ─── Training ───────────────────────────────\n");
    printf("  Epochs: %d  Steps/epoch: %d\n", num_epochs, steps_per_epoch);
    printf("  lr=%.1e  warmup=%d  wd=%.2f  ls=%.0f\n\n",
           base_lr, warmup_steps, weight_decay, loss_scale);

    float best_loss = 1e10f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Timer epoch_timer;
        double total_loss = 0.0;
        int total_valid = 0;

        for (int step = 0; step < steps_per_epoch; step++) {
            int start_idx = step * B;

            // Prepare batch
            for (int b = 0; b < B; b++) {
                int idx = (start_idx + b) % N;
                int* src = td.data.data() + idx * S;
                for (int s = 0; s < S - 1; s++) {
                    h_input[b * S + s] = src[s];
                    h_labels64[b * S + s] = (int64_t)src[s + 1];
                }
                h_input[b * S + (S - 1)] = 0;
                h_labels64[b * S + (S - 1)] = -100;
            }

            CUDA_CHECK(cudaMemcpyAsync(d_input_ids, h_input.data(),
                        (size_t)B*S*sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels64, h_labels64.data(),
                        (size_t)B*S*sizeof(int64_t), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(),
                        (size_t)B*S*sizeof(int64_t), cudaMemcpyHostToDevice, ctx->stream));

            // ═══ FORWARD (with gradient checkpointing) ═══
            // Checkpoints at layers 3, 7, 11, 15, 19 for ckpt=4, L=22
            // Last layer output goes to a dedicated slot to avoid collision (20/4 == 22/4)
            auto ckpt_slot = [&](int layer) -> half* {
                int slot = layer / ckpt;
                return d_trace + (size_t)slot * rows * D;
            };
            auto last_slot = [&]() -> half* {
                return d_trace + (size_t)(num_ckpt - 1) * rows * D;
            };

            // Embed
            embed_fwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
                w_embed.w, d_input_ids, ckpt_slot(0), B, S, D, V);

            half* prev = ckpt_slot(0);

            // Forward through all layers, saving checkpoints
            for (int l = 0; l < L; l++) {
                half* curr;
                if (l == L - 1) {
                    curr = last_slot();  // dedicated slot, no checkpoint collision
                } else if (l % ckpt == ckpt - 1) {
                    curr = ckpt_slot(l + 1);  // checkpoint store
                } else {
                    curr = ctx->workspace_f16;
                    curr += (size_t)(l % ckpt) * rows * D;
                }

                mus_transformer_block_forward_f16(ctx, prev, curr,
                    w_qkv[l].w, w_o[l].w,
                    w_gate[l].w, w_up[l].w, w_down[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    B, S, D, H, d_ff);
                prev = curr;
            }

            // Final norm + LM Head
            // prev = last_slot() = last block output (before RMS norm)
            mus_rmsnorm_forward_f16(ctx, prev, fn_w, d_fn_out, rows, D);
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, rows, D, w_embed.w, V, d_fn_out, rows, d_logits, V);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaGetLastError());

            int valid = 0;
            for (int i = 0; i < B*S; i++)
                if (h_labels64[i] != -100) valid++;

            float loss_val = mus_weighted_ce_forward_f16(ctx, d_logits, d_labels64,
                                                           d_weights, d_loss, B, S, V);
            float step_loss = loss_val / fmaxf(1, valid);
            total_loss += loss_val;
            total_valid += valid;
            global_step++;

            float current_lr = (global_step < warmup_steps)
                ? base_lr * ((float)global_step / warmup_steps)
                : base_lr;

            // ═══ BACKWARD ═══
            CUDA_CHECK(cudaMemsetAsync(w_embed.g, 0, (size_t)V*D*sizeof(half), ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(d_embed_grad_f32, 0, (size_t)V*D*sizeof(float), ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(fn_g, 0, (size_t)D*sizeof(float), ctx->stream));
            for (int l = 0; l < L; l++) {
                CUDA_CHECK(cudaMemsetAsync(w_qkv[l].g, 0, (size_t)D*3*D*sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_o[l].g, 0, (size_t)D*D*sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_gate[l].g, 0, (size_t)D*d_ff*sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_up[l].g, 0, (size_t)D*d_ff*sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_down[l].g, 0, (size_t)d_ff*D*sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn1[l], 0, (size_t)D*sizeof(float), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn2[l], 0, (size_t)D*sizeof(float), ctx->stream));
            }

            float scale = loss_scale / fmaxf(1, valid);
            mus_weighted_ce_backward_f16(ctx, d_logits, d_labels64, d_weights, scale,
                                          d_logits, B, S, V);

            // Compute d_embed_grad FROM LM HEAD: d_logits @ fn_out^T
            half* d_embed_temp = ctx->workspace_f16;
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, D, rows, d_logits, V, d_fn_out, D, d_embed_temp, V);
            mus_convert_f16_to_f32(ctx, d_embed_temp, d_embed_grad_f32, V*D);
            CUDA_CHECK(cudaMemsetAsync(ctx->workspace_f16, 0, ctx->workspace_size, ctx->stream));

            mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N,
                         D, rows, V, w_embed.w, V, d_logits, V, d_fn_out, D);

            mus_rmsnorm_backward_f16(ctx, prev, fn_w, d_fn_out, d_fn_out, fn_g, rows, D);
            half* d_prev = d_fn_out;

            // Backward through layers (reverse, checkpointed)
            for (int l = L - 1; l >= 0; l--) {
                int ckpt_base = (l / ckpt) * ckpt;
                half* layer_in = ckpt_slot(ckpt_base);

                if (l > ckpt_base) {
                    half* tmp = (half*)((char*)ctx->workspace_f16 + ctx->workspace_size);
                    tmp -= 3 * rows * D;
                    for (int rp = ckpt_base; rp < l; rp++) {
                        half* skip = tmp;
                        half* norm1 = tmp + (size_t)rows * D;
                        half* attn  = tmp + 2 * (size_t)rows * D;
                        if (rp == ckpt_base) {
                            cudaMemcpyAsync(skip, layer_in, (size_t)rows * D * sizeof(half),
                                            cudaMemcpyDeviceToDevice, ctx->stream);
                        }
                        mus_rmsnorm_forward_f16(ctx, skip, w_rn1[rp], norm1, rows, D);
                        mus_attention_forward_f16(ctx, norm1, w_qkv[rp].w, w_o[rp].w,
                            d_pos, attn, B, S, D, H);
                        mus_add_vectors_f16(ctx, attn, skip, rows, D);
                        mus_rmsnorm_forward_f16(ctx, attn, w_rn2[rp], norm1, rows, D);
                        mus_swiglu_forward_f16(ctx, norm1, w_gate[rp].w, w_up[rp].w,
                            w_down[rp].w, norm1, rows, D, d_ff);
                        mus_add_vectors_f16(ctx, norm1, attn, rows, D);
                        cudaMemcpyAsync(skip, norm1, (size_t)rows * D * sizeof(half),
                                        cudaMemcpyDeviceToDevice, ctx->stream);
                        layer_in = skip;
                    }
                }

                mus_clip_gradients_f16(ctx, d_prev, rows * D, 1.0f * loss_scale);
                mus_transformer_block_backward_f16(ctx, layer_in, d_prev,
                    w_qkv[l].w, w_o[l].w,
                    w_gate[l].w, w_up[l].w, w_down[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    d_fn_out,
                    w_qkv[l].g, w_o[l].g,
                    w_gate[l].g, w_up[l].g, w_down[l].g,
                    g_rn1[l], g_rn2[l],
                    B, S, D, H, d_ff);
                mus_clip_gradients_f16(ctx, d_fn_out, rows * D, 1.0f * loss_scale);
                d_prev = d_fn_out;
            }

            // Embedding backward
            embed_bwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
                d_prev, d_input_ids, d_embed_grad_f32, B, S, D, V);
            mus_convert_f32_to_f16(ctx, d_embed_grad_f32, w_embed.g, V*D);

            // ═══ OPTIMIZER ═══
            float inv_scale = 1.0f / loss_scale;

            // All weight types — all layers — fully enabled
            mus_unscale_gradients_f16(ctx, w_embed.g, V*D, inv_scale);
            mus_clip_gradients_f16(ctx, w_embed.g, V*D, 1.0f);
            mus_adamw_step_f16(ctx, w_embed.w, w_embed.m, w_embed.v, w_embed.g, V*D,
                               current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

            // mus_unscale_f32(ctx, fn_g, D, inv_scale);
            // mus_adamw_step(ctx, fn_w, fn_m, fn_v, fn_g, D,
            //                current_lr, 0.9f, 0.999f, 1e-8f, 0.0f, global_step);

            for (int l = 0; l < L; l++) {
                mus_unscale_gradients_f16(ctx, w_qkv[l].g, D*3*D, inv_scale);
                mus_clip_gradients_f16(ctx, w_qkv[l].g, D*3*D, 1.0f);
                mus_adamw_step_f16(ctx, w_qkv[l].w, w_qkv[l].m, w_qkv[l].v, w_qkv[l].g, D*3*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

                mus_unscale_gradients_f16(ctx, w_o[l].g, D*D, inv_scale);
                mus_clip_gradients_f16(ctx, w_o[l].g, D*D, 1.0f);
                mus_adamw_step_f16(ctx, w_o[l].w, w_o[l].m, w_o[l].v, w_o[l].g, D*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

                mus_unscale_gradients_f16(ctx, w_gate[l].g, D*d_ff, inv_scale);
                mus_clip_gradients_f16(ctx, w_gate[l].g, D*d_ff, 1.0f);
                mus_adamw_step_f16(ctx, w_gate[l].w, w_gate[l].m, w_gate[l].v, w_gate[l].g, D*d_ff,
                                   current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

                mus_unscale_gradients_f16(ctx, w_up[l].g, D*d_ff, inv_scale);
                mus_clip_gradients_f16(ctx, w_up[l].g, D*d_ff, 1.0f);
                mus_adamw_step_f16(ctx, w_up[l].w, w_up[l].m, w_up[l].v, w_up[l].g, D*d_ff,
                                   current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

                mus_unscale_gradients_f16(ctx, w_down[l].g, d_ff*D, inv_scale);
                mus_clip_gradients_f16(ctx, w_down[l].g, d_ff*D, 1.0f);
                mus_adamw_step_f16(ctx, w_down[l].w, w_down[l].m, w_down[l].v, w_down[l].g, d_ff*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, weight_decay, global_step);

                // mus_unscale_f32(ctx, g_rn1[l], D, inv_scale);
                // mus_adamw_step(ctx, w_rn1[l], rn1_m[l], rn1_v[l], g_rn1[l], D,
                //                current_lr, 0.9f, 0.999f, 1e-8f, 0.0f, global_step);
                // mus_unscale_f32(ctx, g_rn2[l], D, inv_scale);
                // mus_adamw_step(ctx, w_rn2[l], rn2_m[l], rn2_v[l], g_rn2[l], D,
                //                current_lr, 0.9f, 0.999f, 1e-8f, 0.0f, global_step);
            }

            // Logging (every step)
            double avg = total_loss / fmaxf(1.0f, (float)total_valid);
            printf("  ep %2d/%d step %4d  loss=%.4f  step=%.4f  lr=%.2e\n",
                   epoch+1, num_epochs, step, avg, step_loss, current_lr);
        }

        float et = epoch_timer.ms() / 1000.0f;
        double avg = total_loss / fmaxf(1.0, (double)total_valid);
        if (avg < best_loss) best_loss = (float)avg;
        printf("  ── epoch %2d/%d  loss=%.4f  %.1fs  best=%.4f\n\n",
               epoch+1, num_epochs, avg, et, best_loss);
    }

    printf("  Done! Best loss: %.4f\n", best_loss);
    mus_destroy_context(ctx);
    return 0;
}
