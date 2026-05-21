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

// ══════════════════════════════════════════════════════════════════════
//  Binary cache loader (same as train.cu)
// ══════════════════════════════════════════════════════════════════════

struct PhotoData {
    std::vector<int> data;
    std::vector<int64_t> labels;
    int num_samples, seq_len;
};

PhotoData load_cache(const std::string& path) {
    PhotoData pd;
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    pd.data.resize(bytes / sizeof(int));
    f.read((char*)pd.data.data(), bytes);
    pd.seq_len = 512;
    pd.num_samples = (int)pd.data.size() / pd.seq_len;
    printf("  Loaded %d samples x %d tokens (%zu MB)\n", pd.num_samples, pd.seq_len, bytes/1024/1024);
    return pd;
}

// ══════════════════════════════════════════════════════════════════════
//  FP16 weight allocator
// ══════════════════════════════════════════════════════════════════════

struct WeightBuf {
    half* w;    // weights
    half* g;    // gradients
    half* m;    // Adam momentum
    half* v;    // Adam variance
    int n;      // number of elements
};

WeightBuf alloc_f16(int n, std::mt19937& rng, float stddev, bool ones=false, float scale=1.0f) {
    WeightBuf buf;
    buf.n = n;
    CUDA_CHECK(cudaMalloc(&buf.w, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.g, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.m, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf.v, n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.g, 0, n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.m, 0, n * sizeof(half)));
    CUDA_CHECK(cudaMemset(buf.v, 0, n * sizeof(half)));
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0, stddev);
    for (int i = 0; i < n; i++) {
        float val = ones ? 1.0f : nd(rng);
        if (scale != 1.0f) val *= scale;
        h[i] = val;
    }
    std::vector<half> hh(n);
    for (int i = 0; i < n; i++) hh[i] = __float2half(h[i]);
    CUDA_CHECK(cudaMemcpy(buf.w, hh.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    return buf;
}

// ══════════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int dev; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("MUS-CUDA FP16 Training (500M)\n");
    printf("  GPU: %s  VRAM: %.0f MB\n", prop.name, prop.totalGlobalMem / 1e6f);

    // ─── 500M Model Config ────────────────────────────────────────────
    MUSConfig cfg;
    cfg.vocab_size = 48000;
    cfg.hidden_dim = 1280;
    cfg.num_layers = 22;
    cfg.num_heads = 20;
    cfg.head_dim = 64;
    cfg.ffn_dim = 3456;
    cfg.max_seq_len = 8192;

    int D = cfg.hidden_dim, V = cfg.vocab_size, L = cfg.num_layers;
    int H = cfg.num_heads, d_ff = cfg.ffn_dim;
    int B = 1, S = 256;  // small batch to fit in 6GB
    int rows = B * S;

    printf("  Config: D=%d V=%d L=%d H=%d D_ff=%d B=%d S=%d\n",
           D, V, L, H, d_ff, B, S);

    // ─── Data ─────────────────────────────────────────────────────────
    std::string cache_path = "data/russian_bpe_train_cache.bin";
    if (argc > 1) cache_path = argv[1];
    PhotoData pd = load_cache(cache_path);
    int N = pd.num_samples;
    printf("  Data: %d samples\n", N);

    // ─── RNG ──────────────────────────────────────────────────────────
    std::mt19937 rng(42);

    // ─── Context (FP16 workspace: 512 MB) ─────────────────────────────
    size_t ws_size = 512 * 1024 * 1024;
    printf("  Creating context with %zu MB FP16 workspace...\n", ws_size/1024/1024);
    MUSContext* ctx = mus_create_context(ws_size);

    printf("  Context created\n");

    // ─── Allocate weights ─────────────────────────────────────────────
    // Total params ≈ V×D + L×(4D² + 3·D·d_ff + 2D) + D
    // = 48000×1280 + 22×(4×1280² + 3×1280×3456 + 2×1280) + 1280
    // ≈ 498M params

    printf("  Allocating 500M model weights (FP16)...\n");

    // Embedding
    WeightBuf w_embed = alloc_f16(V * D, rng, 0.02f);
    printf("    embed: %d x %d\n", V, D);

    // RMSNorm weights (stored in FP32 for numerical stability)
    auto alloc_f32 = [&](int n, float val=1.0f) {
        float* d;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        std::vector<float> h(n, val);
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };

    std::vector<WeightBuf> w_qkv(L), w_o(L), w_gate(L), w_up(L), w_down(L);
    std::vector<float*> w_rn1(L), w_rn2(L);
    std::vector<float*> g_rn1(L), g_rn2(L);

    float std_w = 0.02f;

    for (int l = 0; l < L; l++) {
        printf("    Layer %d/%d...", l+1, L); fflush(stdout);

        // RMSNorm weights stored in FP32 (important for precision)
        w_rn1[l] = alloc_f32(D, 1.0f);
        w_rn2[l] = alloc_f32(D, 1.0f);

        // RMSNorm gradients stored in FP32 (accumulated by atomics)
        CUDA_CHECK(cudaMalloc(&g_rn1[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_rn2[l], D * sizeof(float)));

        // QKV: [D, 3*D]
        w_qkv[l] = alloc_f16(D * 3 * D, rng, std_w);
        printf("qkv "); fflush(stdout);

        // Attention output: [D, D]
        w_o[l] = alloc_f16(D * D, rng, std_w, false, 0.2f);
        printf("o "); fflush(stdout);

        // SwiGLU gate+up+down: [D, d_ff], [D, d_ff], [d_ff, D]
        w_gate[l] = alloc_f16(D * d_ff, rng, std_w);
        printf("g "); fflush(stdout);
        w_up[l] = alloc_f16(D * d_ff, rng, std_w);
        printf("u "); fflush(stdout);
        w_down[l] = alloc_f16(d_ff * D, rng, std_w, false, 0.2f);
        printf("d "); fflush(stdout);

        printf(" OK\n");
    }

    // Final norm (FP32)
    float *fn_w = alloc_f32(D, 1.0f);
    float *fn_g;
    CUDA_CHECK(cudaMalloc(&fn_g, D * sizeof(float)));

    size_t f16_params = (size_t)V * D + L * ((size_t)D * 3 * D + (size_t)D * D + 3 * (size_t)D * d_ff);
    size_t f32_params = (size_t)L * 2 * D + D;
    double total_params = (double)(f16_params + f32_params);
    printf("\n  Total params: %.1fM (FP16: %.1fM + FP32: %.1fM)\n",
           total_params / 1e6, (double)f16_params / 1e6, (double)f32_params / 1e6);

    double mem_weights = (double)f16_params * sizeof(half) + (double)f32_params * sizeof(float);
    double mem_grads = (double)f16_params * sizeof(half) + (double)L * 2 * D * sizeof(float) + D * sizeof(float);
    double mem_optim = (double)f16_params * sizeof(half) * 2; // m + v in FP16
    printf("  Memory: weights=%.1f MB grads=%.1f MB optim=%.1f MB\n",
           mem_weights/1e6, mem_grads/1e6, mem_optim/1e6);

    // ─── Token weights ────────────────────────────────────────────────
    float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, V * sizeof(float)));
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), V * sizeof(float), cudaMemcpyHostToDevice));

    // ─── Training buffers ─────────────────────────────────────────────
    // Input/target buffers (host)
    std::vector<int> h_input(B * S);
    std::vector<int64_t> h_labels64(B * S);
    std::vector<int64_t> h_pos(B * S);
    for (int i = 0; i < B * S; i++) h_pos[i] = (int64_t)(i % S);

    // Device buffers
    int *d_input_ids;
    int64_t *d_labels64, *d_pos;
    half *d_logits, *d_trace, *d_fn_out, *d_loss_f32;

    size_t trace_sz = (size_t)(L + 1) * rows * D * sizeof(half);
    CUDA_CHECK(cudaMalloc(&d_input_ids, (size_t)B * S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels64, (size_t)B * S * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * V * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_loss_f32, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_sz));
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)B * S * sizeof(int64_t)));

    // FP32 conversion scratch (for embedding grad accumulation)
    float *d_embed_grad_f32;
    CUDA_CHECK(cudaMalloc(&d_embed_grad_f32, (size_t)V * D * sizeof(float)));

    // ─── Training loop ────────────────────────────────────────────────
    int num_epochs = 100;
    int steps_per_epoch = N / B;
    int global_step = 0;

    float base_lr = 1e-4f;
    int warmup_steps = 200;
    float loss_scale = 128.0f;       // initial loss scale for FP16

    // RMSNorm FP32 Adam states
    std::vector<float*> rn1_m(L), rn1_v(L), rn2_m(L), rn2_v(L);
    float *fn_m, *fn_v;
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMalloc(&rn1_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn1_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn2_m[l], D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rn2_v[l], D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn1_m[l], 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn1_v[l], 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn2_m[l], 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(rn2_v[l], 0, D * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&fn_m, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fn_v, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(fn_m, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(fn_v, 0, D * sizeof(float)));

    printf("\nStarting FP16 training: %d samples, B=%d, %d steps/epoch, %d epochs\n",
           N, B, steps_per_epoch, num_epochs);
    printf("  base_lr=%.1e warmup=%d loss_scale=%.0f\n", base_lr, warmup_steps, loss_scale);

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
                int* src = pd.data.data() + idx * S;
                for (int s = 0; s < S - 1; s++) {
                    h_input[b * S + s] = src[s];
                    h_labels64[b * S + s] = (int64_t)src[s + 1];
                }
                h_input[b * S + (S - 1)] = 0;
                h_labels64[b * S + (S - 1)] = -100;
            }

            // Copy to device
            CUDA_CHECK(cudaMemcpyAsync(d_input_ids, h_input.data(), B*S*sizeof(int),
                                       cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels64, h_labels64.data(), B*S*sizeof(int64_t),
                                       cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), B*S*sizeof(int64_t),
                                       cudaMemcpyHostToDevice, ctx->stream));

            // ═══ FORWARD PASS (all FP16) ═══
            half* trace[23]; // L+1 = 23
            for (int l = 0; l <= L; l++)
                trace[l] = d_trace + (size_t)l * rows * D;

            // Embed
            embed_fwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
                w_embed.w, d_input_ids, trace[0], B, S, D, V);

            // Transformer layers
            for (int l = 0; l < L; l++) {
                mus_transformer_block_forward_f16(ctx, trace[l], trace[l+1],
                    w_qkv[l].w, w_o[l].w,
                    w_gate[l].w, w_up[l].w, w_down[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    B, S, D, H, d_ff);
            }

            // Final norm + LM Head (FP16 GEMM)
            mus_rmsnorm_forward_f16(ctx, trace[L], fn_w, d_fn_out, rows, D);
            // logits[V, rows] = embed[V, D] @ fn_out^T[D, rows]
            // col-major: A(V×D, lda=V), B(rows×D, ldb=rows) w/OP_T→D×rows, C(V×rows, ldc=V)
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, rows, D, w_embed.w, V, d_fn_out, rows, d_logits, V);

            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // Count valid tokens
            int valid = 0;
            for (int i = 0; i < B*S; i++) if (h_labels64[i] != -100) valid++;

            // Compute loss (FP16 forward + FP32 reduction internally)
            float loss_val = mus_weighted_ce_forward_f16(ctx, d_logits, d_labels64,
                                                          d_weights, d_loss_f32, B, S, V);
            float step_loss = loss_val / fmaxf(1, valid);
            total_loss += loss_val;
            total_valid += valid;
            global_step++;

            // Learning rate with warmup
            float current_lr = (global_step < warmup_steps)
                ? base_lr * ((float)global_step / warmup_steps)
                : base_lr;

            // ═══ BACKWARD PASS ═══
            // Zero all FP16 gradients
            CUDA_CHECK(cudaMemsetAsync(w_embed.g, 0, (size_t)V * D * sizeof(half), ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(d_embed_grad_f32, 0, (size_t)V * D * sizeof(float), ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(fn_g, 0, D * sizeof(float), ctx->stream));
            for (int l = 0; l < L; l++) {
                CUDA_CHECK(cudaMemsetAsync(w_qkv[l].g, 0, (size_t)D * 3 * D * sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_o[l].g, 0, (size_t)D * D * sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_gate[l].g, 0, (size_t)D * d_ff * sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_up[l].g, 0, (size_t)D * d_ff * sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(w_down[l].g, 0, (size_t)d_ff * D * sizeof(half), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn1[l], 0, D * sizeof(float), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn2[l], 0, D * sizeof(float), ctx->stream));
            }

            // Loss-scaled CE backward
            float scale = loss_scale / fmaxf(1, valid);
            mus_weighted_ce_backward_f16(ctx, d_logits, d_labels64, d_weights, scale,
                                         d_logits, B, S, V);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // Zero workspace to eliminate garbage
            CUDA_CHECK(cudaMemsetAsync(ctx->workspace_f16, 0, ctx->workspace_size, ctx->stream));

            // d_fn_out = embed^T @ d_logits  [D, rows] = [D, V] @ [V, rows]
            mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N,
                         D, rows, V, w_embed.w, V, d_logits, V, d_fn_out, D);

            // d_embed_grad = d_logits @ fn_out^T  [V, D] = [V, rows] @ [D, rows]^T
            // Use FP32 buffer to combine LM head grad + embedding lookup grad
            CUDA_CHECK(cudaMemsetAsync(d_embed_grad_f32, 0, (size_t)V*D*sizeof(float), ctx->stream));
            // First, convert the already-computed FP16 w_embed.g (from step 3b)
            // Actually we haven't computed it yet - let's compute directly into f32
            // So: compute d_embed = d_logits @ d_fn_out^T directly into f32 buffer
            // But mus_gemm_f16 writes to half*, so we need temporary half storage
            // Use workspace_f16 for temp, then accumulate into f32 buffer
            half* d_embed_temp = ctx->workspace_f16;
            mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                         V, D, rows, d_logits, V, d_fn_out, D, d_embed_temp, V);
            // Convert FP16 LM head grad → FP32 buffer for accumulation
            mus_convert_f16_to_f32(ctx, d_embed_temp, d_embed_grad_f32, V*D);

            // Final norm backward
            mus_rmsnorm_backward_f16(ctx, trace[L], fn_w, d_fn_out, d_fn_out, fn_g, rows, D);
            half* d_prev = d_fn_out;

            // Transformer layers backward (reverse)
            for (int l = L-1; l >= 0; l--) {
                mus_clip_gradients_f16(ctx, d_prev, rows * D, 0.5f * loss_scale);
                mus_transformer_block_backward_f16(ctx, trace[l], d_prev,
                    w_qkv[l].w, w_o[l].w,
                    w_gate[l].w, w_up[l].w, w_down[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    d_fn_out,  // reuse for d_x_in
                    w_qkv[l].g, w_o[l].g,
                    w_gate[l].g, w_up[l].g, w_down[l].g,
                    g_rn1[l], g_rn2[l],
                    B, S, D, H, d_ff);
                d_prev = d_fn_out;
            }

            // Embedding backward: accumulate into FP32 buffer
            embed_bwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
                d_prev, d_input_ids, d_embed_grad_f32, B, S, D, V);
            // Convert accumulated FP32 grads → FP16
            mus_convert_f32_to_f16(ctx, d_embed_grad_f32, w_embed.g, V*D);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // ═══ OPTIMIZER (FP16 Adam) ═══
            // Unscale gradients before optimizer
            float inv_scale = 1.0f / loss_scale;

            // Embedding
            mus_unscale_gradients_f16(ctx, w_embed.g, V*D, inv_scale);
            mus_clip_gradients_f16(ctx, w_embed.g, V*D, 0.5f);
            mus_adamw_step_f16(ctx, w_embed.w, w_embed.m, w_embed.v, w_embed.g, V*D,
                               current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

            for (int l = 0; l < L; l++) {
                mus_unscale_gradients_f16(ctx, w_qkv[l].g, D*3*D, inv_scale);
                mus_unscale_gradients_f16(ctx, w_o[l].g, D*D, inv_scale);
                mus_unscale_gradients_f16(ctx, w_gate[l].g, D*d_ff, inv_scale);
                mus_unscale_gradients_f16(ctx, w_up[l].g, D*d_ff, inv_scale);
                mus_unscale_gradients_f16(ctx, w_down[l].g, d_ff*D, inv_scale);

                mus_clip_gradients_f16(ctx, w_qkv[l].g, D*3*D, 0.5f);
                mus_clip_gradients_f16(ctx, w_o[l].g, D*D, 0.5f);
                mus_clip_gradients_f16(ctx, w_gate[l].g, D*d_ff, 0.5f);
                mus_clip_gradients_f16(ctx, w_up[l].g, D*d_ff, 0.5f);
                mus_clip_gradients_f16(ctx, w_down[l].g, d_ff*D, 0.5f);

                mus_adamw_step_f16(ctx, w_qkv[l].w, w_qkv[l].m, w_qkv[l].v, w_qkv[l].g, D*3*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step_f16(ctx, w_o[l].w, w_o[l].m, w_o[l].v, w_o[l].g, D*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step_f16(ctx, w_gate[l].w, w_gate[l].m, w_gate[l].v, w_gate[l].g, D*d_ff,
                                   current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step_f16(ctx, w_up[l].w, w_up[l].m, w_up[l].v, w_up[l].g, D*d_ff,
                                   current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step_f16(ctx, w_down[l].w, w_down[l].m, w_down[l].v, w_down[l].g, d_ff*D,
                                   current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
            }

            // RMSNorm FP32 Adam (all layer norms + final norm)
            // Unscale FP32 grads in-place on device
            mus_unscale_f32(ctx, fn_g, D, inv_scale);
            mus_adamw_step(ctx, fn_w, fn_m, fn_v, fn_g, D,
                           current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

            for (int l = 0; l < L; l++) {
                mus_unscale_f32(ctx, g_rn1[l], D, inv_scale);
                mus_unscale_f32(ctx, g_rn2[l], D, inv_scale);
                mus_adamw_step(ctx, w_rn1[l], rn1_m[l], rn1_v[l], g_rn1[l], D,
                               current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, w_rn2[l], rn2_m[l], rn2_v[l], g_rn2[l], D,
                               current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
            }

            // ═══ Loss Scale Adjustment (simple heuristic) ═══
            // Check if loss is NaN → reduce loss scale
            if (isnan(step_loss) || isinf(step_loss)) {
                loss_scale *= 0.5f;
                printf("\n  NaN detected! Reducing loss_scale to %.0f\n", loss_scale);
                if (loss_scale < 1.0f) {
                    printf("  Loss scale too low, aborting\n");
                    break;
                }
            } else if (global_step > warmup_steps && global_step % 100 == 0) {
                // Gradually increase loss scale if no NaN
                loss_scale = fminf(loss_scale * 1.01f, 65536.0f);
            }

            // Logging
            if (step < 5 || step % 50 == 0 || step == steps_per_epoch - 1) {
                double avg = total_loss / fmaxf(1.0f, (float)total_valid);
                printf("  epoch %2d/%d step %4d/%d  loss=%.4f  step=%.4f  valid=%d  lr=%.2e  ls=%.0f\n",
                       epoch+1, num_epochs, step, steps_per_epoch, avg, step_loss, valid, current_lr, loss_scale);
            }
        }

        float et = epoch_timer.ms() / 1000.0f;
        double avg = total_loss / fmaxf(1.0, (double)total_valid);
        if (avg < best_loss) best_loss = (float)avg;
        printf("  epoch %2d/%d  loss=%.4f  time=%.1fs  step=%d  best=%.4f\n",
               epoch+1, num_epochs, avg, et, global_step, best_loss);
    }

    printf("\nDone! Best loss: %.4f\n", best_loss);
    mus_destroy_context(ctx);
    return 0;
}
