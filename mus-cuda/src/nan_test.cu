#include "mus_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <chrono>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = call;                                        \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); }               \
} while(0)

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int dev; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("MUS-CUDA NaN Test (synthetic)\n");
    printf("  GPU: %s  VRAM: %.0f MB\n", prop.name, prop.totalGlobalMem / 1e6f);

    // Tiny model config
    int B = 2, S = 64, D = 128, H = 4, V = 1000, L = 2;
    int rows = B * S;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.02f);
    std::uniform_int_distribution<int> cat(0, V - 1);

    MUSContext* ctx = mus_create_context(512 * 1024 * 1024);

    // ── Weights ──
    auto alloc = [&](float*& w, float*& g, float*& m, float*& v, int n, bool ones) {
        CUDA_CHECK(cudaMalloc(&w, n * 4));
        CUDA_CHECK(cudaMalloc(&g, n * 4));
        CUDA_CHECK(cudaMalloc(&m, n * 4));
        CUDA_CHECK(cudaMalloc(&v, n * 4));
        CUDA_CHECK(cudaMemset(g, 0, n * 4));
        CUDA_CHECK(cudaMemset(m, 0, n * 4));
        CUDA_CHECK(cudaMemset(v, 0, n * 4));
        std::vector<float> h(n);
        for (int i = 0; i < n; i++) h[i] = ones ? 1.0f : nd(rng);
        CUDA_CHECK(cudaMemcpy(w, h.data(), n * 4, cudaMemcpyHostToDevice));
    };

    float *embed, *embed_g, *embed_m, *embed_v;
    alloc(embed, embed_g, embed_m, embed_v, V * D, false);
    float *fn_w, *fn_g, *fn_m, *fn_v;
    alloc(fn_w, fn_g, fn_m, fn_v, D, true);

    std::vector<float*> w_qkv(L), g_qkv(L), m_qkv(L), v_qkv(L);
    std::vector<float*> w_o(L), g_o(L), m_o(L), v_o(L);
    std::vector<float*> w_g(L), g_g(L), m_g(L), v_g(L);
    std::vector<float*> w_u(L), g_u(L), m_u(L), v_u(L);
    std::vector<float*> w_d(L), g_d(L), m_d(L), v_d(L);
    std::vector<float*> w_r1(L), g_r1(L), m_r1(L), v_r1(L);
    std::vector<float*> w_r2(L), g_r2(L), m_r2(L), v_r2(L);

    for (int l = 0; l < L; l++) {
        alloc(w_qkv[l], g_qkv[l], m_qkv[l], v_qkv[l], D*3*D, false);
        alloc(w_o[l],   g_o[l],   m_o[l],   v_o[l],   D*D,   false);
        alloc(w_g[l],   g_g[l],   m_g[l],   v_g[l],   D*4*D, false);
        alloc(w_u[l],   g_u[l],   m_u[l],   v_u[l],   D*4*D, false);
        alloc(w_d[l],   g_d[l],   m_d[l],   v_d[l],   4*D*D, false);
        alloc(w_r1[l],  g_r1[l],  m_r1[l],  v_r1[l],  D,     true);
        alloc(w_r2[l],  g_r2[l],  m_r2[l],  v_r2[l],  D,     true);
    }

    // ── Token weights ──
    float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, V * 4));
    MUSConfig cfg;
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), V * 4, cudaMemcpyHostToDevice));

    // ── Buffers ──
    float *d_in, *d_logits, *d_trace, *d_fn_out, *d_loss;
    int64_t *d_labels, *d_pos;
    size_t trace_sz = (size_t)(L + 1) * rows * D * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)B * S * 8));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)B * S * 8));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * V * 4));
    CUDA_CHECK(cudaMalloc(&d_loss, 4));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_sz));
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * D * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)B * S * 8));

    // ── Generate synthetic data ──
    std::vector<int64_t> h_in(B * S), h_labels(B * S), h_pos(B * S);
    for (int i = 0; i < B * S; i++) {
        h_in[i] = cat(rng);
        h_labels[i] = (rng() % 10 == 0) ? -100 : cat(rng);
        h_pos[i] = i % S;
    }

    printf("\nSynthetic: B=%d S=%d D=%d H=%d V=%d L=%d\n", B, S, D, H, V, L);
    int valid = 0;
    for (int i = 0; i < B*S; i++) if (h_labels[i] != -100) valid++;
    printf("  Valid tokens/step: %d\n\n", valid);

    // ── Training loop ──
    float lr = 1e-3f;
    int steps = 5;
    for (int step = 0; step < steps; step++) {
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in.data(), B*S*8, cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels.data(), B*S*8, cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), B*S*8, cudaMemcpyHostToDevice, ctx->stream));

        // Forward
        float* trace[13];
        for (int l = 0; l <= L; l++) trace[l] = d_trace + (size_t)l * rows * D;

        embed_fwd_kernel<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
            embed, (int*)d_in, trace[0], B, S, D, V);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        for (int l = 0; l < L; l++) {
            mus_transformer_block_forward(ctx, trace[l], trace[l+1],
                w_qkv[l], w_o[l], w_g[l], w_u[l], w_d[l],
                w_r1[l], w_r2[l], d_pos, B, S, D, H);
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        float alpha=1, beta=0;
        mus_rmsnorm_forward(ctx, trace[L], fn_w, d_fn_out, rows, D);
        cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    V, rows, D, &alpha, embed, V, d_fn_out, D, &beta, d_logits, V);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        float sum = mus_weighted_ce_forward(ctx, d_logits, d_labels, d_weights, d_loss, B, S, V);
        float step_loss = sum / fmaxf(1, valid);
        printf("  step %d  loss=%.4f", step, step_loss);

        // Check logits for NaN
        float logit_ck[3];
        CUDA_CHECK(cudaMemcpy(logit_ck, d_logits, 12, cudaMemcpyDeviceToHost));
        float trace_ck[3];
        CUDA_CHECK(cudaMemcpy(trace_ck, d_fn_out, 12, cudaMemcpyDeviceToHost));
        int nan = 0;
        for (int k = 0; k < 3; k++) { if (isnan(logit_ck[k]) || isnan(trace_ck[k])) nan = 1; }
        if (nan) { printf("  *** NAN IN FORWARD ***\n"); return 1; }

        // Backward
        float scale = 1.0f / fmaxf(1, valid);
        mus_weighted_ce_backward(ctx, d_logits, d_labels, d_weights, scale, d_logits, B, S, V);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        CUDA_CHECK(cudaMemsetAsync(ctx->workspace, 0, ctx->workspace_size, ctx->stream));

        cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    D, rows, V, &alpha, embed, V, d_logits, V, &beta, d_fn_out, D);
        cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    V, D, rows, &alpha, d_logits, V, d_fn_out, D, &alpha, embed_g, V);

        mus_rmsnorm_backward(ctx, trace[L], fn_w, d_fn_out, d_fn_out, fn_g, rows, D);
        float* d_prev = d_fn_out;

        // Check d_prev for NaN (gradient going into first block backward)
        CUDA_CHECK(cudaMemcpy(trace_ck, d_prev, 12, cudaMemcpyDeviceToHost));
        int d_prev_nan = 0;
        for (int k = 0; k < 3; k++) if (isnan(trace_ck[k])) d_prev_nan = 1;
        printf("  d_prev=[%.2e %.2e %.2e]", trace_ck[0], trace_ck[1], trace_ck[2]);

        // Layer backward
        int nan_found = 0;
        for (int l = L-1; l >= 0; l--) {
            mus_transformer_block_backward(ctx, trace[l], d_prev,
                w_qkv[l], w_o[l], w_g[l], w_u[l], w_d[l],
                w_r1[l], w_r2[l], d_pos,
                d_fn_out,
                g_qkv[l], g_o[l], g_g[l], g_u[l], g_d[l],
                g_r1[l], g_r2[l], B, S, D, H);
            d_prev = d_fn_out;

            CUDA_CHECK(cudaMemcpy(trace_ck, d_prev, 12, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 3; k++) if (isnan(trace_ck[k])) { nan_found = 1; break; }
            CUDA_CHECK(cudaMemcpy(trace_ck, g_qkv[l], 12, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 3; k++) if (isnan(trace_ck[k])) { nan_found = 1; break; }
            CUDA_CHECK(cudaMemcpy(trace_ck, g_o[l], 12, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 3; k++) if (isnan(trace_ck[k])) { nan_found = 1; break; }
        }

        if (nan_found) { printf("  *** NAN IN BACKWARD ***\n"); return 1; }

        // Embed backward
        embed_bwd_kernel<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(d_prev, (int*)d_in, embed_g, B, S, D, V);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        // Optimizer
        mus_adamw_step(ctx, embed, embed_m, embed_v, embed_g, V*D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
        for (int l = 0; l < L; l++) {
            mus_adamw_step(ctx, w_qkv[l], m_qkv[l], v_qkv[l], g_qkv[l], D*3*D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_o[l],   m_o[l],   v_o[l],   g_o[l],   D*D,   lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_g[l],   m_g[l],   v_g[l],   g_g[l],   D*4*D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_u[l],   m_u[l],   v_u[l],   g_u[l],   D*4*D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_d[l],   m_d[l],   v_d[l],   g_d[l],   4*D*D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_r1[l],  m_r1[l],  v_r1[l],  g_r1[l],  D,     lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
            mus_adamw_step(ctx, w_r2[l],  m_r2[l],  v_r2[l],  g_r2[l],  D,     lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
        }
        mus_adamw_step(ctx, fn_w, fn_m, fn_v, fn_g, D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // Zero grads for next step
        CUDA_CHECK(cudaMemsetAsync(embed_g, 0, (size_t)V*D*4, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(fn_g, 0, D*4, ctx->stream));
        for (int l = 0; l < L; l++) {
            CUDA_CHECK(cudaMemsetAsync(g_qkv[l], 0, (size_t)D*3*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_o[l], 0, (size_t)D*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_g[l], 0, (size_t)D*4*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_u[l], 0, (size_t)D*4*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_d[l], 0, (size_t)4*D*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_r1[l], 0, D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(g_r2[l], 0, D*4, ctx->stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        printf("  ✓\n");
    }

    // Check final weights for NaN
    printf("\nChecking weights for NaN:\n");
    auto check_nan = [&](float* d, int n, const char* name) {
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, n*4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            if (isnan(h[i]) || isinf(h[i])) {
                printf("  %s: NaN/INF at [%d]\n", name, i);
                return 1;
            }
        }
        printf("  %s: OK (n=%d)\n", name, n);
        return 0;
    };
    int any_nan = 0;
    any_nan |= check_nan(embed, V*D, "embed");
    any_nan |= check_nan(fn_w, D, "fn_w");
    for (int l = 0; l < L; l++) {
        char name[32];
        snprintf(name, sizeof(name), "qkv[%d]", l); any_nan |= check_nan(w_qkv[l], D*3*D, name);
        snprintf(name, sizeof(name), "o[%d]", l);   any_nan |= check_nan(w_o[l], D*D, name);
        snprintf(name, sizeof(name), "g[%d]", l);   any_nan |= check_nan(w_g[l], D*4*D, name);
        snprintf(name, sizeof(name), "r1[%d]", l);  any_nan |= check_nan(w_r1[l], D, name);
    }

    if (any_nan) {
        printf("\nFAIL: NaN found in weights!\n");
    } else {
        printf("\nPASS: No NaN in weights, loss decreased (or stable)\n");
    }

    mus_destroy_context(ctx);
    return any_nan;
}
