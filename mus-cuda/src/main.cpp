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

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point s;
    Timer() : s(clock::now()) {}
    float ms() { return std::chrono::duration<float,std::milli>(clock::now()-s).count(); }
    void reset() { s = clock::now(); }
};

// ─── CPU reference: weighted CE ──────────────────────────────────────
float cpu_weighted_ce(const float* logits, const int64_t* labels,
                      const float* weights, int B, int S, int V) {
    float total = 0; int valid = 0;
    for (int p = 0; p < B * S; p++) {
        int64_t lab = labels[p];
        if (lab == -100) continue;
        valid++;
        const float* pos = logits + p * V;
        float mx = -FLT_MAX;
        for (int i = 0; i < V; i++) mx = fmaxf(mx, pos[i]);
        float den = 0;
        for (int i = 0; i < V; i++) den += expf(pos[i] - mx);
        float sm = expf(pos[lab] - mx) / den;
        total += -logf(fmaxf(sm, 1e-30f)) * weights[lab];
    }
    return total / fmaxf(1, valid);
}

// ─── Demo 1: Weighted CE ────────────────────────────────────────────
void demo_ce(MUSContext* ctx, const MUSConfig& cfg) {
    printf("\n─── Demo 1: Weighted Cross-Entropy ───\n");
    int B = 4, S = 512, V = cfg.vocab_size;
    int np = B * S;
    std::vector<float> h_logits(np * V);
    std::vector<int64_t> h_labels(np);
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.5f);
    std::uniform_int_distribution<int> ud(0, V - 1);
    for (int i = 0; i < np * V; i++) h_logits[i] = nd(rng);
    for (int i = 0; i < np; i++) h_labels[i] = (i % 50 == 0) ? -100 : ud(rng);

    float *dL, *dW, *dLoss;
    int64_t* dLab;
    CUDA_CHECK(cudaMalloc(&dL, np * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLab, np * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dW, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLoss, 4));
    CUDA_CHECK(cudaMemcpy(dL, h_logits.data(), np * V * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLab, h_labels.data(), np * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, h_weights.data(), V * sizeof(float), cudaMemcpyHostToDevice));

    // GPU
    int warm = 10, it = 100;
    for (int i = 0; i < warm; i++)
        mus_weighted_ce_forward(ctx, dL, dLab, dW, dLoss, B, S, V);
    cudaStreamSynchronize(ctx->stream);
    Timer t;
    float gpu_loss = 0;
    for (int i = 0; i < it; i++)
        gpu_loss = mus_weighted_ce_forward(ctx, dL, dLab, dW, dLoss, B, S, V);
    float gpu_t = t.ms() / it;

    // CPU
    t.reset();
    float cpu_loss = 0;
    for (int i = 0; i < it; i++)
        cpu_loss = cpu_weighted_ce(h_logits.data(), h_labels.data(), h_weights.data(), B, S, V);
    float cpu_t = t.ms() / it;

    int valid = 0;
    for (int i = 0; i < np; i++) if (h_labels[i] != -100) valid++;

    printf("  GPU:  loss=%.6f  %.3f ms\n", gpu_loss / valid, gpu_t);
    printf("  CPU:  loss=%.6f  %.3f ms\n", cpu_loss, cpu_t);
    printf("  Diff: %.2e  %s\n", fabsf(gpu_loss/valid - cpu_loss),
           fabsf(gpu_loss/valid - cpu_loss) < 1e-4f ? "PASS" : "FAIL");
    printf("  Speedup: %.1fx\n", cpu_t / fmaxf(0.001f, gpu_t));

    // Backward
    float *dGrad;
    CUDA_CHECK(cudaMalloc(&dGrad, np * V * sizeof(float)));
    t.reset();
    for (int i = 0; i < it; i++)
        mus_weighted_ce_backward(ctx, dL, dLab, dW, 1.0f/valid, dGrad, B, S, V);
    cudaStreamSynchronize(ctx->stream);
    printf("  Backward: %.3f ms\n", t.ms() / it);

    CUDA_CHECK(cudaFree(dL)); CUDA_CHECK(cudaFree(dLab));
    CUDA_CHECK(cudaFree(dW)); CUDA_CHECK(cudaFree(dLoss));
    CUDA_CHECK(cudaFree(dGrad));
}

// ─── Demo 2: AdamW ──────────────────────────────────────────────────
void demo_adamw(MUSContext* ctx) {
    printf("\n─── Demo 2: Fused AdamW ───\n");
    int N = 10 * 1000 * 1000;
    float *p, *m, *v, *g;
    CUDA_CHECK(cudaMalloc(&p, N * 4));
    CUDA_CHECK(cudaMalloc(&m, N * 4));
    CUDA_CHECK(cudaMalloc(&v, N * 4));
    CUDA_CHECK(cudaMalloc(&g, N * 4));
    CUDA_CHECK(cudaMemset(m, 0, N * 4));
    CUDA_CHECK(cudaMemset(v, 0, N * 4));

    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0, 0.01f);
    std::vector<float> hp(N), hg(N);
    for (int i = 0; i < N; i++) { hp[i] = nd(rng); hg[i] = nd(rng); }
    CUDA_CHECK(cudaMemcpy(p, hp.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g, hg.data(), N * 4, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++)
        mus_adamw_step(ctx, p, m, v, g, N, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 1);
    cudaStreamSynchronize(ctx->stream);
    Timer t;
    for (int i = 0; i < 100; i++)
        mus_adamw_step(ctx, p, m, v, g, N, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, 100 + i);
    cudaStreamSynchronize(ctx->stream);
    float tm = t.ms() / 100;
    printf("  10M params: %.3f ms  (%.0f M/s)\n", tm, N / tm / 1000);

    CUDA_CHECK(cudaFree(p)); CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaFree(v)); CUDA_CHECK(cudaFree(g));
}

// ─── Demo 3: RMSNorm ────────────────────────────────────────────────
void demo_rmsnorm(MUSContext* ctx) {
    printf("\n─── Demo 3: RMSNorm ───\n");
    int rows = 4 * 512, cols = 768;
    float *x, *w, *y;
    CUDA_CHECK(cudaMalloc(&x, rows * cols * 4));
    CUDA_CHECK(cudaMalloc(&w, cols * 4));
    CUDA_CHECK(cudaMalloc(&y, rows * cols * 4));
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> hx(rows * cols), hw(cols, 1);
    for (int i = 0; i < rows * cols; i++) hx[i] = nd(rng);
    CUDA_CHECK(cudaMemcpy(x, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w, hw.data(), cols * 4, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) mus_rmsnorm_forward(ctx, x, w, y, rows, cols);
    cudaStreamSynchronize(ctx->stream);
    Timer t;
    for (int i = 0; i < 1000; i++) mus_rmsnorm_forward(ctx, x, w, y, rows, cols);
    cudaStreamSynchronize(ctx->stream);
    float tm = t.ms() / 1000;
    float bw = (float)rows * cols * 4 * 2 / tm / 1e6f;
    printf("  (%d x %d): %.3f ms  (%.0f GB/s)\n", rows, cols, tm, bw);
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(w)); CUDA_CHECK(cudaFree(y));
}

// ─── Demo 4: RoPE ───────────────────────────────────────────────────
void demo_rope(MUSContext* ctx) {
    printf("\n─── Demo 4: RoPE ───\n");
    int B = 4, H = 12, S = 512, D = 64;
    float *q, *k;
    int64_t *pos;
    CUDA_CHECK(cudaMalloc(&q, B*H*S*D * 4));
    CUDA_CHECK(cudaMalloc(&k, B*H*S*D * 4));
    CUDA_CHECK(cudaMalloc(&pos, B*S * 8));
    std::vector<int64_t> hp(B*S);
    for (int b = 0; b < B; b++) for (int s = 0; s < S; s++) hp[b*S+s] = s;
    CUDA_CHECK(cudaMemcpy(pos, hp.data(), B*S*8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(q, 0, B*H*S*D*4));
    CUDA_CHECK(cudaMemset(k, 0, B*H*S*D*4));

    for (int i = 0; i < 10; i++) mus_rope_forward(ctx, q, k, pos, B, H, S, D);
    cudaStreamSynchronize(ctx->stream);
    Timer t;
    for (int i = 0; i < 1000; i++) mus_rope_forward(ctx, q, k, pos, B, H, S, D);
    cudaStreamSynchronize(ctx->stream);
    printf("  (B=%d H=%d S=%d D=%d): %.3f ms\n", B, H, S, D, t.ms() / 1000);
    CUDA_CHECK(cudaFree(q)); CUDA_CHECK(cudaFree(k)); CUDA_CHECK(cudaFree(pos));
}

// ─── Demo 5: SwiGLU MLP ─────────────────────────────────────────────
void demo_swiglu(MUSContext* ctx) {
    printf("\n─── Demo 5: SwiGLU MLP ───\n");
    int rows = 4 * 512, dim = 768, cols4 = dim * 4;
    float *x, *gw, *uw, *dw, *out;
    CUDA_CHECK(cudaMalloc(&x, rows * dim * 4));
    CUDA_CHECK(cudaMalloc(&gw, dim * cols4 * 4));
    CUDA_CHECK(cudaMalloc(&uw, dim * cols4 * 4));
    CUDA_CHECK(cudaMalloc(&dw, cols4 * dim * 4));
    CUDA_CHECK(cudaMalloc(&out, rows * dim * 4));

    std::mt19937 rng(456);
    std::normal_distribution<float> nd(0, 0.1f);
    std::vector<float> hx(rows * dim);
    for (int i = 0; i < rows * dim; i++) hx[i] = nd(rng);
    CUDA_CHECK(cudaMemcpy(x, hx.data(), rows * dim * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gw, 0, dim * cols4 * 4));
    CUDA_CHECK(cudaMemset(uw, 0, dim * cols4 * 4));
    CUDA_CHECK(cudaMemset(dw, 0, cols4 * dim * 4));
    // Init weights (simple identity-ish for testing)
    // Actually just use random init
    std::vector<float> hw(dim * cols4);
    for (int i = 0; i < dim * cols4; i++) hw[i] = nd(rng);
    CUDA_CHECK(cudaMemcpy(gw, hw.data(), dim * cols4 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(uw, hw.data(), dim * cols4 * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < cols4 * dim; i++) hw[i] = nd(rng);
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), cols4 * dim * 4, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++)
        mus_swiglu_forward(ctx, x, gw, uw, dw, out, rows, dim);
    cudaStreamSynchronize(ctx->stream);
    Timer t;
    for (int i = 0; i < 100; i++)
        mus_swiglu_forward(ctx, x, gw, uw, dw, out, rows, dim);
    cudaStreamSynchronize(ctx->stream);
    float tm = t.ms() / 100;
    // FLOPs: 2 * (rows * dim * 4*dim + rows * 4*dim + rows * 4*dim * dim) ≈ 4 * rows * dim * 4*dim
    double flops = 4.0 * rows * dim * cols4; // rough estimate
    printf("  (%d x %d): %.3f ms  (%.0f GFLOP/s)\n", rows, dim, tm, flops / tm / 1e6f);
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(gw));
    CUDA_CHECK(cudaFree(uw)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(out));
}

// ─── Demo 6: Full training step estimate ─────────────────────────────
void demo_training_estimate(const MUSConfig& cfg) {
    printf("\n─── Demo 6: Training Step Profile ───\n");
    int B = 4, S = 512, D = cfg.hidden_dim, H = cfg.num_heads, L = cfg.num_layers, V = cfg.vocab_size;
    int rows = B * S;

    // Memory read/write per step (GB)
    double params_mem = (double)(L * (3 * D * 4 * D + D * 4 * D * 2 + 2 * D + D * V) + D * V) * 4;
    double act_mem = (double)(rows * (L * D * 4 + D + V)) * 4;
    double total_bytes = (params_mem * 2 + act_mem * 2) / 1e9; // read + write

    // FLOPs per step (rough)
    double flops_attn = (double)L * 2 * rows * D * 3 * D;  // QKV proj
    flops_attn += (double)L * 2 * rows * S * D;            // attn scores (simplified)
    double flops_mlp = (double)L * 4 * rows * D * 4 * D;   // SwiGLU
    double flops_embed = (double)rows * D;                  // embedding (negligible)
    double flops_head = (double)rows * D * V;               // lm_head
    double total_flops = (flops_attn + flops_mlp + flops_head) * 2; // fwd + bwd

    printf("  B=%d S=%d D=%d H=%d L=%d V=%d\n", B, S, D, H, L, V);
    printf("  Memory per step: ~%.1f GB (read+write)\n", total_bytes);
    printf("  FLOPs per step:  ~%.0f GFLOP\n", total_flops / 1e9);
    printf("  At 200 GB/s:     ~%.0f ms memory-bound\n", total_bytes / 0.2);
    printf("  At 10 TFLOP/s:   ~%.0f ms compute-bound\n", total_flops / 1e13);
}

// ─── Main ───────────────────────────────────────────────────────────
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int dev; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("╔══════════════════════════════════════════╗\n");
    printf("║     Uragan 1.0 — C++ CUDA Training      ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  GPU: %-36s ║\n", prop.name);
    printf("║  SM:  %d.%d  |  Mem: %.1f GB             ║\n",
           prop.major, prop.minor, (float)prop.totalGlobalMem/1e9f);
    printf("╚══════════════════════════════════════════╝\n");

    MUSConfig cfg;
    MUSContext* ctx = mus_create_context(128 * 1024 * 1024); // 128 MB workspace

    demo_ce(ctx, cfg);
    demo_adamw(ctx);
    demo_rmsnorm(ctx);
    demo_rope(ctx);
    demo_swiglu(ctx);
    demo_training_estimate(cfg);

    mus_destroy_context(ctx);
    printf("\n✓ All demos complete.\n");
    return 0;
}
