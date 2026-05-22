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
    printf("  Loaded %d samples x %d tokens (%zu MB)\n", td.num_samples, td.seq_len, bytes/1024/1024);
    return td;
}

__global__ void init_weights_kernel(half* w, int n, float stddev, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int rng = seed + i * 2654435761u;
    rng = (rng * 1103515245u + 12345u) & 0x7fffffff;
    float u1 = (float)rng / 2147483648.0f;
    rng = (rng * 1103515245u + 12345u) & 0x7fffffff;
    float u2 = (float)rng / 2147483648.0f;
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1e-10f)));
    float theta = 6.2831853f * u2;
    w[i] = __float2half(r * cosf(theta) * stddev);
}

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
    int bs = 256, grid = (n + bs - 1) / bs;
    int seed = (int)(stddev * 10000) ^ (int)(n);
    init_weights_kernel<<<grid, bs, 0, 0>>>(buf.w, n, stddev, seed);
    CUDA_CHECK(cudaGetLastError());
    return buf;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Forward-only diagnostic test\n  GPU: %s\n\n", prop.name);

    MUSConfig cfg;
    cfg = get_500m_gtx1660ti_config();
    int D = cfg.get_D(), V = cfg.get_V(), L = cfg.get_L();
    int H = cfg.get_H(), d_ff = cfg.get_D_ff();
    int B = 1, S = 256;
    int rows = B * S;
    printf("  Architecture: D=%d L=%d H=%d FF=%d V=%d\n", D, L, H, d_ff, V);

    TrainData td = load_binary_cache(argc > 1 ? argv[1] : "data/train_cache.bin");
    int N = td.num_samples;

    std::mt19937 rng(42);
    size_t ws_size = 64 * 1024 * 1024;
    MUSContext* ctx = mus_create_context(ws_size);
    printf("  Context created\n");

    // Allocate model weights
    printf("  Allocating model weights...\n");
    WeightBuf w_embed = alloc_f16(V * D, rng, 0.02f);

    std::vector<WeightBuf> w_qkv(L), w_o(L), w_gate(L), w_up(L), w_down(L);
    std::vector<float*> w_rn1(L), w_rn2(L);

    auto alloc_f32 = [&](int n, float val=1.0f) {
        float* d;
        CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(float)));
        std::vector<float> h(n, val);
        CUDA_CHECK(cudaMemcpy(d, h.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };
    for (int l = 0; l < L; l++) {
        w_rn1[l] = alloc_f32(D, 1.0f);
        w_rn2[l] = alloc_f32(D, 1.0f);
        w_qkv[l]  = alloc_f16(D * 3 * D,   rng, 0.02f);
        w_o[l]    = alloc_f16(D * D,       rng, 0.01f);
        w_gate[l] = alloc_f16(D * d_ff,    rng, 0.02f);
        w_up[l]   = alloc_f16(D * d_ff,    rng, 0.02f);
        w_down[l] = alloc_f16(d_ff * D,    rng, 0.01f);
    }
    float *fn_w = alloc_f32(D, 1.0f);

    // Token weights
    float *d_weights, *d_weights_ones;
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_ones, (size_t)V * sizeof(float)));
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                          (size_t)V * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> h_ones(V, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_weights_ones, h_ones.data(),
                          (size_t)V * sizeof(float), cudaMemcpyHostToDevice));

    // Buffers
    std::vector<int> h_input((size_t)B * S);
    std::vector<int64_t> h_labels64((size_t)B * S);
    std::vector<int64_t> h_pos((size_t)B * S);
    for (int i = 0; i < B * S; i++) h_pos[i] = (int64_t)(i % S);

    int *d_input_ids;
    int64_t *d_labels64, *d_pos;
    half *d_logits, *d_trace, *d_fn_out;
    float *d_loss;

    int ckpt = cfg.checkpoint_interval;
    int num_ckpt = L / ckpt + 2;
    size_t trace_ckpt_sz = (size_t)num_ckpt * rows * D * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_input_ids, (size_t)B * S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels64, (size_t)B * S * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * V * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_ckpt_sz));
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)B * S * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    // Weight categories:
    //   [0..cpp_tokens_start)              -> weight 15 (aerospace)
    //   [cpp_tokens_start..cpp_tokens_end]  -> weight 5  (C++ palette)
    //   [multimodal_tags_start..multimodal_tags_end] -> weight 15 (aerospace multimodal)
    //   rest                                 -> weight 1  (text)
    printf("  Token ranges: aer=[0..%d]∪[%d..%d] cpp=[%d..%d] text=rest\n",
           cfg.cpp_tokens_start - 1,
           cfg.multimodal_tags_start, cfg.multimodal_tags_end,
           cfg.cpp_tokens_start, cfg.cpp_tokens_end);

    // ─── Forward-only loop ───────────────────────────────
    int num_steps = 50;
    printf("\n  Forward-only test (%d samples):\n", num_steps);
    printf("  %5s %8s %10s %8s %6s %6s %6s\n","Step","WtLoss","UnwtLoss","Acc","nAer","nCpp","nText");

    for (int step = 0; step < num_steps; step++) {
        int start_idx = step * B;
        int na = 0, nc = 0, nw = 0;
        for (int b = 0; b < B; b++) {
            int idx = (start_idx + b) % N;
            int* src = td.data.data() + idx * S;
            for (int s = 0; s < S - 1; s++) {
                h_input[b * S + s] = src[s];
                int64_t lbl = (int64_t)src[s + 1];
                h_labels64[b * S + s] = lbl;
                if (lbl >= 0 && lbl < cfg.cpp_tokens_start) na++;
                else if (lbl >= cfg.cpp_tokens_start && lbl <= cfg.cpp_tokens_end) nc++;
                else if (lbl >= cfg.multimodal_tags_start && lbl <= cfg.multimodal_tags_end) na++;
                else nw++;
            }
            h_input[b * S + (S - 1)] = 0;
            h_labels64[b * S + (S - 1)] = -100;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_input_ids, h_input.data(), (size_t)B*S*sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_labels64, h_labels64.data(), (size_t)B*S*sizeof(int64_t), cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), (size_t)B*S*sizeof(int64_t), cudaMemcpyHostToDevice, ctx->stream));

        auto ckpt_slot = [&](int layer) -> half* {
            int slot = layer / ckpt;
            return d_trace + (size_t)slot * rows * D;
        };
        auto last_slot = [&]() -> half* {
            return d_trace + (size_t)(num_ckpt - 1) * rows * D;
        };

        // Forward
        embed_fwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(
            w_embed.w, d_input_ids, ckpt_slot(0), B, S, D, V);

        half* prev = ckpt_slot(0);
        for (int l = 0; l < L; l++) {
            half* curr;
            if (l == L - 1) {
                curr = last_slot();
            } else if (l % ckpt == ckpt - 1) {
                curr = ckpt_slot(l + 1);
            } else {
                curr = ctx->workspace_f16;
                curr += (size_t)(l % ckpt) * rows * D;
            }
            mus_transformer_block_forward_f16(ctx, prev, curr,
                w_qkv[l].w, w_o[l].w, w_gate[l].w, w_up[l].w, w_down[l].w,
                w_rn1[l], w_rn2[l], d_pos, B, S, D, H, d_ff);
            prev = curr;
        }

        mus_rmsnorm_forward_f16(ctx, prev, fn_w, d_fn_out, rows, D);
        mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                     V, rows, D, w_embed.w, V, d_fn_out, rows, d_logits, V);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        int valid = B * S - 1;
        float loss_val = mus_weighted_ce_forward_f16(ctx, d_logits, d_labels64, d_weights, d_loss, B, S, V);
        float step_loss = loss_val / fmaxf(1, valid);

        // Unweighted loss with all-ones weights
        float unweighted = mus_weighted_ce_forward_f16(ctx, d_logits, d_labels64, d_weights_ones, d_loss, B, S, V);
        float uw_loss = unweighted / fmaxf(1, valid);
        float acc = expf(-uw_loss);

        printf("  %5d %8.2f %10.4f %7.4f%% %6d %6d %6d\n", step, step_loss, uw_loss, acc*100.0f, na, nc, nw);
    }

    printf("\n  Done.\n");
    mus_destroy_context(ctx);
    return 0;
}
