#include "mus_cuda.h"
#include <cuda_runtime.h>
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
//  Load binary cache
// ══════════════════════════════════════════════════════════════════════

struct PhotoData {
    std::vector<int> data;
    std::vector<int64_t> labels;
    std::vector<float> weights;
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

    // Optional: load precomputed labels (train_labels.bin parallel to train_cache.bin)
    std::string lb_path = path.substr(0, path.rfind('_')) + "_labels.bin";
    std::ifstream lb_f(lb_path, std::ios::binary);
    if (lb_f) {
        lb_f.seekg(0, std::ios::end);
        size_t lb_bytes = lb_f.tellg();
        lb_f.seekg(0);
        pd.labels.resize(lb_bytes / sizeof(int64_t));
        lb_f.read((char*)pd.labels.data(), lb_bytes);
        printf("  Loaded labels (%zu samples)\n", pd.labels.size() / pd.seq_len);
    }

    // Optional: load precomputed weights
    std::string w_path = path.substr(0, path.rfind('_')) + "_weights.bin";
    std::ifstream w_f(w_path, std::ios::binary);
    if (w_f) {
        w_f.seekg(0, std::ios::end);
        size_t w_bytes = w_f.tellg();
        w_f.seekg(0);
        pd.weights.resize(w_bytes / sizeof(float));
        w_f.read((char*)pd.weights.data(), w_bytes);
        printf("  Loaded weights (%zu samples)\n", pd.weights.size() / pd.seq_len);
    }

    return pd;
}

struct BatchData {
    std::vector<int> input;
    std::vector<int64_t> labels64;
};

BatchData make_batch(PhotoData& pd, int start, int B, int S) {
    BatchData bd;
    bd.input.resize(B * S);
    bd.labels64.resize(B * S);

    bool has_labels = pd.labels.size() == (size_t)pd.num_samples * S;
    bool has_weights = pd.weights.size() == (size_t)pd.num_samples * S;

    for (int b = 0; b < B; b++) {
        int idx = (start + b) % pd.num_samples;
        int* src = pd.data.data() + idx * S;
        for (int s = 0; s < S - 1; s++) {
            bd.input[b * S + s] = src[s];
            if (has_labels) {
                bd.labels64[b * S + s] = pd.labels[idx * S + s];
            } else {
                bd.labels64[b * S + s] = (int64_t)src[s + 1];
            }
        }
        if (has_labels) {
            bd.input[b * S + (S - 1)] = 0;
            bd.labels64[b * S + (S - 1)] = pd.labels[idx * S + (S - 1)];
        } else {
            bd.input[b * S + (S - 1)] = 0;
            bd.labels64[b * S + (S - 1)] = -100;
        }
    }

    return bd;
}

// ══════════════════════════════════════════════════════════════════════
//  Helper: allocate GPU weights with init
// ══════════════════════════════════════════════════════════════════════

void alloc_weights(float*& w, float*& g, float*& m, float*& v, int n,
                   std::mt19937& rng, float stddev, bool ones, bool is_output=false) {
    printf(" alloc_w..."); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&w, n * 4));
    printf("w "); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&g, n * 4));
    printf("g "); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&m, n * 4));
    printf("m "); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&v, n * 4));
    printf("v "); fflush(stdout);
    CUDA_CHECK(cudaMemset(g, 0, n * 4));
    printf("zg "); fflush(stdout);
    CUDA_CHECK(cudaMemset(m, 0, n * 4));
    printf("zm "); fflush(stdout);
    CUDA_CHECK(cudaMemset(v, 0, n * 4));
    printf("zv "); fflush(stdout);
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0, stddev);
    for (int i = 0; i < n; i++) {
        float val = ones ? 1.0f : nd(rng);
        if (is_output) val *= 0.2f; // Дополнительное ослабление для выходных проекций
        h[i] = val;
    }
    printf("gen "); fflush(stdout);
    CUDA_CHECK(cudaMemcpy(w, h.data(), n * 4, cudaMemcpyHostToDevice));
    printf("cpy "); fflush(stdout);
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
    printf("MUS-CUDA Full Training (12 layers)\n"); fflush(stdout);
    printf("  GPU: %s  VRAM: %.0f MB\n", prop.name, prop.totalGlobalMem / 1e6f); fflush(stdout);

    MUSConfig cfg;
    cfg.num_layers = 12;
    cfg.vocab_size = 10301;  // BPE vocab (AER 2001 + CPP 100 + Tags 200 + BPE 8000)

    printf("  Model: D=%d V=%d L=%d H=%d\n", cfg.hidden_dim, cfg.vocab_size, cfg.num_layers, cfg.num_heads); fflush(stdout);

    int B = 4, S = 512, D = cfg.hidden_dim, V = cfg.vocab_size, L = cfg.num_layers, H = cfg.num_heads;
    int rows = B * S;

    // ── Data ──
    std::string cache_path = "data/russian_bpe_train_cache.bin";
    if (argc > 1) cache_path = argv[1];
    PhotoData pd = load_cache(cache_path);
    int N = pd.num_samples;

    printf("  Data loaded: %d samples\n", N); fflush(stdout);

    // ── Context ──
    size_t ws_size = 1024 * 1024 * 1024;
    printf("  Creating context with %zu MB workspace...\n", ws_size/1024/1024); fflush(stdout);
    MUSContext* ctx = mus_create_context(ws_size);
printf("  Context created OK\n"); fflush(stdout);
    cudaDeviceSynchronize();
    printf("  Device sync OK\n"); fflush(stdout);

    // ── Embed ──
    float *d_embed, *d_embed_grad, *d_embed_m, *d_embed_v;
    {
        size_t n = (size_t)V * D;
        CUDA_CHECK(cudaMalloc(&d_embed, n * 4));
        CUDA_CHECK(cudaMalloc(&d_embed_grad, n * 4));
        CUDA_CHECK(cudaMalloc(&d_embed_m, n * 4));
        CUDA_CHECK(cudaMalloc(&d_embed_v, n * 4));
        CUDA_CHECK(cudaMemset(d_embed_m, 0, n * 4));
        CUDA_CHECK(cudaMemset(d_embed_v, 0, n * 4));
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0, 0.02f);
        std::vector<float> h(n);
        for (int d = 0; d < D; d++)
            for (int v = 0; v < V; v++)
                h[v + d * V] = nd(rng);
        CUDA_CHECK(cudaMemcpy(d_embed, h.data(), n * 4, cudaMemcpyHostToDevice));
    }
    printf("  Embedding allocated\n"); fflush(stdout);

    // Layer weights
    printf("  Allocating layer weights (L=%d)...\n", L); fflush(stdout);
    std::vector<float*> w_qkv(L), w_o(L), w_mlp_g(L), w_mlp_u(L), w_mlp_d(L), w_rn1(L), w_rn2(L);
    std::vector<float*> g_qkv(L), g_o(L), g_mlp_g(L), g_mlp_u(L), g_mlp_d(L), g_rn1(L), g_rn2(L);
    std::vector<float*> m_qkv(L), m_o(L), m_mlp_g(L), m_mlp_u(L), m_mlp_d(L), m_rn1(L), m_rn2(L);
    std::vector<float*> v_qkv(L), v_o(L), v_mlp_g(L), v_mlp_u(L), v_mlp_d(L), v_rn1(L), v_rn2(L);

    std::mt19937 w_rng(42);
    float std_w = 0.02f;
    for (int l = 0; l < L; l++) {
        printf("    Layer %d/%d...", l+1, L); fflush(stdout);
        // Allocate RMS norm weights as single block then split
        {
            size_t n = D * 2;
            float *tmp_w, *tmp_g, *tmp_m, *tmp_v;
            CUDA_CHECK(cudaMalloc(&tmp_w, n * 4));
            CUDA_CHECK(cudaMalloc(&tmp_g, n * 4));
            CUDA_CHECK(cudaMalloc(&tmp_m, n * 4));
            CUDA_CHECK(cudaMalloc(&tmp_v, n * 4));
            CUDA_CHECK(cudaMemset(tmp_g, 0, n * 4));
            CUDA_CHECK(cudaMemset(tmp_m, 0, n * 4));
            CUDA_CHECK(cudaMemset(tmp_v, 0, n * 4));
            std::vector<float> h(n, 1.0f);
            CUDA_CHECK(cudaMemcpy(tmp_w, h.data(), n * 4, cudaMemcpyHostToDevice));
            w_rn1[l] = tmp_w;
            w_rn2[l] = tmp_w + D;
            g_rn1[l] = tmp_g;
            g_rn2[l] = tmp_g + D;
            m_rn1[l] = tmp_m;
            m_rn2[l] = tmp_m + D;
            v_rn1[l] = tmp_v;
            v_rn2[l] = tmp_v + D;
        }
        printf("rn "); fflush(stdout);
        alloc_weights(w_qkv[l], g_qkv[l], m_qkv[l], v_qkv[l], D*3*D, w_rng, std_w, false);
        alloc_weights(w_o[l],   g_o[l],   m_o[l],   v_o[l],   D*D,   w_rng, std_w, false, true); // output projection
        alloc_weights(w_mlp_g[l], g_mlp_g[l], m_mlp_g[l], v_mlp_g[l], D*4*D, w_rng, std_w, false);
        alloc_weights(w_mlp_u[l], g_mlp_u[l], m_mlp_u[l], v_mlp_u[l], D*4*D, w_rng, std_w, false);
        alloc_weights(w_mlp_d[l], g_mlp_d[l], m_mlp_d[l], v_mlp_d[l], 4*D*D, w_rng, std_w, false, true); // output projection
        printf(" OK\n"); fflush(stdout);
    }

    // Final norm
    float *d_fn_w, *d_fn_grad, *d_fn_m, *d_fn_v;
    {
        CUDA_CHECK(cudaMalloc(&d_fn_w, D * 4));
        CUDA_CHECK(cudaMalloc(&d_fn_grad, D * 4));
        CUDA_CHECK(cudaMalloc(&d_fn_m, D * 4));
        CUDA_CHECK(cudaMalloc(&d_fn_v, D * 4));
        CUDA_CHECK(cudaMemset(d_fn_grad, 0, D * 4));
        CUDA_CHECK(cudaMemset(d_fn_m, 0, D * 4));
        CUDA_CHECK(cudaMemset(d_fn_v, 0, D * 4));
        std::vector<float> h(D, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_fn_w, h.data(), D * 4, cudaMemcpyHostToDevice));
    }

    // Token weights
    float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, V * 4));
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), V * 4, cudaMemcpyHostToDevice));

    // Training buffers
    float *d_input_ids, *d_logits, *d_trace, *d_fn_out, *d_loss;
    int64_t *d_pos;
    int64_t *d_labels64;
    size_t trace_sz = (size_t)(L + 1) * rows * D * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input_ids, (size_t)B * S * 4));
    CUDA_CHECK(cudaMalloc(&d_labels64, (size_t)B * S * 8));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * V * 4));
    CUDA_CHECK(cudaMalloc(&d_loss, 4));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_sz));
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * D * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)B * S * 8));

    // Host batch buffers
    std::vector<int64_t> h_pos(B * S);
    for (int i = 0; i < B * S; i++) h_pos[i] = (int64_t)(i % S);

    // Training loop
    int num_epochs = 100, steps_per_epoch = N / B, global_step = 0;
    float base_lr = 5e-5f;
    int warmup_steps = 200;
    float current_lr = (global_step < warmup_steps) ? base_lr * ((float)global_step / warmup_steps) : base_lr;

    printf("\nStarting: %d samples, B=%d, %d steps/epoch, %d epochs\n", N, B, steps_per_epoch, num_epochs); fflush(stdout);
    printf("  Total params: %.1fM\n\n", (double)(V*D + L*(D*3*D + D*D + 2*D*4*D + 4*D*D + 2*D) + D)/1e6); fflush(stdout);

    float best_loss = 1e10f;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Timer epoch_timer;
        float total_loss = 0.0f;
        int total_valid = 0;

        printf("Starting training loop...\n"); fflush(stdout);
        for (int step = 0; step < steps_per_epoch; step++) {
            printf("Step %d/%d...\n", step, steps_per_epoch); fflush(stdout);
            int start = step * B;

            // Prepare batch
            BatchData bd = make_batch(pd, start, B, S);

            float* trace[13];
            for (int l = 0; l <= L; l++)
                trace[l] = d_trace + (size_t)l * rows * D;

            CUDA_CHECK(cudaMemcpyAsync(d_input_ids, bd.input.data(), B*S*4, cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels64, bd.labels64.data(), B*S*8, cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), B*S*8, cudaMemcpyHostToDevice, ctx->stream));

            // Embed
            embed_fwd_kernel<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(d_embed, (int*)d_input_ids, trace[0], B, S, D, V);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // Transformer layers
            for (int l = 0; l < L; l++) {
                mus_transformer_block_forward(ctx, trace[l], trace[l+1],
                    w_qkv[l], w_o[l], w_mlp_g[l], w_mlp_u[l], w_mlp_d[l],
                    w_rn1[l], w_rn2[l], d_pos, B, S, D, H);
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // Final norm + LM Head
            float alpha=1, beta=0;
            mus_rmsnorm_forward(ctx, trace[L], d_fn_w, d_fn_out, rows, D);
            cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        V, rows, D, &alpha, d_embed, V, d_fn_out, D, &beta, d_logits, V);

            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            int valid = 0;
            for (int i = 0; i < B*S; i++) if (bd.labels64[i] != -100) valid++;
            float sum = mus_weighted_ce_forward(ctx, d_logits, d_labels64, d_weights, d_loss, B, S, V);
            float loss_val = sum / fmaxf(1, valid);
            float step_avg = sum / fmaxf(1, valid);
            total_loss += sum;
            total_valid += valid;
            global_step++;
            current_lr = (global_step < warmup_steps) ? base_lr * ((float)global_step / warmup_steps) : base_lr;

            // ═══ BACKWARD ═══
            float scale = 1.0f / fmaxf(1, valid);
            mus_weighted_ce_backward(ctx, d_logits, d_labels64, d_weights, scale, d_logits, B, S, V);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // Zero workspace to eliminate garbage → NaN risk
            CUDA_CHECK(cudaMemsetAsync(ctx->workspace, 0, ctx->workspace_size, ctx->stream));

            // d_embed_grad = d_logits_grad @ fn_out^T  [V,rows] × [D,rows]^T → [V,D]

            // d_fn_out_grad = embed^T @ d_logits_grad  [D,V] × [V,rows] → [D,rows]
            cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        D, rows, V, &alpha, d_embed, V, d_logits, V, &beta, d_fn_out, D);
            // d_embed_grad += d_logits_grad @ fn_out^T  [V,rows] × [D,rows]^T → [V,D]
            // Must happen before rmsnorm backward overwrites d_fn_out
            cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        V, D, rows, &alpha, d_logits, V, d_fn_out, D, &alpha, d_embed_grad, V);

            // Final norm backward
            mus_rmsnorm_backward(ctx, trace[L], d_fn_w, d_fn_out, d_fn_out, d_fn_grad, rows, D);
            float* d_prev = d_fn_out;

            // Check d_prev (gradient into layers) on first step
            if (step == 0) {
                float ck[4];
                CUDA_CHECK(cudaMemcpy(ck, d_prev, 16, cudaMemcpyDeviceToHost));
                printf("  d_prev[0..3]=[%.2e %.2e %.2e %.2e]\n", ck[0], ck[1], ck[2], ck[3]);
            }

            // Transformer layers backward (reverse)
            for (int l = L-1; l >= 0; l--) {
                mus_clip_gradients(ctx, d_prev, rows * D, 0.5f);
                mus_transformer_block_backward(ctx, trace[l], d_prev,
                    w_qkv[l], w_o[l], w_mlp_g[l], w_mlp_u[l], w_mlp_d[l],
                    w_rn1[l], w_rn2[l], d_pos,
                    d_fn_out,   // reuse d_fn_out for d_x_in
                    g_qkv[l], g_o[l], g_mlp_g[l], g_mlp_u[l], g_mlp_d[l],
                    g_rn1[l], g_rn2[l], B, S, D, H);
                d_prev = d_fn_out;
                // Per-layer NaN check on first step
                if (step == 0) {
                    float ck[4]; int nan;
                    CUDA_CHECK(cudaMemcpy(ck, d_fn_out, 16, cudaMemcpyDeviceToHost));
                    nan = 0; for (int k = 0; k < 4; k++) if (isnan(ck[k])||isinf(ck[k])) nan = 1;
                    if (nan) { printf("  NAN at layer %d BACKWARD (d_fn_out)\n", l); break; }
                    CUDA_CHECK(cudaMemcpy(ck, g_qkv[l], 16, cudaMemcpyDeviceToHost));
                    nan = 0; for (int k = 0; k < 4; k++) if (isnan(ck[k])||isinf(ck[k])) nan = 1;
                    if (nan) { printf("  NAN at layer %d BACKWARD (g_qkv[%d])\n", l, l); break; }
                }
            }
            // Per-layer NaN check on first step
            if (step == 0) {
                for (int l = L-1; l >= 0; l--) {
                    float ck[4];
                    CUDA_CHECK(cudaMemcpy(ck, g_qkv[l], 16, cudaMemcpyDeviceToHost));
                    int nan = 0;
                    for (int k = 0; k < 4; k++) if (isnan(ck[k]) || isinf(ck[k])) nan = 1;
                    if (nan) printf("  NAN appears at layer %d backward (g_qkv)\n", l);
                    CUDA_CHECK(cudaMemcpy(ck, d_fn_out, 16, cudaMemcpyDeviceToHost));
                    nan = 0;
                    for (int k = 0; k < 4; k++) if (isnan(ck[k]) || isinf(ck[k])) nan = 1;
                    if (nan) printf("  NAN appears at layer %d backward (d_fn_out)\n", l);
                }
            }

            // Embedding backward
            embed_bwd_kernel<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(d_prev, (int*)d_input_ids, d_embed_grad, B, S, D, V);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            // ═══ OPTIMIZER ═══
            mus_clip_gradients(ctx, d_embed_grad, V*D, 0.5f);
            mus_adamw_step(ctx, d_embed, d_embed_m, d_embed_v, d_embed_grad, V*D, current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
            for (int l = 0; l < L; l++) {
mus_clip_gradients(ctx, g_qkv[l], D*3*D, 0.5f);
mus_clip_gradients(ctx, g_o[l], D*D, 0.5f);
mus_clip_gradients(ctx, g_mlp_g[l], D*4*D, 0.5f);
mus_clip_gradients(ctx, g_mlp_u[l], D*4*D, 0.5f);
mus_clip_gradients(ctx, g_mlp_d[l], 4*D*D, 0.5f);
mus_clip_gradients(ctx, g_rn1[l], D, 0.5f);
mus_clip_gradients(ctx, g_rn2[l], D, 0.5f);
                mus_adamw_step(ctx, w_rn2[l], m_rn2[l], v_rn2[l], g_rn2[l], D, current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
            }
            mus_clip_gradients(ctx, d_fn_grad, D, 1.0f);
            mus_adamw_step(ctx, d_fn_w, d_fn_m, d_fn_v, d_fn_grad, D, current_lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

            // Zero grads
            CUDA_CHECK(cudaMemsetAsync(d_embed_grad, 0, (size_t)V*D*4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(d_fn_grad, 0, D*4, ctx->stream));
            for (int l = 0; l < L; l++) {
                CUDA_CHECK(cudaMemsetAsync(g_qkv[l], 0, (size_t)D*3*D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_o[l], 0, (size_t)D*D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_mlp_g[l], 0, (size_t)D*4*D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_mlp_u[l], 0, (size_t)D*4*D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_mlp_d[l], 0, (size_t)4*D*D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn1[l], 0, D*4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn2[l], 0, D*4, ctx->stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            if (step < 10 || step % 50 == 0 || step == steps_per_epoch - 1) {
                float avg = total_loss / fmaxf(1.0f, (float)total_valid);
                printf("  epoch %2d/%d step %4d/%d  loss=%.4f  step=%.4f  valid=%d  lr=%.2e  |\r",
                       epoch+1, num_epochs, step, steps_per_epoch, avg, step_avg, valid, current_lr);
                fflush(stdout);
            }
        }

        float et = epoch_timer.ms() / 1000.0f;
        float avg = total_loss / fmaxf(1.0f, (float)total_valid);
        if (avg < best_loss) best_loss = avg;
        printf("  epoch %2d/%d  loss=%.4f  time=%.1fs  step=%d  best=%.4f\n",
               epoch+1, num_epochs, avg, et, global_step, best_loss);

        if ((epoch+1) % 10 == 0) {
            std::vector<float> save(V*D);
            CUDA_CHECK(cudaMemcpy(save.data(), d_embed, V*D*4, cudaMemcpyDeviceToHost));
            char fn[256];
            snprintf(fn, sizeof(fn), "checkpoints/cuda_epoch_%d.bin", epoch+1);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite(save.data(), 4, V*D, f); fclose(f); printf("  Saved %s\n", fn); }
        }
    }

    printf("\nDone! Best loss: %.4f\n", best_loss);
    mus_destroy_context(ctx);
    return 0;
}
