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

// ─── Weight Allocator ──────────────────────────────────────────────────────
struct WeightBuf {
    float *w, *g, *m, *v;
    int n;
};

WeightBuf alloc_weights(int n, std::mt19937& rng, float stddev, float scale=1.0f) {
    WeightBuf b;
    b.n = n;
    CUDA_CHECK(cudaMalloc(&b.w, n * 4));
    CUDA_CHECK(cudaMalloc(&b.g, n * 4));
    CUDA_CHECK(cudaMalloc(&b.m, n * 4));
    CUDA_CHECK(cudaMalloc(&b.v, n * 4));
    CUDA_CHECK(cudaMemset(b.g, 0, n * 4));
    CUDA_CHECK(cudaMemset(b.m, 0, n * 4));
    CUDA_CHECK(cudaMemset(b.v, 0, n * 4));
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0, stddev);
    for (int i = 0; i < n; i++) {
        float val = scale == 1.0f ? nd(rng) : nd(rng) * scale;
        if (scale != 1.0f) val *= scale;
        h[i] = val;
    }
    CUDA_CHECK(cudaMemcpy(b.w, h.data(), n * 4, cudaMemcpyHostToDevice));
    return b;
}

float* alloc_f32(int n, float val) {
    float* p;
    CUDA_CHECK(cudaMalloc(&p, n * 4));
    std::vector<float> h(n, val);
    CUDA_CHECK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
    return p;
}

// ─── Data Loader ───────────────────────────────────────────────────────────
struct Dataset {
    std::vector<int> data;
    std::vector<int64_t> labels;
    int num_samples, seq_len;
    bool synthetic;
};

Dataset load_data(const std::string& path, int V, int S) {
    Dataset ds;
    ds.seq_len = S;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        printf("  No data file, using synthetic data (V=%d)\n", V);
        ds.synthetic = true;
        ds.num_samples = 512;
        ds.data.resize(ds.num_samples * S);
        std::mt19937 rng(42);
        for (int i = 0; i < ds.num_samples * S; i++)
            ds.data[i] = (int)(rng() % (V - 1)) + 1;
        return ds;
    }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    ds.data.resize(bytes / sizeof(int));
    f.read((char*)ds.data.data(), bytes);
    ds.seq_len = S;
    ds.num_samples = (int)ds.data.size() / S;
    printf("  Loaded %d samples x %d tokens (%zu MB)\n",
           ds.num_samples, S, bytes/1024/1024);

    std::string lb_path = path.substr(0, path.rfind('_')) + "_labels.bin";
    std::ifstream lbf(lb_path, std::ios::binary);
    if (lbf) {
        lbf.seekg(0, std::ios::end);
        size_t lbytes = lbf.tellg();
        lbf.seekg(0);
        ds.labels.resize(lbytes / sizeof(int64_t));
        lbf.read((char*)ds.labels.data(), lbytes);
        printf("  Loaded labels\n");
    }
    ds.synthetic = false;
    return ds;
}

void make_batch(Dataset& ds, int* h_input, int64_t* h_labels, int start, int B, int S) {
    for (int b = 0; b < B; b++) {
        int idx = (start + b) % ds.num_samples;
        int* src = ds.data.data() + idx * S;
        bool has_labels = ds.labels.size() == (size_t)ds.num_samples * S;
        for (int s = 0; s < S - 1; s++) {
            h_input[b * S + s] = src[s];
            h_labels[b * S + s] = has_labels ? ds.labels[idx * S + s] : (int64_t)src[s + 1];
        }
        h_input[b * S + (S - 1)] = 0;
        h_labels[b * S + (S - 1)] = -100;
    }
}

// ─── Model Config ──────────────────────────────────────────────────────────
struct ModelCfg {
    int D, V, L, H, d_ff, B, S;
};

ModelCfg auto_config(int vram_gb, int V, int S) {
    ModelCfg cfg;
    cfg.V = V;
    cfg.S = S;
    if (vram_gb >= 40) {
        cfg.D = 1280; cfg.L = 22; cfg.H = 20; cfg.d_ff = 3456; cfg.B = 8;
    } else if (vram_gb >= 16) {
        cfg.D = 1024; cfg.L = 16; cfg.H = 16; cfg.d_ff = 2816; cfg.B = 4;
    } else if (vram_gb >= 8) {
        cfg.D = 768;  cfg.L = 12; cfg.H = 12; cfg.d_ff = 2048; cfg.B = 4;
    } else {
        cfg.D = 512;  cfg.L = 8;  cfg.H = 8;  cfg.d_ff = 1536; cfg.B = 2;
    }
    return cfg;
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    cudaDeviceProp prop;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int vram_gb = (int)(prop.totalGlobalMem / (1024*1024*1024));

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  МУС — H100 Training (test model architecture)             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  GPU: %-45s ║\n", prop.name);
    printf("║  VRAM: %d GB                                                ║\n", vram_gb);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    int V = 42201;
    int S = 512;
    if (argc > 1) V = atoi(argv[1]);
    if (argc > 2) S = atoi(argv[2]);

    ModelCfg cfg = auto_config(vram_gb, V, S);
    printf("\nModel config: D=%d L=%d H=%d d_ff=%d V=%d B=%d S=%d\n",
           cfg.D, cfg.L, cfg.H, cfg.d_ff, cfg.V, cfg.B, cfg.S);
    int rows = cfg.B * cfg.S;

    double total_p = (double)cfg.V * cfg.D
                   + (double)cfg.L * ((double)cfg.D * 3 * cfg.D
                                    + (double)cfg.D * cfg.D
                                    + 2 * (double)cfg.D * cfg.d_ff
                                    + (double)cfg.d_ff * cfg.D
                                    + 2 * (double)cfg.D)
                   + (double)cfg.D;
    printf("Total params: %.1fM\n", total_p / 1e6);

    // ─── Data ────────────────────────────────────────────────────────────
    std::string data_path = "data/train_cache.bin";
    if (argc > 3) data_path = argv[3];
    Dataset ds = load_data(data_path, cfg.V, cfg.S);
    int N = ds.num_samples;

    // ─── Init ────────────────────────────────────────────────────────────
    std::mt19937 rng(42);
    size_t ws_size = 512ULL * 1024 * 1024;
    printf("Creating context (%zu MB)...\n", ws_size / 1024 / 1024);
    MUSContext* ctx = mus_create_context(ws_size);
    printf("OK\n");

    // ─── Embed ───────────────────────────────────────────────────────────
    size_t embed_n = (size_t)cfg.V * cfg.D;
    WeightBuf w_embed = alloc_weights(embed_n, rng, 0.02f);

    // ─── Layers ──────────────────────────────────────────────────────────
    std::vector<WeightBuf> w_qkv(cfg.L), w_o(cfg.L);
    std::vector<WeightBuf> w_mlp_g(cfg.L), w_mlp_u(cfg.L), w_mlp_d(cfg.L);

    // RMSNorm (single allocation per layer: block of 2*D)
    std::vector<float*> w_rn1(cfg.L), w_rn2(cfg.L);
    std::vector<float*> g_rn1(cfg.L), g_rn2(cfg.L);
    std::vector<float*> m_rn1(cfg.L), m_rn2(cfg.L);
    std::vector<float*> v_rn1(cfg.L), v_rn2(cfg.L);

    for (int l = 0; l < cfg.L; l++) {
        printf("Layer %d/%d...\r", l+1, cfg.L); fflush(stdout);
        int n_rn = cfg.D * 2;
        float *rn_w = alloc_f32(n_rn, 1.0f);
        w_rn1[l] = rn_w; w_rn2[l] = rn_w + cfg.D;
        float *rn_g; CUDA_CHECK(cudaMalloc(&rn_g, n_rn * 4));
        float *rn_m; CUDA_CHECK(cudaMalloc(&rn_m, n_rn * 4));
        float *rn_v; CUDA_CHECK(cudaMalloc(&rn_v, n_rn * 4));
        CUDA_CHECK(cudaMemset(rn_g, 0, n_rn * 4));
        CUDA_CHECK(cudaMemset(rn_m, 0, n_rn * 4));
        CUDA_CHECK(cudaMemset(rn_v, 0, n_rn * 4));
        g_rn1[l] = rn_g; g_rn2[l] = rn_g + cfg.D;
        m_rn1[l] = rn_m; m_rn2[l] = rn_m + cfg.D;
        v_rn1[l] = rn_v; v_rn2[l] = rn_v + cfg.D;

        w_qkv[l]   = alloc_weights(cfg.D * 3 * cfg.D, rng, 0.02f);
        w_o[l]     = alloc_weights(cfg.D * cfg.D, rng, 0.02f, 0.2f);
        w_mlp_g[l] = alloc_weights(cfg.D * cfg.d_ff, rng, 0.02f);
        w_mlp_u[l] = alloc_weights(cfg.D * cfg.d_ff, rng, 0.02f);
        w_mlp_d[l] = alloc_weights(cfg.d_ff * cfg.D, rng, 0.02f, 0.2f);
    }
    printf("          \n");

    // Final norm
    float *fn_w = alloc_f32(cfg.D, 1.0f);
    float *fn_g, *fn_m, *fn_v;
    CUDA_CHECK(cudaMalloc(&fn_g, cfg.D * 4));
    CUDA_CHECK(cudaMalloc(&fn_m, cfg.D * 4));
    CUDA_CHECK(cudaMalloc(&fn_v, cfg.D * 4));
    CUDA_CHECK(cudaMemset(fn_g, 0, cfg.D * 4));
    CUDA_CHECK(cudaMemset(fn_m, 0, cfg.D * 4));
    CUDA_CHECK(cudaMemset(fn_v, 0, cfg.D * 4));

    // Token weights
    float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, cfg.V * 4));
    MUSConfig mcfg = get_1b_config();
    mcfg.vocab_size = cfg.V;
    std::vector<float> h_weights(cfg.V);
    build_weight_table(mcfg, h_weights.data(), cfg.V);
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), cfg.V * 4, cudaMemcpyHostToDevice));

    // ─── Training buffers ───────────────────────────────────────────────
    int *d_input_ids;
    int64_t *d_labels64, *d_pos;
    float *d_logits, *d_trace, *d_fn_out, *d_loss;

    size_t trace_sz = (size_t)(cfg.L + 1) * rows * cfg.D * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input_ids, (size_t)cfg.B * cfg.S * 4));
    CUDA_CHECK(cudaMalloc(&d_labels64, (size_t)cfg.B * cfg.S * 8));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)rows * cfg.V * 4));
    CUDA_CHECK(cudaMalloc(&d_loss, 4));
    CUDA_CHECK(cudaMalloc(&d_trace, trace_sz));
    CUDA_CHECK(cudaMalloc(&d_fn_out, (size_t)rows * cfg.D * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)cfg.B * cfg.S * 8));

    std::vector<int> h_input(cfg.B * cfg.S);
    std::vector<int64_t> h_labels64(cfg.B * cfg.S);
    std::vector<int64_t> h_pos(cfg.B * cfg.S);
    for (int i = 0; i < cfg.B * cfg.S; i++) h_pos[i] = i % cfg.S;

    // ─── Training loop ──────────────────────────────────────────────────
    int steps_per_epoch = N / cfg.B;
    int num_epochs = 10;
    int global_step = 0;
    float base_lr = 5e-5f;
    int warmup = 200;
    float best_loss = 1e10f;

    printf("\nTraining: %d steps/epoch x %d epochs, lr=%.0e\n",
           steps_per_epoch, num_epochs, base_lr);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Timer epoch_timer;
        double total_loss = 0.0;
        int total_valid = 0;

        for (int step = 0; step < steps_per_epoch; step++) {
            int start = step * cfg.B;
            make_batch(ds, h_input.data(), h_labels64.data(), start, cfg.B, cfg.S);

            float* trace[23];
            int used_L = cfg.L;
            for (int l = 0; l <= used_L; l++)
                trace[l] = d_trace + (size_t)l * rows * cfg.D;

            CUDA_CHECK(cudaMemcpyAsync(d_input_ids, h_input.data(),
                cfg.B * cfg.S * 4, cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels64, h_labels64.data(),
                cfg.B * cfg.S * 8, cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(),
                cfg.B * cfg.S * 8, cudaMemcpyHostToDevice, ctx->stream));

            // Forward
            embed_fwd_kernel<<<(rows*cfg.D+255)/256, 256, 0, ctx->stream>>>(
                w_embed.w, (int*)d_input_ids, trace[0], cfg.B, cfg.S, cfg.D, cfg.V);

            for (int l = 0; l < used_L; l++) {
                mus_transformer_block_forward(ctx, trace[l], trace[l+1],
                    w_qkv[l].w, w_o[l].w,
                    w_mlp_g[l].w, w_mlp_u[l].w, w_mlp_d[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    cfg.B, cfg.S, cfg.D, cfg.H);
            }

            mus_rmsnorm_forward(ctx, trace[used_L], fn_w, d_fn_out, rows, cfg.D);
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cfg.V, rows, cfg.D, &alpha, w_embed.w, cfg.V,
                d_fn_out, cfg.D, &beta, d_logits, cfg.V);

            int valid = 0;
            for (int i = 0; i < cfg.B*cfg.S; i++)
                if (h_labels64[i] != -100) valid++;
            float loss_val = mus_weighted_ce_forward(ctx, d_logits, d_labels64,
                d_weights, d_loss, cfg.B, cfg.S, cfg.V);

            float step_loss = loss_val / fmaxf(1, valid);
            total_loss += loss_val;
            total_valid += valid;
            global_step++;

            float lr = (global_step < warmup)
                ? base_lr * ((float)global_step / warmup)
                : base_lr;

            // Backward
            float scale = 1.0f / fmaxf(1, valid);
            mus_weighted_ce_backward(ctx, d_logits, d_labels64,
                d_weights, scale, d_logits, cfg.B, cfg.S, cfg.V);

            CUDA_CHECK(cudaMemsetAsync(ctx->workspace, 0, ctx->workspace_size, ctx->stream));

            cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                cfg.D, rows, cfg.V, &alpha, w_embed.w, cfg.V,
                d_logits, cfg.V, &beta, d_fn_out, cfg.D);
            cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                cfg.V, cfg.D, rows, &alpha, d_logits, cfg.V,
                d_fn_out, cfg.D, &alpha, w_embed.g, cfg.V);

            mus_rmsnorm_backward(ctx, trace[used_L], fn_w,
                d_fn_out, d_fn_out, fn_g, rows, cfg.D);
            float* d_prev = d_fn_out;

            for (int l = used_L - 1; l >= 0; l--) {
                mus_clip_gradients(ctx, d_prev, rows * cfg.D, 0.5f);
                mus_transformer_block_backward(ctx, trace[l], d_prev,
                    w_qkv[l].w, w_o[l].w,
                    w_mlp_g[l].w, w_mlp_u[l].w, w_mlp_d[l].w,
                    w_rn1[l], w_rn2[l], d_pos,
                    d_fn_out,
                    w_qkv[l].g, w_o[l].g,
                    w_mlp_g[l].g, w_mlp_u[l].g, w_mlp_d[l].g,
                    g_rn1[l], g_rn2[l],
                    cfg.B, cfg.S, cfg.D, cfg.H);
                d_prev = d_fn_out;
            }

            embed_bwd_kernel<<<(rows*cfg.D+255)/256, 256, 0, ctx->stream>>>(
                d_prev, (int*)d_input_ids, w_embed.g, cfg.B, cfg.S, cfg.D, cfg.V);

            // Optimizer
            mus_clip_gradients(ctx, w_embed.g, embed_n, 0.5f);
            mus_adamw_step(ctx, w_embed.w, w_embed.m, w_embed.v,
                w_embed.g, embed_n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

            for (int l = 0; l < used_L; l++) {
                auto& q = w_qkv[l]; auto& o = w_o[l];
                auto& g = w_mlp_g[l]; auto& u = w_mlp_u[l]; auto& d = w_mlp_d[l];
                mus_clip_gradients(ctx, q.g, q.n, 0.5f);
                mus_clip_gradients(ctx, o.g, o.n, 0.5f);
                mus_clip_gradients(ctx, g.g, g.n, 0.5f);
                mus_clip_gradients(ctx, u.g, u.n, 0.5f);
                mus_clip_gradients(ctx, d.g, d.n, 0.5f);

                mus_adamw_step(ctx, q.w, q.m, q.v, q.g, q.n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, o.w, o.m, o.v, o.g, o.n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, g.w, g.m, g.v, g.g, g.n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, u.w, u.m, u.v, u.g, u.n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, d.w, d.m, d.v, d.g, d.n, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

                mus_clip_gradients(ctx, g_rn1[l], cfg.D, 1.0f);
                mus_clip_gradients(ctx, g_rn2[l], cfg.D, 1.0f);
                mus_adamw_step(ctx, w_rn1[l], m_rn1[l], v_rn1[l], g_rn1[l], cfg.D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
                mus_adamw_step(ctx, w_rn2[l], m_rn2[l], v_rn2[l], g_rn2[l], cfg.D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);
            }
            mus_clip_gradients(ctx, fn_g, cfg.D, 1.0f);
            mus_adamw_step(ctx, fn_w, fn_m, fn_v, fn_g, cfg.D, lr, 0.9f, 0.999f, 1e-8f, 0.01f, global_step);

            CUDA_CHECK(cudaMemsetAsync(w_embed.g, 0, embed_n * 4, ctx->stream));
            CUDA_CHECK(cudaMemsetAsync(fn_g, 0, cfg.D * 4, ctx->stream));
            for (int l = 0; l < used_L; l++) {
                auto& q = w_qkv[l]; auto& o = w_o[l];
                auto& g = w_mlp_g[l]; auto& u = w_mlp_u[l]; auto& d = w_mlp_d[l];
                CUDA_CHECK(cudaMemsetAsync(q.g, 0, q.n * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(o.g, 0, o.n * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g.g, 0, g.n * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(u.g, 0, u.n * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(d.g, 0, d.n * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn1[l], 0, cfg.D * 4, ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(g_rn2[l], 0, cfg.D * 4, ctx->stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

            if (step < 10 || step % 100 == 0 || step == steps_per_epoch - 1) {
                double avg = total_loss / fmaxf(1.0f, (float)total_valid);
                printf("  ep %2d/%d step %4d/%d  loss=%.4f  step=%.4f  valid=%d  lr=%.1e\n",
                       epoch+1, num_epochs, step, steps_per_epoch, avg, step_loss, valid, lr);
            }
        }

        float et = epoch_timer.ms() / 1000.0f;
        double avg = total_loss / fmaxf(1.0, (double)total_valid);
        if (avg < best_loss) best_loss = (float)avg;
        printf("  ═══ ep %2d/%d  loss=%.4f  time=%.1fs  best=%.4f ═══\n",
               epoch+1, num_epochs, avg, et, best_loss);

        // Save checkpoint
        if ((epoch + 1) % 5 == 0) {
            std::vector<float> save(embed_n);
            CUDA_CHECK(cudaMemcpy(save.data(), w_embed.w, embed_n * 4, cudaMemcpyDeviceToHost));
            char fn[256];
            snprintf(fn, sizeof(fn), "checkpoints/h100_epoch_%d.bin", epoch+1);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite(save.data(), 4, embed_n, f); fclose(f);
                printf("  Saved %s\n", fn); }
        }
    }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Done! Best loss: %.4f                                     ║\n", best_loss);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    mus_destroy_context(ctx);
    return 0;
}
