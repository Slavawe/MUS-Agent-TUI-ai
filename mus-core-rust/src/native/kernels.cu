#include "mus_cuda.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = call;                                        \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); }               \
} while(0)

// ══════════════════════════════════════════════════════════════════════
//  Device helpers
// ══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}
__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float s[34];
    int t = threadIdx.x, w = t / 32, l = t % 32;
    v = warp_reduce_max(v);
    if (l == 0) s[w] = v;
    __syncthreads();
    v = (t < (blockDim.x + 31) / 32) ? s[t] : -FLT_MAX;
    v = warp_reduce_max(v);
    if (t == 0) s[33] = v;
    __syncthreads();
    return s[33];
}
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float s[34];
    int t = threadIdx.x, w = t / 32, l = t % 32;
    v = warp_reduce_sum(v);
    if (l == 0) s[w] = v;
    __syncthreads();
    v = (t < (blockDim.x + 31) / 32) ? s[t] : 0.0f;
    v = warp_reduce_sum(v);
    if (t == 0) s[33] = v;
    __syncthreads();
    return s[33];
}

// ══════════════════════════════════════════════════════════════════════
//  Context
// ══════════════════════════════════════════════════════════════════════

__global__ void find_first_nan_kernel(const float* x, int n, int* out_pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (isnan(x[i]) || isinf(x[i]))
        atomicMin(out_pos, i);
}

MUSContext* mus_create_context(size_t ws_bytes) {
    MUSContext* ctx = new MUSContext();
    cudaStreamCreate(&ctx->stream);
    cublasCreate(&ctx->cublas);
    cublasSetStream(ctx->cublas, ctx->stream);
    CUDA_CHECK(cudaMalloc(&ctx->workspace, ws_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->workspace_f16, ws_bytes));
    // Both workspace and workspace_f32 point to same FP32 buffer
    ctx->workspace_f32 = (float*)ctx->workspace;
    ctx->workspace_size = ws_bytes;
    ctx->workspace_bytes = ws_bytes;
    return ctx;
}
void mus_destroy_context(MUSContext* ctx) {
    cudaStreamSynchronize(ctx->stream);
    cudaFree(ctx->workspace);
    cudaFree(ctx->workspace_f16);
    cublasDestroy(ctx->cublas);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

// ══════════════════════════════════════════════════════════════════════
//  Weight table
// ══════════════════════════════════════════════════════════════════════

void build_weight_table(const MUSConfig& cfg, float* w, int V) {
    for (int i = 0; i < V; i++) {
        if (i < cfg.cpp_tokens_start) w[i] = cfg.loss_weight_aer;
        else if (i <= cfg.cpp_tokens_end) w[i] = cfg.loss_weight_cpp;
        else if (i >= cfg.multimodal_tags_start && i <= cfg.multimodal_tags_end) w[i] = cfg.loss_weight_aer;
        else w[i] = cfg.loss_weight_text;
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Weighted Cross-Entropy Forward
// ══════════════════════════════════════════════════════════════════════

__global__ void ce_fwd_kernel(const float* __restrict__ logits, const int64_t* __restrict__ labels,
    const float* __restrict__ weights, float* __restrict__ loss_out,
    int B, int S, int V, int64_t ignore) {
    int bid = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    if (bid >= B * S) return;
    int b = bid / S, s = bid % S;
    int64_t label = labels[b * S + s];
    if (label == ignore) return;
    const float* pos = logits + (b * S + s) * V;
    float lmax = -FLT_MAX;
    for (int i = tid; i < V; i += bs) lmax = fmaxf(lmax, pos[i]);
    float gmax = block_reduce_max(lmax);
    float lsum = 0.0f;
    for (int i = tid; i < V; i += bs) lsum += expf(pos[i] - gmax);
    float denom = block_reduce_sum(lsum);
    __shared__ float s_sm, s_w;
    if ((label % bs) == tid) {
        int idx = (int)label;
        s_sm = expf(pos[idx] - gmax) / denom;
        s_w = weights[idx];
    }
    __syncthreads();
    float sm = fmaxf(s_sm, 1e-30f);
    if (tid == 0) atomicAdd(loss_out, -logf(sm) * s_w);
}

float mus_weighted_ce_forward(MUSContext* ctx, const float* logits,
    const int64_t* labels, const float* weights, float* loss,
    int B, int S, int V) {
    float zero = 0.0f;
    cudaMemcpyAsync(loss, &zero, 4, cudaMemcpyHostToDevice, ctx->stream);
    ce_fwd_kernel<<<B*S, 256, 0, ctx->stream>>>(logits, labels, weights, loss, B, S, V, -100LL);
    float h;
    cudaMemcpyAsync(&h, loss, 4, cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return h;
}

// ══════════════════════════════════════════════════════════════════════
//  Weighted Cross-Entropy Backward
// ══════════════════════════════════════════════════════════════════════

__global__ void ce_bwd_kernel(const float* __restrict__ logits, const int64_t* __restrict__ labels,
    const float* __restrict__ weights, float scale,
    float* __restrict__ dldx, int B, int S, int V, int64_t ignore) {
    int bid = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    if (bid >= B * S) return;
    int b = bid / S, s = bid % S;
    int64_t label = labels[b * S + s];
    const float* pos = logits + (b * S + s) * V;
    float* grad = dldx + (b * S + s) * V;
    if (label == ignore) {
        for (int i = tid; i < V; i += bs) grad[i] = 0.0f;
        return;
    }
    float lmax = -FLT_MAX;
    for (int i = tid; i < V; i += bs) lmax = fmaxf(lmax, pos[i]);
    float gmax = block_reduce_max(lmax);
    float lsum = 0.0f;
    for (int i = tid; i < V; i += bs) lsum += expf(pos[i] - gmax);
    float denom = block_reduce_sum(lsum);
    __shared__ float s_w;
    if ((label % bs) == tid) s_w = weights[(int)label];
    __syncthreads();
    float w = s_w;
    for (int i = tid; i < V; i += bs) {
        float sm = expf(pos[i] - gmax) / denom;
        grad[i] = (sm - (i == (int)label ? 1.0f : 0.0f)) * w * scale;
    }
}

void mus_weighted_ce_backward(MUSContext* ctx, const float* logits,
    const int64_t* labels, const float* weights, float scale,
    float* dldx, int B, int S, int V) {
    ce_bwd_kernel<<<B*S, 256, 0, ctx->stream>>>(logits, labels, weights, scale, dldx, B, S, V, -100LL);
}

// ══════════════════════════════════════════════════════════════════════
//  AdamW
// ══════════════════════════════════════════════════════════════════════

__global__ void adamw_kernel(float* p, float* m, float* v, const float* g,
    int N, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = g[i];
    float mi = m[i] = b1 * m[i] + (1.0f - b1) * gi;
    float vi = v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    float denom = sqrtf(vi / c2) + eps;
    if (denom < 1e-30f) denom = 1e-30f;
    p[i] -= lr * (mi / c1 / denom + wd * p[i]);
}

void mus_adamw_step(MUSContext* ctx, float* p, float* m, float* v,
    const float* g, int N, float lr, float b1, float b2,
    float eps, float wd, int step) {
    float c1 = fmaxf(1.0f - powf(b1, (float)step), 1e-10f);
    float c2 = fmaxf(1.0f - powf(b2, (float)step), 1e-10f);
    int bs = 256, grid = (N + bs - 1) / bs;
    adamw_kernel<<<grid, bs, 0, ctx->stream>>>(p, m, v, g, N, lr, b1, b2, eps, wd, c1, c2);
}

// ══════════════════════════════════════════════════════════════════════
//  RMSNorm Forward
// ══════════════════════════════════════════════════════════════════════

__global__ void rmsnorm_fwd_kernel(const float* __restrict__ x,
    const float* __restrict__ w, float* __restrict__ y, int cols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const float* rx = x + row * cols;
    float* ry = y + row * cols;
    float ls = 0.0f;
    for (int i = tid; i < cols; i += bs) ls += rx[i] * rx[i];
    float rms = sqrtf(block_reduce_sum(ls) / cols + eps);
    float ir = 1.0f / rms;
    for (int i = tid; i < cols; i += bs) ry[i] = rx[i] * ir * w[i];
}

void mus_rmsnorm_forward(MUSContext* ctx, const float* x,
    const float* w, float* y, int rows, int cols, float eps) {
    rmsnorm_fwd_kernel<<<rows, min(cols, 1024), 0, ctx->stream>>>(x, w, y, cols, eps);
}

// ══════════════════════════════════════════════════════════════════════
//  RMSNorm Backward
// ══════════════════════════════════════════════════════════════════════

__global__ void rmsnorm_bwd_kernel(const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ dy,
    float* __restrict__ dx, float* __restrict__ dw, int cols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const float* rx = x + row * cols;
    const float* rdy = dy + row * cols;
    float* rdx = dx + row * cols;
    float ls = 0.0f;
    for (int i = tid; i < cols; i += bs) ls += rx[i] * rx[i];
    float rms = sqrtf(block_reduce_sum(ls) / cols + eps);
    float ir = 1.0f / rms;
    float ldw = 0.0f;
    for (int i = tid; i < cols; i += bs) ldw += rdy[i] * rx[i] * ir;
    ldw = warp_reduce_sum(ldw);
    if ((tid % 32) == 0) atomicAdd(dw + tid / 32, ldw);
    for (int i = tid; i < cols; i += bs) rdx[i] = rdy[i] * w[i] * ir;
}

void mus_rmsnorm_backward(MUSContext* ctx, const float* x,
    const float* w, const float* dy, float* dx, float* dw,
    int rows, int cols, float eps) {
    cudaMemsetAsync(dw, 0, cols * sizeof(float), ctx->stream);
    rmsnorm_bwd_kernel<<<rows, min(cols, 1024), 0, ctx->stream>>>(x, w, dy, dx, dw, cols, eps);
}

// ══════════════════════════════════════════════════════════════════════
//  RoPE Forward
// ══════════════════════════════════════════════════════════════════════

__global__ void rope_fwd_kernel(float* __restrict__ q, float* __restrict__ k,
    const int64_t* __restrict__ pos, int B, int H, int S, int D) {
    int idx = blockIdx.x, tid = threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int b = idx / (H * S), r = idx % (H * S), h = r / S, s = r % S;
    float* rq = q + ((b * H + h) * S + s) * D;
    float* rk = k + ((b * H + h) * S + s) * D;
    float p = (float)pos[b * S + s];
    for (int i = tid; i < D / 2; i += blockDim.x) {
        float inv = 1.0f / powf(10000.0f, (2.0f * i) / (float)D);
        float th = p * inv, c = cosf(th), s_v = sinf(th);
        int i2 = i * 2;
        float q0 = rq[i2], q1 = rq[i2 + 1];
        rq[i2] = q0 * c - q1 * s_v; rq[i2 + 1] = q0 * s_v + q1 * c;
        float k0 = rk[i2], k1 = rk[i2 + 1];
        rk[i2] = k0 * c - k1 * s_v; rk[i2 + 1] = k0 * s_v + k1 * c;
    }
}

void mus_rope_forward(MUSContext* ctx, float* q, float* k,
    const int64_t* pos, int B, int H, int S, int D) {
    rope_fwd_kernel<<<B * H * S, min(D / 2, 256), 0, ctx->stream>>>(q, k, pos, B, H, S, D);
}

// ══════════════════════════════════════════════════════════════════════
//  RoPE Backward
// ══════════════════════════════════════════════════════════════════════

__global__ void rope_bwd_kernel(const int64_t* __restrict__ pos,
    float* __restrict__ dq, float* __restrict__ dk, int B, int H, int S, int D) {
    int idx = blockIdx.x, tid = threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int b = idx / (H * S), r = idx % (H * S), h = r / S, s = r % S;
    float* rdq = dq + ((b * H + h) * S + s) * D;
    float* rdk = dk + ((b * H + h) * S + s) * D;
    float p = (float)pos[b * S + s];
    for (int i = tid; i < D / 2; i += blockDim.x) {
        float inv = 1.0f / powf(10000.0f, (2.0f * i) / (float)D);
        float th = -p * inv, c = cosf(th), s_v = sinf(th);
        int i2 = i * 2;
        float q0 = rdq[i2], q1 = rdq[i2 + 1];
        rdq[i2] = q0 * c - q1 * s_v; rdq[i2 + 1] = q0 * s_v + q1 * c;
        float k0 = rdk[i2], k1 = rdk[i2 + 1];
        rdk[i2] = k0 * c - k1 * s_v; rdk[i2 + 1] = k0 * s_v + k1 * c;
    }
}

void mus_rope_backward(MUSContext* ctx, const int64_t* pos,
    float* dq, float* dk, int B, int H, int S, int D) {
    rope_bwd_kernel<<<B * H * S, min(D / 2, 256), 0, ctx->stream>>>(pos, dq, dk, B, H, S, D);
}

// ══════════════════════════════════════════════════════════════════════
//  SwiGLU Forward
// ══════════════════════════════════════════════════════════════════════

__global__ void swiglu_act_kernel(const float* __restrict__ gate,
    const float* __restrict__ up, float* __restrict__ act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    act[i] = (1.0f / (1.0f + expf(-g))) * g * up[i];
}

void mus_swiglu_forward(MUSContext* ctx, const float* x,
    const float* gate_w, const float* up_w, const float* down_w,
    float* out, int rows, int dim) {
    int cols4 = dim * 4;
    size_t sz = (size_t)rows * cols4 * sizeof(float);
    if (ctx->workspace_size < 3 * sz) {
        fprintf(stderr, "[swiglu] workspace need %.1f MB\n", (float)(3*sz)/1e6f);
        return;
    }
    float* gate = (float*)ctx->workspace;
    float* up = gate + (size_t)rows * cols4;
    float* act = up + (size_t)rows * cols4;
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cols4, rows, dim, &alpha, gate_w, cols4, x, dim, &beta, gate, cols4);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cols4, rows, dim, &alpha, up_w, cols4, x, dim, &beta, up, cols4);
    swiglu_act_kernel<<<(rows*cols4+255)/256, 256, 0, ctx->stream>>>(gate, up, act, rows*cols4);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, rows, cols4, &alpha, down_w, dim, act, cols4, &beta, out, dim);
}

// ══════════════════════════════════════════════════════════════════════
//  SwiGLU Backward
// ══════════════════════════════════════════════════════════════════════

__global__ void swiglu_bwd_kernel(const float* __restrict__ gate,
    const float* __restrict__ up, const float* __restrict__ d_act,
    float* __restrict__ d_gate, float* __restrict__ d_up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float silu = sig * g;
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    float da = d_act[i];
    d_gate[i] = da * up[i] * dsilu;
    d_up[i] = da * silu;
}

void mus_swiglu_backward(MUSContext* ctx,
    const float* x, const float* gate_w, const float* up_w, const float* down_w,
    const float* d_out, float* d_x,
    float* d_gate_w, float* d_up_w, float* d_down_w,
    int rows, int dim) {
    int cols4 = dim * 4;
    size_t sz = (size_t)rows * cols4 * sizeof(float);
    if (ctx->workspace_size < 4 * sz) {
        fprintf(stderr, "[swiglu_bwd] ws need %.1f MB\n", (float)(4*sz)/1e6f);
        return;
    }
    float* gate = (float*)ctx->workspace;
    float* up = gate + (size_t)rows * cols4;
    float* act = up + (size_t)rows * cols4;
    float* d_act = act + (size_t)rows * cols4;
    float alpha = 1.0f, beta = 0.0f;

    // Recompute forward
    #ifdef DEBUG_BLOCK_BWD
    float in_test[4]; cudaMemcpy(in_test, x, 16, cudaMemcpyDeviceToHost);
    int in_nan=0; for(int i=0;i<4;i++) if(isnan(in_test[i])||isinf(in_test[i])) in_nan=1;
    if(in_nan) printf("[SWIGLU] x (input) NaN at start!\n");
    cudaMemcpy(in_test, d_out, 16, cudaMemcpyDeviceToHost);
    in_nan=0; for(int i=0;i<4;i++) if(isnan(in_test[i])||isinf(in_test[i])) in_nan=1;
    if(in_nan) printf("[SWIGLU] d_out (upstream) NaN at start!\n");
    #endif

    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cols4, rows, dim, &alpha, gate_w, cols4, x, dim, &beta, gate, cols4);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cols4, rows, dim, &alpha, up_w, cols4, x, dim, &beta, up, cols4);
    swiglu_act_kernel<<<(rows*cols4+255)/256, 256, 0, ctx->stream>>>(gate, up, act, rows*cols4);

    #ifdef DEBUG_BLOCK_BWD
    float test[4]; cudaMemcpy(test, gate, 16, cudaMemcpyDeviceToHost);
    int nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] gate forward NaN!\n");
    #endif

    // d_down_w = d_out @ act^T, stored [dim, cols4] to match w_mlp_d(layout [D,4D])
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                dim, cols4, rows, &alpha, d_out, dim, act, cols4, &beta, d_down_w, dim);

    // d_act = down_w^T @ d_out
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                cols4, rows, dim, &alpha, down_w, dim, d_out, dim, &beta, d_act, cols4);

    #ifdef DEBUG_BLOCK_BWD
    cudaMemcpy(test, d_act, 16, cudaMemcpyDeviceToHost);
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_act AFTER down_w^T@d_out NaN!\n");
    #endif

    // Element-wise: d_gate, d_up (reuse gate/up buffers)
    swiglu_bwd_kernel<<<(rows*cols4+255)/256, 256, 0, ctx->stream>>>(gate, up, d_act, gate, up, rows*cols4);

    #ifdef DEBUG_BLOCK_BWD
    cudaMemcpy(test, gate, 16, cudaMemcpyDeviceToHost); // gate now contains d_gate
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_gate AFTER swiglu_bwd_kernel NaN!\n");
    cudaMemcpy(test, up, 16, cudaMemcpyDeviceToHost); // up now contains d_up
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_up AFTER swiglu_bwd_kernel NaN!\n");
    #endif

    // d_gate_w = gate^T @ x, stored [cols4, dim] to match w_mlp_g (layout [4D,D])
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                cols4, dim, rows, &alpha, gate, cols4, x, dim, &beta, d_gate_w, cols4);

    #ifdef DEBUG_BLOCK_BWD
    cudaMemcpy(test, d_gate_w, 16, cudaMemcpyDeviceToHost);
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_gate_w NaN!\n");
    #endif

    // d_up_w = up^T @ x, stored [cols4, dim] to match w_mlp_u (layout [4D,D])
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                cols4, dim, rows, &alpha, up, cols4, x, dim, &beta, d_up_w, cols4);

    // d_x = gate_w^T @ d_gate + up_w^T @ d_up
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                dim, rows, cols4, &alpha, gate_w, cols4, gate, cols4, &beta, d_x, dim);

    #ifdef DEBUG_BLOCK_BWD
    cudaMemcpy(test, d_x, 16, cudaMemcpyDeviceToHost);
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_x AFTER first GEMM NaN!\n");
    #endif

    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                dim, rows, cols4, &alpha, up_w, cols4, up, cols4, &alpha, d_x, dim);

    #ifdef DEBUG_BLOCK_BWD
    cudaMemcpy(test, d_x, 16, cudaMemcpyDeviceToHost);
    nan=0; for(int i=0;i<4;i++) if(isnan(test[i])||isinf(test[i])) nan=1;
    if(nan) printf("[SWIGLU] d_x FINAL NaN!\n");
    #endif
}

// ══════════════════════════════════════════════════════════════════════
//  Attention Forward
// ══════════════════════════════════════════════════════════════════════

__global__ void split_qkv_kernel(const float* __restrict__ qkv,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    int B, int S, int D, int H, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * hd;
    if (i >= total) return;
    int b = i / (H * S * hd), r = i % (H * S * hd);
    int h = r / (S * hd), r2 = r % (S * hd), s = r2 / hd, d_h = r2 % hd;
    int pos = b * S + s;
    int qkv_ofs = pos * 3 * D + h * hd + d_h;
    q[i] = qkv[qkv_ofs];
    k[i] = qkv[qkv_ofs + D];
    v[i] = qkv[qkv_ofs + 2 * D];
}

__global__ void merge_heads_kernel(const float* __restrict__ heads,
    float* __restrict__ out, int B, int S, int D, int H, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int b = i / (S * D), r = i % (S * D), s = r / D, d_full = r % D;
    int h = d_full / hd, d_h = d_full % hd;
    out[i] = heads[((b * H + h) * S + s) * hd + d_h];
}

__global__ void split_dout_kernel(const float* __restrict__ flat,
    float* __restrict__ heads, int B, int S, int D, int H, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * hd;
    if (i >= total) return;
    int b = i / (H * S * hd), r = i % (H * S * hd);
    int h = r / (S * hd), r2 = r % (S * hd), s = r2 / hd, d_h = r2 % hd;
    heads[i] = flat[(b * S + s) * D + h * hd + d_h];
}

__global__ void causal_softmax_kernel(float* __restrict__ scores, int B, int H, int S) {
    int bh = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    int b = bh / H, h = bh % H;
    float* head_scores = scores + ((size_t)b * H + h) * S * S;
    for (int i = tid; i < S; i += bs) {
        float* row = head_scores + i * S;
        float lmax = -FLT_MAX;
        for (int j = 0; j <= i; j++) lmax = fmaxf(lmax, row[j]);
        float lsum = 0.0f;
        for (int j = 0; j <= i; j++) lsum += expf(row[j] - lmax);
        float inv = 1.0f / fmaxf(lsum, 1e-30f);
        for (int j = 0; j <= i; j++) row[j] = expf(row[j] - lmax) * inv;
        for (int j = i+1; j < S; j++) row[j] = 0.0f;
    }
}

void mus_attention_forward(MUSContext* ctx, const float* x,
    const float* qkv_w, const float* o_w,
    const int64_t* pos, float* out, int B, int S, int D, int H) {
    int hd = D / H, rows = B * S;
    size_t sz_qkv = (size_t)rows * 3 * D * sizeof(float);
    size_t sz_h = (size_t)rows * D * sizeof(float);
    size_t sz_P = (size_t)B * H * S * S * sizeof(float);
    size_t need = sz_qkv + 4 * sz_h + sz_P;
    if (ctx->workspace_size < need) { fprintf(stderr,"[attn_fwd] need %.1f MB\n",(float)need/1e6f); return; }

    float* qkv_buf = (float*)ctx->workspace;
    float* q = (float*)((char*)qkv_buf + sz_qkv);
    float* k = (float*)((char*)q + sz_h);
    float* v = (float*)((char*)k + sz_h);
    float* P = (float*)((char*)v + sz_h);
    float* pre_out = (float*)((char*)P + sz_P);

    float alpha=1, beta=0;
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                3*D, rows, D, &alpha, qkv_w, 3*D, x, D, &beta, qkv_buf, 3*D);
    split_qkv_kernel<<<(B*H*S*hd+255)/256, 256, 0, ctx->stream>>>(qkv_buf, q, k, v, B, S, D, H, hd);
    mus_rope_forward(ctx, q, k, pos, B, H, S, hd);
    float scl = 1.0f/sqrtf((float)hd);
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        S, S, hd, &scl, q, hd, S*hd, k, hd, S*hd, &beta, P, S, S*S, B*H);
    causal_softmax_kernel<<<B*H, 256, 0, ctx->stream>>>(P, B, H, S);
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        hd, S, S, &alpha, v, hd, S*hd, P, S, S*S, &beta, q, hd, S*hd, B*H);
    merge_heads_kernel<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(q, pre_out, B, S, D, H, hd);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, rows, D, &alpha, o_w, D, pre_out, D, &beta, out, D);
}

// ══════════════════════════════════════════════════════════════════════
//  Attention Backward
// ══════════════════════════════════════════════════════════════════════

__global__ void softmax_bwd_kernel(const float* __restrict__ P,
    float* __restrict__ dP, int B, int H, int S) {
    int bh = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    int b = bh / H, h = bh % H;
    const float* hP = P + ((size_t)b*H+h)*S*S;
    float* hdP = dP + ((size_t)b*H+h)*S*S;
    for (int i = tid; i < S; i += bs) {
        const float* p_row = hP + i*S;
        float* dp_row = hdP + i*S;
        float sum_pdp = 0.0f;
        for (int k = 0; k <= i; k++) sum_pdp += p_row[k] * dp_row[k];
        for (int j = 0; j <= i; j++) dp_row[j] = p_row[j] * (dp_row[j] - sum_pdp);
        for (int j = i+1; j < S; j++) dp_row[j] = 0.0f;
    }
}

// Forward declarations for kernels defined below
__global__ void add_vectors_kernel(float* __restrict__ a, const float* __restrict__ b, int n);

void mus_attention_backward(MUSContext* ctx,
    const float* x, const float* qkv_w, const float* o_w,
    const int64_t* pos, const float* d_out,
    float* d_x, float* d_qkv_w, float* d_o_w,
    int B, int S, int D, int H) {
    int hd = D / H, rows = B * S;
    size_t sz_qkv = (size_t)rows*3*D*sizeof(float);
    size_t sz_h = (size_t)rows*D*sizeof(float);
    size_t sz_P = (size_t)B*H*S*S*sizeof(float);
    size_t need = sz_qkv + 3*sz_h + sz_P + 3*sz_h + 2*sz_P;
    if (ctx->workspace_size < need) { fprintf(stderr,"[attn_bwd] need %.1f MB\n",(float)need/1e6f); return; }

    float* qkv_buf = (float*)ctx->workspace;
    float* q = (float*)((char*)qkv_buf + sz_qkv);
    float* k = (float*)((char*)q + sz_h);
    float* v = (float*)((char*)k + sz_h);
    float* P = (float*)((char*)v + sz_h);
    float* dQ = (float*)((char*)P + sz_P);
    float* dK = (float*)((char*)dQ + sz_h);
    float* dV = (float*)((char*)dK + sz_h);
    float* dS_buf = (float*)((char*)dV + sz_h);     // [B,H,S,S]
    float* temp_h = (float*)((char*)dS_buf + sz_P);  // [rows,D] for d_pre

    float alpha=1, beta=0;

    // Recompute Q/K/V/P
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, 3*D,rows,D,&alpha, qkv_w,3*D, x,D,&beta, qkv_buf,3*D);
    split_qkv_kernel<<<(B*H*S*hd+255)/256,256,0,ctx->stream>>>(qkv_buf,q,k,v,B,S,D,H,hd);
    mus_rope_forward(ctx,q,k,pos,B,H,S,hd);
    float scl = 1.0f/sqrtf((float)hd);
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_T,CUBLAS_OP_N, S,S,hd,&scl, q,hd,S*hd, k,hd,S*hd, &beta, P,S,S*S, B*H);
    causal_softmax_kernel<<<B*H,256,0,ctx->stream>>>(P,B,H,S);

    // 1. d_pre = d_out @ o_w^T
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D,&alpha, o_w,D, d_out,D, &beta, temp_h,D);

    // 2. Split d_pre into heads → dQ as [B,H,S,hd] (dO per head)
    split_dout_kernel<<<(B*H*S*hd+255)/256,256,0,ctx->stream>>>(temp_h, dQ, B,S,D,H,hd);

    // 3. dV[dO] = dO @ P  [hd×S = hd×S @ S×S]
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_N, hd,S,S,&alpha, dQ,hd,S*hd, P,S,S*S, &beta, dV,hd,S*hd, B*H);

    // 4. dP = dO^T @ V  [S×S = S×hd @ hd×S]
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_T,CUBLAS_OP_N, S,S,hd,&alpha, dQ,hd,S*hd, v,hd,S*hd, &beta, dS_buf,S,S*S, B*H);

    // 5. Softmax backward: dS = P * (dP - sum P*dP)  (in-place dS_buf)
    softmax_bwd_kernel<<<(B*H+255)/256,256,0,ctx->stream>>>(P, dS_buf, B, H, S);

    // Temp buffers for Qrot, Krot gradients (head-sized)
    float* dQr = qkv_buf;  // reuse qkv_buf [B,S,D]
    float* dKr = (float*)((char*)dQr + sz_h);

    // 6. dQr = (1/√hd) * K @ dS^T
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_T, hd,S,S,&scl, k,hd,S*hd, dS_buf,S,S*S, &beta, dQr,hd,S*hd, B*H);
    // 7. dKr = (1/√hd) * Q @ dS
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_N, hd,S,S,&scl, q,hd,S*hd, dS_buf,S,S*S, &beta, dKr,hd,S*hd, B*H);

    // 8. RoPE backward on dQr, dKr
    mus_rope_backward(ctx, pos, dQr, dKr, B, H, S, hd);

    // 9. Merge heads → flat [B,S,D] using dS_buf as output (no overlap with dQr/dKr/dV)
    float* dQ_flat = dS_buf;
    float* dK_flat = dQ_flat + rows * D;
    float* dV_flat = dK_flat + rows * D;
    merge_heads_kernel<<<(rows*D+255)/256,256,0,ctx->stream>>>(dQr, dQ_flat, B,S,D,H,hd);
    merge_heads_kernel<<<(rows*D+255)/256,256,0,ctx->stream>>>(dKr, dK_flat, B,S,D,H,hd);
    cudaMemcpyAsync(temp_h, dV, sz_h, cudaMemcpyDeviceToDevice, ctx->stream);
    merge_heads_kernel<<<(rows*D+255)/256,256,0,ctx->stream>>>(temp_h, dV_flat, B,S,D,H,hd);

    // 10. d_qkv_w: dW = d_y @ x^T, stored [3D,D] with lda=3*D
    //     Use (N,T) to avoid transposing x[D,rows] (lda=D < rows)
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows,&alpha, dQ_flat,D, x,D, &beta, d_qkv_w, 3*D);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows,&alpha, dK_flat,D, x,D, &beta, d_qkv_w+D, 3*D);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows,&alpha, dV_flat,D, x,D, &beta, d_qkv_w+2*D, 3*D);

    // 11. d_o_w: dW = d_out @ pre_out^T, stored [D,D] with lda=D
    //     Use (N,T) to avoid transposing dV_flat[D,rows] (lda=D < rows)
    cublasSgemmStridedBatched(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_T, hd,S,S,&alpha, v,hd,S*hd, P,S,S*S, &beta, dQr,hd,S*hd, B*H);
    merge_heads_kernel<<<(rows*D+255)/256,256,0,ctx->stream>>>(dQr, dV_flat, B,S,D,H,hd);
    cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows,&alpha, d_out,D, dV_flat,D, &beta, d_o_w,D);

    // 12. d_x: gradient w.r.t. input, sum over Q/K/V blocks
    //     d_x += d_qkv_w_block^T @ dQ_flat  → use (T,N)
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D,&alpha, d_qkv_w, 3*D, dQ_flat,D, &beta, d_x,D);
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D,&alpha, d_qkv_w+D, 3*D, dK_flat,D, &alpha, d_x,D);
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D,&alpha, d_qkv_w+2*D, 3*D, dV_flat,D, &alpha, d_x,D);

    // 13. + d_out @ o_w^T (output proj gradient)
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D,&alpha, o_w,D, d_out,D, &beta, temp_h,D);
    int n = rows * D;
    add_vectors_kernel<<<(n+255)/256, 256, 0, ctx->stream>>>(d_x, temp_h, n);
}

// ══════════════════════════════════════════════════════════════════════
//  Utility kernels
// ══════════════════════════════════════════════════════════════════════

__global__ void add_vectors_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

void mus_add_vectors(MUSContext* ctx, float* a, const float* b, int rows, int D) {
    int n = rows * D;
    add_vectors_kernel<<<(n+255)/256, 256, 0, ctx->stream>>>(a, b, n);
}

// ══════════════════════════════════════════════════════════════════════
//  Transformer Block Forward
// ══════════════════════════════════════════════════════════════════════

void mus_transformer_block_forward(MUSContext* ctx,
    const float* x_in, float* x_out,
    const float* attn_qkv_w, const float* attn_o_w,
    const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos, int B, int S, int D, int H) {
    int rows = B * S;
    size_t sz = (size_t)rows * D * sizeof(float);
    // Use workspace: attention/swiglu internals use [0, ~200MB); block buffers at end
    size_t need = 4 * sz;
    if (ctx->workspace_size < need + 200*1024*1024) { fprintf(stderr, "[block_fwd] ws\n"); return; }
    size_t ofs = ctx->workspace_size - need;
    float* norm1 = (float*)((char*)ctx->workspace + ofs);
    float* attn_out = (float*)((char*)norm1 + sz);
    float* norm2 = (float*)((char*)attn_out + sz);
    float* mlp_out = (float*)((char*)norm2 + sz);

    mus_rmsnorm_forward(ctx, x_in, rms_norm_1_w, norm1, rows, D);
    mus_attention_forward(ctx, norm1, attn_qkv_w, attn_o_w, pos, attn_out, B, S, D, H);
    mus_add_vectors(ctx, attn_out, x_in, rows, D);
    mus_rmsnorm_forward(ctx, attn_out, rms_norm_2_w, norm2, rows, D);
    mus_swiglu_forward(ctx, norm2, mlp_gate_w, mlp_up_w, mlp_down_w, mlp_out, rows, D);
    mus_add_vectors(ctx, mlp_out, attn_out, rows, D);
    cudaMemcpyAsync(x_out, mlp_out, sz, cudaMemcpyDeviceToDevice, ctx->stream);
}

// ══════════════════════════════════════════════════════════════════════
//  Transformer Block Backward
// ══════════════════════════════════════════════════════════════════════

void mus_transformer_block_backward(MUSContext* ctx,
    const float* x_in, const float* d_out,
    const float* attn_qkv_w, const float* attn_o_w,
    const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos,
    float* d_x_in,
    float* d_attn_qkv_w, float* d_attn_o_w,
    float* d_mlp_gate_w, float* d_mlp_up_w, float* d_mlp_down_w,
    float* d_rms_norm_1_w, float* d_rms_norm_2_w,
    int B, int S, int D, int H) {
    int rows = B * S;
    size_t sz = (size_t)rows * D * sizeof(float);
    size_t attn_bwd_max = (size_t)rows * 3 * D * sizeof(float) + 7 * sz
                        + 2 * (size_t)B * H * S * S * sizeof(float);
    size_t blk_ofs = (attn_bwd_max + (1<<20) - 1) >> 20 << 20;
    if (ctx->workspace_size < blk_ofs + 6 * sz) { fprintf(stderr, "[block_bwd] ws\n"); return; }

    float* norm1 = (float*)((char*)ctx->workspace + blk_ofs);
    float* attn = (float*)((char*)norm1 + sz);
    float* norm2 = (float*)((char*)attn + sz);
    float* d_attn = (float*)((char*)norm2 + sz);
    float* d_norm2 = (float*)((char*)d_attn + sz);
    float* d_norm1 = (float*)((char*)d_norm2 + sz);

    // Debug: check d_out for NaN on entry (first call only via global flag)
    #ifdef DEBUG_BLOCK_BWD
    static int debug_first = 1;
    if (debug_first) {
        float test[4]; cudaMemcpy(test, d_out, 16, cudaMemcpyDeviceToHost);
        int nan = 0; for (int i=0;i<4;i++) if (isnan(test[i])||isinf(test[i])) nan=1;
        if (nan) printf("[DEBUG] d_out ENTER has NaN!\n");
        debug_first = 0;
    }
    #endif

    // Recompute forward activations
    mus_rmsnorm_forward(ctx, x_in, rms_norm_1_w, norm1, rows, D);
    mus_attention_forward(ctx, norm1, attn_qkv_w, attn_o_w, pos, attn, B, S, D, H);
    mus_add_vectors(ctx, attn, x_in, rows, D);
    mus_rmsnorm_forward(ctx, attn, rms_norm_2_w, norm2, rows, D);

    // Backward
    mus_swiglu_backward(ctx, norm2, mlp_gate_w, mlp_up_w, mlp_down_w,
                        d_out, d_norm2,
                        d_mlp_gate_w, d_mlp_up_w, d_mlp_down_w, rows, D);
    #ifdef DEBUG_BLOCK_BWD
    float test2[4]; cudaMemcpy(test2, d_norm2, 16, cudaMemcpyDeviceToHost);
    int nan2=0; for(int i=0;i<4;i++) if(isnan(test2[i])||isinf(test2[i])) nan2=1;
    if(nan2) printf("[DEBUG] d_norm2 AFTER swiglu_bwd NaN!\n");
    #endif

    mus_rmsnorm_backward(ctx, attn, rms_norm_2_w, d_norm2, d_attn, d_rms_norm_2_w, rows, D);
    mus_add_vectors(ctx, d_attn, d_out, rows, D);

    mus_attention_backward(ctx, norm1, attn_qkv_w, attn_o_w, pos,
                           d_attn, d_norm1, d_attn_qkv_w, d_attn_o_w, B, S, D, H);

    #ifdef DEBUG_BLOCK_BWD
    float test3[4]; cudaMemcpy(test3, d_norm1, 16, cudaMemcpyDeviceToHost);
    int nan3=0; for(int i=0;i<4;i++) if(isnan(test3[i])||isinf(test3[i])) nan3=1;
    if(nan3) printf("[DEBUG] d_norm1 AFTER attn_bwd NaN!\n");
    #endif

    mus_rmsnorm_backward(ctx, x_in, rms_norm_1_w, d_norm1, d_x_in, d_rms_norm_1_w, rows, D);
    mus_add_vectors(ctx, d_x_in, d_attn, rows, D);

    #ifdef DEBUG_BLOCK_BWD
    float test4[4]; cudaMemcpy(test4, d_x_in, 16, cudaMemcpyDeviceToHost);
    int nan4=0; for(int i=0;i<4;i++) if(isnan(test4[i])||isinf(test4[i])) nan4=1;
    if(nan4) printf("[DEBUG] d_x_in AFTER block NaN!\n");
    #endif
}

// ══════════════════════════════════════════════════════════════════════
//  Embedding
// ══════════════════════════════════════════════════════════════════════

__global__ void embed_fwd_kernel(const float* __restrict__ table,
    const int* __restrict__ input, float* __restrict__ out, int B, int S, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int pos = i / D, d = i % D;
    out[i] = table[input[pos] + d * V];
}

__global__ void embed_bwd_kernel(const float* __restrict__ out_grad,
    const int* __restrict__ input, float* __restrict__ table_grad, int B, int S, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int pos = i / D, d = i % D;
    atomicAdd(&table_grad[input[pos] + d * V], out_grad[i]);
}

// ══════════════════════════════════════════════════════════════════════
//  Debug: compare GPU to CPU reference
// ══════════════════════════════════════════════════════════════════════

float mus_check_tensor(const float* d_data, const float* h_ref, int n,
                       const char* name, MUSContext* ctx) {
    float* h_gpu = new float[n];
    cudaMemcpyAsync(h_gpu, d_data, n * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx ? ctx->stream : 0);
    if (ctx) cudaStreamSynchronize(ctx->stream);
    else cudaDeviceSynchronize();
    float max_err = 0.0f;
    int bad = 0;
    for (int i = 0; i < n && i < 10; i++) {
        float e = fabsf(h_gpu[i] - h_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-4f && bad < 5) {
            printf("  [%s] [%d] gpu=%.6f cpu=%.6f diff=%.2e\n", name, i, h_gpu[i], h_ref[i], e);
            bad++;
        }
    }
    printf("  [%s] max_err=%.2e  %s\n", name, max_err, max_err < 1e-4f ? "PASS" : "FAIL");
    delete[] h_gpu;
    return max_err;
}

// ══════════════════════════════════════════════════════════════════════
//  Gradient Clipping
// ══════════════════════════════════════════════════════════════════════

__global__ void clip_grad_kernel(float* g, int n, float max_norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = g[i];
        if (val > max_norm) g[i] = max_norm;
        else if (val < -max_norm) g[i] = -max_norm;
    }
}

void mus_clip_gradients(MUSContext* ctx, float* g, int n, float max_norm) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clip_grad_kernel<<<blocks, threads, 0, ctx->stream>>>(g, n, max_norm);
}
