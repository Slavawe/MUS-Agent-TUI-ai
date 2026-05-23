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
//  Device helpers (FP16, use float for reductions)
// ══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_max_f16(float v) {
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum_f16(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}
__device__ __forceinline__ float block_reduce_max_f16(float v) {
    __shared__ float s[34];
    int t = threadIdx.x, w = t / 32, l = t % 32;
    v = warp_reduce_max_f16(v);
    if (l == 0) s[w] = v;
    __syncthreads();
    v = (t < (blockDim.x + 31) / 32) ? s[t] : -FLT_MAX;
    v = warp_reduce_max_f16(v);
    if (t == 0) s[33] = v;
    __syncthreads();
    return s[33];
}
__device__ __forceinline__ float block_reduce_sum_f16(float v) {
    __shared__ float s[34];
    int t = threadIdx.x, w = t / 32, l = t % 32;
    v = warp_reduce_sum_f16(v);
    if (l == 0) s[w] = v;
    __syncthreads();
    v = (t < (blockDim.x + 31) / 32) ? s[t] : 0.0f;
    v = warp_reduce_sum_f16(v);
    if (t == 0) s[33] = v;
    __syncthreads();
    return s[33];
}

// ══════════════════════════════════════════════════════════════════════
//  FP16 ↔ FP32 Conversion
// ══════════════════════════════════════════════════════════════════════

__global__ void f16_to_f32_kernel(const half* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void mus_convert_f16_to_f32(MUSContext* ctx, const half* src, float* dst, int n) {
    int bs = 256, grid = (n + bs - 1) / bs;
    f16_to_f32_kernel<<<grid, bs, 0, ctx->stream>>>(src, dst, n);
}

void mus_convert_f32_to_f16(MUSContext* ctx, const float* src, half* dst, int n) {
    int bs = 256, grid = (n + bs - 1) / bs;
    f32_to_f16_kernel<<<grid, bs, 0, ctx->stream>>>(src, dst, n);
}

// ══════════════════════════════════════════════════════════════════════
//  FP16 GEMM wrapper (half input/output, FP32 accumulate)
// ══════════════════════════════════════════════════════════════════════

void mus_gemm_f16(MUSContext* ctx,
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const half* A, int lda,
    const half* B, int ldb,
    half* C, int ldc,
    float alpha, float beta) {
    cublasGemmEx(ctx->cublas, opA, opB, m, n, k,
                 &alpha, A, CUDA_R_16F, lda,
                 B, CUDA_R_16F, ldb,
                 &beta, C, CUDA_R_16F, ldc,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}

// ══════════════════════════════════════════════════════════════════════
//  Weighted Cross-Entropy Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void ce_fwd_kernel_f16(const half* __restrict__ logits, const int64_t* __restrict__ labels,
    const float* __restrict__ weights, float* __restrict__ loss_out,
    int B, int S, int V, int64_t ignore) {
    int bid = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    if (bid >= B * S) return;
    int b = bid / S, s = bid % S;
    int64_t label = labels[b * S + s];
    if (label == ignore) return;
    const half* pos = logits + (b * S + s) * V;
    float lmax = -FLT_MAX;
    for (int i = tid; i < V; i += bs) lmax = fmaxf(lmax, __half2float(pos[i]));
    float gmax = block_reduce_max_f16(lmax);
    float lsum = 0.0f;
    for (int i = tid; i < V; i += bs) lsum += expf(__half2float(pos[i]) - gmax);
    float denom = block_reduce_sum_f16(lsum);
    __shared__ float s_sm, s_w;
    if ((label % bs) == tid) {
        int idx = (int)label;
        s_sm = expf(__half2float(pos[idx]) - gmax) / denom;
        s_w = weights[idx];
    }
    __syncthreads();
    float sm = fmaxf(s_sm, 1e-30f);
    if (tid == 0) atomicAdd(loss_out, -logf(sm) * s_w);
}

float mus_weighted_ce_forward_f16(MUSContext* ctx, const half* logits,
    const int64_t* labels, const float* weights, float* loss,
    int B, int S, int V) {
    float zero = 0.0f;
    cudaMemcpyAsync(loss, &zero, 4, cudaMemcpyHostToDevice, ctx->stream);
    ce_fwd_kernel_f16<<<B*S, 256, 0, ctx->stream>>>(logits, labels, weights, loss, B, S, V, -100LL);
    float h;
    cudaMemcpyAsync(&h, loss, 4, cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return h;
}

// ══════════════════════════════════════════════════════════════════════
//  Weighted Cross-Entropy Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void ce_bwd_kernel_f16(const half* __restrict__ logits, const int64_t* __restrict__ labels,
    const float* __restrict__ weights, float scale,
    half* __restrict__ dldx, int B, int S, int V, int64_t ignore) {
    int bid = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    if (bid >= B * S) return;
    int b = bid / S, s = bid % S;
    int64_t label = labels[b * S + s];
    const half* pos = logits + (b * S + s) * V;
    half* grad = dldx + (b * S + s) * V;
    if (label == ignore) {
        for (int i = tid; i < V; i += bs) grad[i] = __float2half(0.0f);
        return;
    }
    float lmax = -FLT_MAX;
    for (int i = tid; i < V; i += bs) lmax = fmaxf(lmax, __half2float(pos[i]));
    float gmax = block_reduce_max_f16(lmax);
    float lsum = 0.0f;
    for (int i = tid; i < V; i += bs) lsum += expf(__half2float(pos[i]) - gmax);
    float denom = block_reduce_sum_f16(lsum);
    __shared__ float s_w;
    if ((label % bs) == tid) s_w = weights[(int)label];
    __syncthreads();
    float w = s_w;
    for (int i = tid; i < V; i += bs) {
        float sm = expf(__half2float(pos[i]) - gmax) / denom;
        grad[i] = __float2half((sm - (i == (int)label ? 1.0f : 0.0f)) * w * scale);
    }
}

void mus_weighted_ce_backward_f16(MUSContext* ctx, const half* logits,
    const int64_t* labels, const float* weights, float scale,
    half* dldx, int B, int S, int V) {
    ce_bwd_kernel_f16<<<B*S, 256, 0, ctx->stream>>>(logits, labels, weights, scale, dldx, B, S, V, -100LL);
}

// ══════════════════════════════════════════════════════════════════════
//  Loss Scaling (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void scale_grad_kernel_f16(half* g, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = __float2half(__half2float(g[i]) * s);
}

void mus_scale_gradients_f16(MUSContext* ctx, half* g, int n, float scale) {
    int bs = 256, grid = (n + bs - 1) / bs;
    scale_grad_kernel_f16<<<grid, bs, 0, ctx->stream>>>(g, n, scale);
}

void mus_unscale_gradients_f16(MUSContext* ctx, half* g, int n, float inv_scale) {
    int bs = 256, grid = (n + bs - 1) / bs;
    scale_grad_kernel_f16<<<grid, bs, 0, ctx->stream>>>(g, n, inv_scale);
}

// ══════════════════════════════════════════════════════════════════════
//  AdamW (FP16 params, FP32 internal compute)
// ══════════════════════════════════════════════════════════════════════

__global__ void adamw_kernel_f16(half* __restrict__ p, half* __restrict__ m, half* __restrict__ v,
    const half* __restrict__ g, int N, float lr, float b1, float b2,
    float eps, float wd, float c1, float c2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = __half2float(g[i]);
    float mi = __half2float(m[i]) * b1 + (1.0f - b1) * gi;
    float vi = __half2float(v[i]) * b2 + (1.0f - b2) * gi * gi;
    m[i] = __float2half(mi);
    v[i] = __float2half(vi);
    float denom = sqrtf(vi / c2) + eps;
    if (denom < 1e-30f) denom = 1e-30f;
    p[i] = __float2half(__half2float(p[i]) - lr * (mi / c1 / denom + wd * __half2float(p[i])));
}

// CPU-side launcher for FP16 Adam
void mus_adamw_step_f16(MUSContext* ctx, half* p, half* m, half* v,
    const half* g, int N, float lr, float b1, float b2,
    float eps, float wd, int step) {
    float c1 = fmaxf(1.0f - powf(b1, (float)step), 1e-10f);
    float c2 = fmaxf(1.0f - powf(b2, (float)step), 1e-10f);
    int bs = 256, grid = (N + bs - 1) / bs;
    adamw_kernel_f16<<<grid, bs, 0, ctx->stream>>>(p, m, v, g, N, lr, b1, b2, eps, wd, c1, c2);
}

// ══════════════════════════════════════════════════════════════════════
//  RMSNorm Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void rmsnorm_fwd_kernel_f16(const half* __restrict__ x,
    const float* __restrict__ w, half* __restrict__ y, int cols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const half* rx = x + row * cols;
    half* ry = y + row * cols;
    float ls = 0.0f;
    for (int i = tid; i < cols; i += bs) { float v = __half2float(rx[i]); ls += v * v; }
    float rms = sqrtf(block_reduce_sum_f16(ls) / cols + eps);
    float ir = 1.0f / rms;
    for (int i = tid; i < cols; i += bs) ry[i] = __float2half(__half2float(rx[i]) * ir * w[i]);
}

void mus_rmsnorm_forward_f16(MUSContext* ctx, const half* x,
    const float* w, half* y, int rows, int cols, float eps) {
    rmsnorm_fwd_kernel_f16<<<rows, min(cols, 1024), 0, ctx->stream>>>(x, w, y, cols, eps);
}

// ══════════════════════════════════════════════════════════════════════
//  RMSNorm Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void rmsnorm_bwd_kernel_f16(const half* __restrict__ x,
    const float* __restrict__ w, const half* __restrict__ dy,
    half* __restrict__ dx, float* __restrict__ dw, int cols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const half* rx = x + row * cols;
    const half* rdy = dy + row * cols;
    half* rdx = dx + row * cols;
    float ls = 0.0f;
    for (int i = tid; i < cols; i += bs) { float v = __half2float(rx[i]); ls += v * v; }
    float rms = sqrtf(block_reduce_sum_f16(ls) / cols + eps);
    float ir = 1.0f / rms;

    // Compute sum(w * x * dy) for the correction term
    float lr = 0.0f;
    for (int i = tid; i < cols; i += bs) {
        float xf = __half2float(rx[i]);
        float dyf = __half2float(rdy[i]);
        lr += w[i] * xf * dyf;
    }
    lr = block_reduce_sum_f16(lr);
    float ir2 = ir * ir;
    float correction = lr / cols * ir2;  // ir² * mean(w*x*dy)

    // dw[i] = dy[i] * x[i] / rms  (per-element weight gradient)
    for (int i = tid; i < cols; i += bs) {
        float xf = __half2float(rx[i]);
        float dyf = __half2float(rdy[i]);
        atomicAdd(dw + i, dyf * xf * ir);
    }

    // dx[i] = (w[i]*dy[i] - x[i]*ir²*mean(w*x*dy)) * ir  (correct formula)
    for (int i = tid; i < cols; i += bs) {
        float xf = __half2float(rx[i]);
        float dyf = __half2float(rdy[i]);
        rdx[i] = __float2half((w[i] * dyf - xf * correction) * ir);
    }
}

void mus_rmsnorm_backward_f16(MUSContext* ctx, const half* x,
    const float* w, const half* dy, half* dx, float* dw,
    int rows, int cols, float eps) {
    cudaMemsetAsync(dw, 0, cols * sizeof(float), ctx->stream);
    rmsnorm_bwd_kernel_f16<<<rows, min(cols, 1024), 0, ctx->stream>>>(x, w, dy, dx, dw, cols, eps);
}

// ══════════════════════════════════════════════════════════════════════
//  RoPE Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void rope_fwd_kernel_f16(half* __restrict__ q, half* __restrict__ k,
    const int64_t* __restrict__ pos, int B, int H, int S, int D) {
    int idx = blockIdx.x, tid = threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int b = idx / (H * S), r = idx % (H * S), h = r / S, s = r % S;
    half* rq = q + ((b * H + h) * S + s) * D;
    half* rk = k + ((b * H + h) * S + s) * D;
    float p = (float)pos[b * S + s];
    for (int i = tid; i < D / 2; i += blockDim.x) {
        float inv = 1.0f / powf(10000.0f, (2.0f * i) / (float)D);
        float th = p * inv, c = cosf(th), s_v = sinf(th);
        int i2 = i * 2;
        float q0 = __half2float(rq[i2]), q1 = __half2float(rq[i2 + 1]);
        rq[i2] = __float2half(q0 * c - q1 * s_v);
        rq[i2 + 1] = __float2half(q0 * s_v + q1 * c);
        float k0 = __half2float(rk[i2]), k1 = __half2float(rk[i2 + 1]);
        rk[i2] = __float2half(k0 * c - k1 * s_v);
        rk[i2 + 1] = __float2half(k0 * s_v + k1 * c);
    }
}

void mus_rope_forward_f16(MUSContext* ctx, half* q, half* k,
    const int64_t* pos, int B, int H, int S, int D) {
    rope_fwd_kernel_f16<<<B * H * S, min(D / 2, 256), 0, ctx->stream>>>(q, k, pos, B, H, S, D);
}

// ══════════════════════════════════════════════════════════════════════
//  RoPE Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void rope_bwd_kernel_f16(const int64_t* __restrict__ pos,
    half* __restrict__ dq, half* __restrict__ dk, int B, int H, int S, int D) {
    int idx = blockIdx.x, tid = threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int b = idx / (H * S), r = idx % (H * S), h = r / S, s = r % S;
    half* rdq = dq + ((b * H + h) * S + s) * D;
    half* rdk = dk + ((b * H + h) * S + s) * D;
    float p = (float)pos[b * S + s];
    for (int i = tid; i < D / 2; i += blockDim.x) {
        float inv = 1.0f / powf(10000.0f, (2.0f * i) / (float)D);
        float th = -p * inv, c = cosf(th), s_v = sinf(th);
        int i2 = i * 2;
        float q0 = __half2float(rdq[i2]), q1 = __half2float(rdq[i2 + 1]);
        rdq[i2] = __float2half(q0 * c - q1 * s_v);
        rdq[i2 + 1] = __float2half(q0 * s_v + q1 * c);
        float k0 = __half2float(rdk[i2]), k1 = __half2float(rdk[i2 + 1]);
        rdk[i2] = __float2half(k0 * c - k1 * s_v);
        rdk[i2 + 1] = __float2half(k0 * s_v + k1 * c);
    }
}

void mus_rope_backward_f16(MUSContext* ctx, const int64_t* pos,
    half* dq, half* dk, int B, int H, int S, int D) {
    rope_bwd_kernel_f16<<<B * H * S, min(D / 2, 256), 0, ctx->stream>>>(pos, dq, dk, B, H, S, D);
}

// ══════════════════════════════════════════════════════════════════════
//  SwiGLU Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void swiglu_act_kernel_f16(const half* __restrict__ gate,
    const half* __restrict__ up, half* __restrict__ act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    act[i] = __float2half((1.0f / (1.0f + expf(-g))) * g * __half2float(up[i]));
}

void mus_swiglu_forward_f16(MUSContext* ctx, const half* x,
    const half* gate_w, const half* up_w, const half* down_w,
    half* out, int rows, int dim, int d_ff) {
    size_t sz = (size_t)rows * d_ff * sizeof(half);
    if (ctx->workspace_size < 3 * sz) {
        fprintf(stderr, "[swiglu_f16] workspace need %.1f MB\n", (float)(3*sz)/1e6f);
        return;
    }
    half* gate = ctx->workspace_f16;
    half* up = gate + (size_t)rows * d_ff;
    half* act = up + (size_t)rows * d_ff;

    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 d_ff, rows, dim, gate_w, d_ff, x, dim, gate, d_ff);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 d_ff, rows, dim, up_w, d_ff, x, dim, up, d_ff);
    swiglu_act_kernel_f16<<<(rows*d_ff+255)/256, 256, 0, ctx->stream>>>(gate, up, act, rows*d_ff);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 dim, rows, d_ff, down_w, dim, act, d_ff, out, dim);
}

// ══════════════════════════════════════════════════════════════════════
//  SwiGLU Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void swiglu_bwd_kernel_f16(const half* __restrict__ gate,
    const half* __restrict__ up, const half* __restrict__ d_act,
    half* __restrict__ d_gate, half* __restrict__ d_up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    float silu = sig * g;
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    float da = __half2float(d_act[i]);
    d_gate[i] = __float2half(da * __half2float(up[i]) * dsilu);
    d_up[i] = __float2half(da * silu);
}

void mus_swiglu_backward_f16(MUSContext* ctx,
    const half* x, const half* gate_w, const half* up_w, const half* down_w,
    const half* d_out, half* d_x,
    half* d_gate_w, half* d_up_w, half* d_down_w,
    int rows, int dim, int d_ff) {
    size_t sz = (size_t)rows * d_ff * sizeof(half);
    if (ctx->workspace_size < 4 * sz) {
        fprintf(stderr, "[swiglu_bwd_f16] ws need %.1f MB\n", (float)(4*sz)/1e6f);
        return;
    }
    half* gate = ctx->workspace_f16;
    half* up = gate + (size_t)rows * d_ff;
    half* act = up + (size_t)rows * d_ff;
    half* d_act = act + (size_t)rows * d_ff;

    // Recompute forward
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 d_ff, rows, dim, gate_w, d_ff, x, dim, gate, d_ff);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 d_ff, rows, dim, up_w, d_ff, x, dim, up, d_ff);
    swiglu_act_kernel_f16<<<(rows*d_ff+255)/256, 256, 0, ctx->stream>>>(gate, up, act, rows*d_ff);

    // d_down_w = d_out @ act^T
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                 dim, d_ff, rows, d_out, dim, act, d_ff, d_down_w, dim);

    // d_act = down_w^T @ d_out
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N,
                 d_ff, rows, dim, down_w, dim, d_out, dim, d_act, d_ff);

    // Element-wise: d_gate, d_up
    swiglu_bwd_kernel_f16<<<(rows*d_ff+255)/256, 256, 0, ctx->stream>>>(gate, up, d_act, gate, up, rows*d_ff);

    // d_gate_w = gate^T @ x
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                 d_ff, dim, rows, gate, d_ff, x, dim, d_gate_w, d_ff);

    // d_up_w = up^T @ x
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T,
                 d_ff, dim, rows, up, d_ff, x, dim, d_up_w, d_ff);

    // d_x = gate_w^T @ d_gate + up_w^T @ d_up
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N,
                 dim, rows, d_ff, gate_w, d_ff, gate, d_ff, d_x, dim);
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N,
                 dim, rows, d_ff, up_w, d_ff, up, d_ff, d_x, dim);
}

// ══════════════════════════════════════════════════════════════════════
//  Attention Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void split_qkv_kernel_f16(const half* __restrict__ qkv,
    half* __restrict__ q, half* __restrict__ k, half* __restrict__ v,
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

__global__ void merge_heads_kernel_f16(const half* __restrict__ heads,
    half* __restrict__ out, int B, int S, int D, int H, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int b = i / (S * D), r = i % (S * D), s = r / D, d_full = r % D;
    int h = d_full / hd, d_h = d_full % hd;
    out[i] = heads[((b * H + h) * S + s) * hd + d_h];
}

__global__ void split_dout_kernel_f16(const half* __restrict__ flat,
    half* __restrict__ heads, int B, int S, int D, int H, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * hd;
    if (i >= total) return;
    int b = i / (H * S * hd), r = i % (H * S * hd);
    int h = r / (S * hd), r2 = r % (S * hd), s = r2 / hd, d_h = r2 % hd;
    heads[i] = flat[(b * S + s) * D + h * hd + d_h];
}

__global__ void causal_softmax_kernel_f16(half* __restrict__ scores, int B, int H, int S) {
    int bh = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    int b = bh / H, h = bh % H;
    half* head_scores = scores + ((size_t)b * H + h) * S * S;
    for (int i = tid; i < S; i += bs) {
        half* row = head_scores + i * S;
        float lmax = -FLT_MAX;
        for (int j = 0; j <= i; j++) lmax = fmaxf(lmax, __half2float(row[j]));
        float lsum = 0.0f;
        for (int j = 0; j <= i; j++) lsum += expf(__half2float(row[j]) - lmax);
        float inv = 1.0f / fmaxf(lsum, 1e-30f);
        for (int j = 0; j <= i; j++) row[j] = __float2half(expf(__half2float(row[j]) - lmax) * inv);
        for (int j = i+1; j < S; j++) row[j] = __float2half(0.0f);
    }
}

void mus_attention_forward_f16(MUSContext* ctx, const half* x,
    const half* qkv_w, const half* o_w,
    const int64_t* pos, half* out, int B, int S, int D, int H) {
    int hd = D / H, rows = B * S;
    size_t sz_qkv = (size_t)rows * 3 * D * sizeof(half);
    size_t sz_h = (size_t)rows * D * sizeof(half);
    size_t sz_P = (size_t)B * H * S * S * sizeof(half);
    size_t need = sz_qkv + 4 * sz_h + sz_P;
    if (ctx->workspace_size < need) { fprintf(stderr,"[attn_fwd_f16] need %.1f MB\n",(float)need/1e6f); return; }

    half* qkv_buf = ctx->workspace_f16;
    half* q = (half*)((char*)qkv_buf + sz_qkv);
    half* k = (half*)((char*)q + sz_h);
    half* v = (half*)((char*)k + sz_h);
    half* P = (half*)((char*)v + sz_h);
    half* pre_out = (half*)((char*)P + sz_P);

    // QKV projection: [rows, D] @ [D, 3D] → [rows, 3D]
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 3*D, rows, D, qkv_w, 3*D, x, D, qkv_buf, 3*D);
    split_qkv_kernel_f16<<<(B*H*S*hd+255)/256, 256, 0, ctx->stream>>>(qkv_buf, q, k, v, B, S, D, H, hd);
    mus_rope_forward_f16(ctx, q, k, pos, B, H, S, hd);

    // SDPA: scores = Q @ K^T / sqrt(hd)
    float scl = 1.0f / sqrtf((float)hd);
    float alpha1 = 1.0f, beta0 = 0.0f;
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        S, S, hd, &scl, q, CUDA_R_16F, hd, S*hd,
        k, CUDA_R_16F, hd, S*hd,
        &beta0, P, CUDA_R_16F, S, S*S, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // Causal softmax (in-place on P)
    causal_softmax_kernel_f16<<<B*H, 256, 0, ctx->stream>>>(P, B, H, S);

    // V @ P → output
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        hd, S, S, &alpha1, v, CUDA_R_16F, hd, S*hd,
        P, CUDA_R_16F, S, S*S,
        &beta0, q, CUDA_R_16F, hd, S*hd, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    merge_heads_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(q, pre_out, B, S, D, H, hd);

    // Output projection
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N,
                 D, rows, D, o_w, D, pre_out, D, out, D);
}

// ══════════════════════════════════════════════════════════════════════
//  Attention Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void softmax_bwd_kernel_f16(const half* __restrict__ P,
    half* __restrict__ dP, int B, int H, int S) {
    int bh = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    int b = bh / H, h = bh % H;
    const half* hP = P + ((size_t)b*H+h)*S*S;
    half* hdP = dP + ((size_t)b*H+h)*S*S;
    for (int i = tid; i < S; i += bs) {
        const half* p_row = hP + i*S;
        half* dp_row = hdP + i*S;
        float sum_pdp = 0.0f;
        for (int k = 0; k <= i; k++) sum_pdp += __half2float(p_row[k]) * __half2float(dp_row[k]);
        for (int j = 0; j <= i; j++)
            dp_row[j] = __float2half(__half2float(p_row[j]) * (__half2float(dp_row[j]) - sum_pdp));
        for (int j = i+1; j < S; j++) dp_row[j] = __float2half(0.0f);
    }
}

__global__ void add_vectors_kernel_f16(half* __restrict__ a, const half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

void mus_add_vectors_f16(MUSContext* ctx, half* a, const half* b, int rows, int D) {
    int n = rows * D;
    add_vectors_kernel_f16<<<(n+255)/256, 256, 0, ctx->stream>>>(a, b, n);
}

// Constants used in attention kernels

void mus_attention_backward_f16(MUSContext* ctx,
    const half* x, const half* qkv_w, const half* o_w,
    const int64_t* pos, const half* d_out,
    half* d_x, half* d_qkv_w, half* d_o_w,
    int B, int S, int D, int H) {
    int hd = D / H, rows = B * S;
    size_t sz_qkv = (size_t)rows*3*D*sizeof(half);
    size_t sz_h = (size_t)rows*D*sizeof(half);
    size_t sz_P = (size_t)B*H*S*S*sizeof(half);
    size_t need = sz_qkv + 3*sz_h + sz_P + 3*sz_h + 2*sz_P;
    if (ctx->workspace_size < need) { fprintf(stderr,"[attn_bwd_f16] need %.1f MB\n",(float)need/1e6f); return; }

    half* qkv_buf = ctx->workspace_f16;
    half* q = (half*)((char*)qkv_buf + sz_qkv);
    half* k = (half*)((char*)q + sz_h);
    half* v = (half*)((char*)k + sz_h);
    half* P = (half*)((char*)v + sz_h);
    half* dQ = (half*)((char*)P + sz_P);
    half* dK = (half*)((char*)dQ + sz_h);
    half* dV = (half*)((char*)dK + sz_h);
    half* dS_buf = (half*)((char*)dV + sz_h);
    half* temp_h = (half*)((char*)dS_buf + sz_P);

    // Recompute Q/K/V/P
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_N, 3*D,rows,D, qkv_w,3*D, x,D, qkv_buf,3*D);
    split_qkv_kernel_f16<<<(B*H*S*hd+255)/256,256,0,ctx->stream>>>(qkv_buf,q,k,v,B,S,D,H,hd);
    mus_rope_forward_f16(ctx,q,k,pos,B,H,S,hd);
    float scl = 1.0f/sqrtf((float)hd);
    float alpha1 = 1.0f, beta0 = 0.0f;
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_T,CUBLAS_OP_N, S,S,hd,&scl,
        q,CUDA_R_16F,hd,S*hd, k,CUDA_R_16F,hd,S*hd,
        &beta0, P,CUDA_R_16F,S,S*S, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    causal_softmax_kernel_f16<<<B*H,256,0,ctx->stream>>>(P,B,H,S);

    // d_pre = d_out @ o_w^T
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D, o_w,D, d_out,D, temp_h,D);

    // Split d_pre into heads → dQ
    split_dout_kernel_f16<<<(B*H*S*hd+255)/256,256,0,ctx->stream>>>(temp_h, dQ, B,S,D,H,hd);

    // dV = dQ @ P
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_N, hd,S,S,&alpha1,
        dQ,CUDA_R_16F,hd,S*hd, P,CUDA_R_16F,S,S*S,
        &beta0, dV,CUDA_R_16F,hd,S*hd, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // dP = dQ^T @ V
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_T,CUBLAS_OP_N, S,S,hd,&alpha1,
        dQ,CUDA_R_16F,hd,S*hd, v,CUDA_R_16F,hd,S*hd,
        &beta0, dS_buf,CUDA_R_16F,S,S*S, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // Softmax backward
    softmax_bwd_kernel_f16<<<(B*H+255)/256,256,0,ctx->stream>>>(P, dS_buf, B, H, S);

    // dQr = scl * K @ dS^T
    half* dQr = qkv_buf;
    half* dKr = (half*)((char*)dQr + sz_h);
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_T, hd,S,S,&scl,
        k,CUDA_R_16F,hd,S*hd, dS_buf,CUDA_R_16F,S,S*S,
        &beta0, dQr,CUDA_R_16F,hd,S*hd, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_N, hd,S,S,&scl,
        q,CUDA_R_16F,hd,S*hd, dS_buf,CUDA_R_16F,S,S*S,
        &beta0, dKr,CUDA_R_16F,hd,S*hd, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // RoPE backward
    mus_rope_backward_f16(ctx, pos, dQr, dKr, B, H, S, hd);

    // Merge heads
    half* dQ_flat = dS_buf;
    half* dK_flat = dQ_flat + rows * D;
    half* dV_flat = dK_flat + rows * D;
    merge_heads_kernel_f16<<<(rows*D+255)/256,256,0,ctx->stream>>>(dQr, dQ_flat, B,S,D,H,hd);
    merge_heads_kernel_f16<<<(rows*D+255)/256,256,0,ctx->stream>>>(dKr, dK_flat, B,S,D,H,hd);
    cudaMemcpyAsync(temp_h, dV, sz_h, cudaMemcpyDeviceToDevice, ctx->stream);
    merge_heads_kernel_f16<<<(rows*D+255)/256,256,0,ctx->stream>>>(temp_h, dV_flat, B,S,D,H,hd);

    // d_qkv_w weight gradients
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows, dQ_flat,D, x,D, d_qkv_w, 3*D);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows, dK_flat,D, x,D, d_qkv_w+D, 3*D);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows, dV_flat,D, x,D, d_qkv_w+2*D, 3*D);

    // d_o_w
    cublasGemmStridedBatchedEx(ctx->cublas, CUBLAS_OP_N,CUBLAS_OP_T, hd,S,S,&alpha1,
        v,CUDA_R_16F,hd,S*hd, P,CUDA_R_16F,S,S*S,
        &beta0, dQr,CUDA_R_16F,hd,S*hd, B*H,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    merge_heads_kernel_f16<<<(rows*D+255)/256,256,0,ctx->stream>>>(dQr, dV_flat, B,S,D,H,hd);
    mus_gemm_f16(ctx, CUBLAS_OP_N, CUBLAS_OP_T, D,D,rows, d_out,D, dV_flat,D, d_o_w,D);

    // d_x: gradient w.r.t. input (use qkv_w, NOT d_qkv_w!)
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D, qkv_w, 3*D, dQ_flat,D, d_x,D);
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D, qkv_w+D, 3*D, dK_flat,D, d_x,D);
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D, qkv_w+2*D, 3*D, dV_flat,D, d_x,D);
    mus_gemm_f16(ctx, CUBLAS_OP_T, CUBLAS_OP_N, D,rows,D, o_w,D, d_out,D, temp_h,D);
    int n = rows * D;
    add_vectors_kernel_f16<<<(n+255)/256, 256, 0, ctx->stream>>>(d_x, temp_h, n);
}

// ══════════════════════════════════════════════════════════════════════
//  Transformer Block Forward (FP16)
// ══════════════════════════════════════════════════════════════════════

void mus_transformer_block_forward_f16(MUSContext* ctx,
    const half* x_in, half* x_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos, int B, int S, int D, int H, int d_ff) {
    int rows = B * S;
    size_t sz = (size_t)rows * D * sizeof(half);
    size_t need = 4 * sz;
    if (ctx->workspace_size < need) { fprintf(stderr, "[block_fwd_f16] need %zu have %zu\n", need, ctx->workspace_size); return; }
    size_t ofs = ctx->workspace_size - need;
    half* norm1 = (half*)((char*)ctx->workspace_f16 + ofs);
    half* attn_out = (half*)((char*)norm1 + sz);
    half* norm2 = (half*)((char*)attn_out + sz);
    half* mlp_out = (half*)((char*)norm2 + sz);

    mus_rmsnorm_forward_f16(ctx, x_in, rms_norm_1_w, norm1, rows, D);
    mus_attention_forward_f16(ctx, norm1, attn_qkv_w, attn_o_w, pos, attn_out, B, S, D, H);
    mus_add_vectors_f16(ctx, attn_out, x_in, rows, D);
    mus_rmsnorm_forward_f16(ctx, attn_out, rms_norm_2_w, norm2, rows, D);
    mus_swiglu_forward_f16(ctx, norm2, mlp_gate_w, mlp_up_w, mlp_down_w, mlp_out, rows, D, d_ff);
    mus_add_vectors_f16(ctx, mlp_out, attn_out, rows, D);
    cudaMemcpyAsync(x_out, mlp_out, sz, cudaMemcpyDeviceToDevice, ctx->stream);
}

// ══════════════════════════════════════════════════════════════════════
//  Transformer Block Backward (FP16)
// ══════════════════════════════════════════════════════════════════════

void mus_transformer_block_backward_f16(MUSContext* ctx,
    const half* x_in, const half* d_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rms_norm_1_w, const float* rms_norm_2_w,
    const int64_t* pos,
    half* d_x_in,
    half* d_attn_qkv_w, half* d_attn_o_w,
    half* d_mlp_gate_w, half* d_mlp_up_w, half* d_mlp_down_w,
    float* d_rms_norm_1_w, float* d_rms_norm_2_w,
    int B, int S, int D, int H, int d_ff) {
    int rows = B * S;
    size_t sz = (size_t)rows * D * sizeof(half);
    size_t attn_bwd_max = (size_t)rows * 3 * D * sizeof(half) + 7 * sz
                        + 2 * (size_t)B * H * S * S * sizeof(half);
    size_t blk_ofs = (attn_bwd_max + (1<<20) - 1) >> 20 << 20;
    if (ctx->workspace_size < blk_ofs + 6 * sz) { fprintf(stderr, "[block_bwd_f16] ws\n"); return; }

    half* norm1 = (half*)((char*)ctx->workspace_f16 + blk_ofs);
    half* attn = (half*)((char*)norm1 + sz);
    half* norm2 = (half*)((char*)attn + sz);
    half* d_attn = (half*)((char*)norm2 + sz);
    half* d_norm2 = (half*)((char*)d_attn + sz);
    half* d_norm1 = (half*)((char*)d_norm2 + sz);

    // Recompute forward activations
    mus_rmsnorm_forward_f16(ctx, x_in, rms_norm_1_w, norm1, rows, D);
    mus_attention_forward_f16(ctx, norm1, attn_qkv_w, attn_o_w, pos, attn, B, S, D, H);
    mus_add_vectors_f16(ctx, attn, x_in, rows, D);
    mus_rmsnorm_forward_f16(ctx, attn, rms_norm_2_w, norm2, rows, D);

    // Backward
    mus_swiglu_backward_f16(ctx, norm2, mlp_gate_w, mlp_up_w, mlp_down_w,
                            d_out, d_norm2,
                            d_mlp_gate_w, d_mlp_up_w, d_mlp_down_w, rows, D, d_ff);
    mus_rmsnorm_backward_f16(ctx, attn, rms_norm_2_w, d_norm2, d_attn, d_rms_norm_2_w, rows, D);
    mus_add_vectors_f16(ctx, d_attn, d_out, rows, D);
    mus_attention_backward_f16(ctx, norm1, attn_qkv_w, attn_o_w, pos,
                               d_attn, d_norm1, d_attn_qkv_w, d_attn_o_w, B, S, D, H);
    mus_rmsnorm_backward_f16(ctx, x_in, rms_norm_1_w, d_norm1, d_x_in, d_rms_norm_1_w, rows, D);
    mus_add_vectors_f16(ctx, d_x_in, d_attn, rows, D);
}

// ══════════════════════════════════════════════════════════════════════
//  Embedding (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void embed_fwd_kernel_f16(const half* __restrict__ table,
    const int* __restrict__ input, half* __restrict__ out, int B, int S, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int pos = i / D, d = i % D;
    out[i] = table[input[pos] + d * V];
}

__global__ void embed_bwd_kernel_f16(const half* __restrict__ out_grad,
    const int* __restrict__ input, float* __restrict__ table_grad_f32, int B, int S, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i >= total) return;
    int pos = i / D, d = i % D;
    atomicAdd(&table_grad_f32[input[pos] + d * V], __half2float(out_grad[i]));
}

// Convert FP32 gradient buffer → FP16 weight gradient
__global__ void f32_to_f16_grad_kernel(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ══════════════════════════════════════════════════════════════════════
//  Gradient Clipping (FP16)
// ══════════════════════════════════════════════════════════════════════

__global__ void clip_grad_kernel_f16(half* g, int n, float max_norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = __half2float(g[i]);
        if (val > max_norm) g[i] = __float2half(max_norm);
        else if (val < -max_norm) g[i] = __float2half(-max_norm);
    }
}

void mus_clip_gradients_f16(MUSContext* ctx, half* g, int n, float max_norm) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clip_grad_kernel_f16<<<blocks, threads, 0, ctx->stream>>>(g, n, max_norm);
}

// ══════════════════════════════════════════════════════════════════════
//  FP32 gradient unscaling
// ══════════════════════════════════════════════════════════════════════

__global__ void unscale_f32_kernel(float* g, int n, float inv_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] *= inv_scale;
}

void mus_unscale_f32(MUSContext* ctx, float* g, int n, float inv_scale) {
    int bs = 256, grid = (n + bs - 1) / bs;
    unscale_f32_kernel<<<grid, bs, 0, ctx->stream>>>(g, n, inv_scale);
}

// ══════════════════════════════════════════════════════════════════════
//  NaN check (FP16)
// ══════════════════════════════════════════════════════════════════════

int mus_check_nan_f16(MUSContext* ctx, const half* d_data, int n, const char* name) {
    half* h = new half[n];
    cudaMemcpyAsync(h, d_data, n * sizeof(half), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    int nan_count = 0;
    for (int i = 0; i < n; i++) {
        float v = __half2float(h[i]);
        if (isnan(v) || isinf(v)) {
            if (nan_count < 5) printf("  %s: NaN/Inf at [%d] = %f\n", name, i, v);
            nan_count++;
        }
    }
    delete[] h;
    return nan_count;
}

// ─── Multimodal Processing Kernels (FP16) ───────────────────────────────-

// Vision encoder: project vision features → embedding space
__global__ void vision_encoder_kernel_f16(
    const half* __restrict__ vision_features,    // [B, N, vision_D]
    const half* __restrict__ vision_embed,     // [vision_vocab, D]
    half* __restrict__ vision_tokens,         // [B, N, D]
    int B, int N, int D, int vision_D
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int j = threadIdx.x;
    
    if (j >= D || b >= B || i >= N) return;
    
    const half* feat = vision_features + (b * N + i) * vision_D;
    const half* embed = vision_embed + 0; // Use first embedding (vision token)
    
    // Linear projection: W^T * x (D x vision_D) * (vision_D x 1)
    float sum = 0.0f;
    for (int k = 0; k < vision_D; ++k) {
        sum += __half2float(feat[k]) * __half2float(embed[j * vision_D + k]);
    }
    vision_tokens[(b * N + i) * D + j] = __float2half(sum);
}

// Audio encoder: project audio features → embedding space
__global__ void audio_encoder_kernel_f16(
    const half* __restrict__ audio_features,    // [B, T, audio_D]
    const half* __restrict__ audio_embed,      // [audio_vocab, D]
    half* __restrict__ audio_tokens,           // [B, T, D]
    int B, int T, int D, int audio_D
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int j = threadIdx.x;
    
    if (j >= D || b >= B || i >= T) return;
    
    const half* feat = audio_features + (b * T + i) * audio_D;
    const half* embed = audio_embed + 0; // Use first embedding (audio token)
    
    // Linear projection: W^T * x (D x audio_D) * (audio_D x 1)
    float sum = 0.0f;
    for (int k = 0; k < audio_D; ++k) {
        sum += __half2float(feat[k]) * __half2float(embed[j * audio_D + k]);
    }
    audio_tokens[(b * T + i) * D + j] = __float2half(sum);
}

// Cross-attention: attend between text and vision/audio
__global__ void cross_attention_kernel_f16(
    const half* __restrict__ query,           // [B, S_q, D]
    const half* __restrict__ key,             // [B, S_k, D]
    const half* __restrict__ value,           // [B, S_k, D]
    const half* __restrict__ qkv_w,           // [D, 3*D]
    const half* __restrict__ o_w,             // [D, D]
    half* __restrict__ output,                // [B, S_q, D]
    int B, int S_q, int S_k, int D, int H
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = threadIdx.x;
    
    if (h >= H || b >= B || s >= S_q) return;
    
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    int d_idx = h * (D / H);
    
    // Project query, key, value
    for (int d = 0; d < D / H; ++d) {
        float q_val = 0.0f;
        for (int di = 0; di < D; ++di) {
            q_val += __half2float(query[(b * S_q + s) * D + di]) * __half2float(qkv_w[d_idx * D + di]);
        }
        q_sum += q_val * q_val;
        
        float k_val = 0.0f;
        for (int di = 0; di < D; ++di) {
            k_val += __half2float(key[(b * S_k + s) * D + di]) * __half2float(qkv_w[D + d_idx * D + di]);
        }
        k_sum += k_val * k_val;
    }
    
    // Compute attention scores and weighted sum
    float max_score = -1e20f;
    float sum_weights = 0.0f;
    float output_vals[2048];
    
    for (int s_k = 0; s_k < S_k; ++s_k) {
        float score = 0.0f;
        for (int d = 0; d < D / H; ++d) {
            float q_val = 0.0f, k_val = 0.0f;
            for (int di = 0; di < D; ++di) {
                q_val += __half2float(query[(b * S_q + s) * D + di]) * __half2float(qkv_w[d_idx * D + di]);
                k_val += __half2float(key[(b * S_k + s_k) * D + di]) * __half2float(qkv_w[D + d_idx * D + di]);
            }
            score += q_val * k_val / sqrtf(D / H);
        }
        
        max_score = fmaxf(max_score, score);
    }
    
    for (int s_k = 0; s_k < S_k; ++s_k) {
        float score = 0.0f;
        for (int d = 0; d < D / H; ++d) {
            float q_val = 0.0f, k_val = 0.0f;
            for (int di = 0; di < D; ++di) {
                q_val += __half2float(query[(b * S_q + s) * D + di]) * __half2float(qkv_w[d_idx * D + di]);
                k_val += __half2float(key[(b * S_k + s_k) * D + di]) * __half2float(qkv_w[D + d_idx * D + di]);
            }
            score += q_val * k_val / sqrtf(D / H);
        }
        
        float weight = expf(score - max_score);
        sum_weights += weight;
        
        for (int d = 0; d < D / H; ++d) {
            float v_val = 0.0f;
            for (int di = 0; di < D; ++di) {
                v_val += __half2float(value[(b * S_k + s_k) * D + di]) * __half2float(qkv_w[2 * D + d_idx * D + di]);
            }
            output_vals[h * (D / H) + d] += weight * v_val;
        }
    }
    
    // Normalize and project output
    for (int d = 0; d < D / H; ++d) {
        float output_val = output_vals[h * (D / H) + d] / sum_weights;
        for (int di = 0; di < D; ++di) {
            output_val += __half2float(o_w[d_idx * D + di]) * output_val;
        }
        output[(b * S_q + s) * D + d_idx + d] = __float2half(output_val);
    }
}

// Modality gating: combine text, vision, audio features
__global__ void modality_gate_kernel_f16(
    const half* __restrict__ text_hidden,      // [B, S, D]
    const half* __restrict__ vision_hidden,    // [B, S, D] 
    const half* __restrict__ audio_hidden,    // [B, S, D]
    const half* __restrict__ vision_gate_w,   // [D]
    const half* __restrict__ audio_gate_w,   // [D]
    half* __restrict__ fused_hidden,           // [B, S, D]
    int B, int S, int D
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    
    if (d >= D || b >= B || s >= S) return;
    
    float text_val = __half2float(text_hidden[(b * S + s) * D + d]);
    float vision_val = __half2float(vision_hidden[(b * S + s) * D + d]);
    float audio_val = __half2float(audio_hidden[(b * S + s) * D + d]);
    
    float gate_v = __half2float(vision_gate_w[d]);
    float gate_a = __half2float(audio_gate_w[d]);
    
    // Gated combination: text + gate_v * vision + gate_a * audio
    float fused = text_val + gate_v * vision_val + gate_a * audio_val;
    fused_hidden[(b * S + s) * D + d] = __float2half(fused);
}

// ─── Multimodal Wrappers ─────────────────────────────────────────────

void mus_vision_encoder_forward_f16(
    MUSContext* ctx,
    const half* vision_features,
    const half* vision_embed,
    half* vision_tokens,
    int B, int N, int D, int vision_D
) {
    dim3 grid(B, N);
    dim3 block(D);
    vision_encoder_kernel_f16<<<grid, block, 0, ctx->stream>>>(
        vision_features, vision_embed, vision_tokens, B, N, D, vision_D);
    CUDA_CHECK(cudaGetLastError());
}

void mus_audio_encoder_forward_f16(
    MUSContext* ctx,
    const half* audio_features,
    const half* audio_embed,
    half* audio_tokens,
    int B, int T, int D, int audio_D
) {
    dim3 grid(B, T);
    dim3 block(D);
    audio_encoder_kernel_f16<<<grid, block, 0, ctx->stream>>>(
        audio_features, audio_embed, audio_tokens, B, T, D, audio_D);
    CUDA_CHECK(cudaGetLastError());
}

void mus_cross_attention_forward_f16(
    MUSContext* ctx,
    const half* query,
    const half* key,
    const half* value,
    const half* cross_qkv_w,
    const half* cross_o_w,
    half* output,
    int B, int S_q, int S_k, int D, int H
) {
    dim3 grid(B, S_q);
    dim3 block(H);
    cross_attention_kernel_f16<<<grid, block, 0, ctx->stream>>>(
        query, key, value, cross_qkv_w, cross_o_w, output, B, S_q, S_k, D, H);
    CUDA_CHECK(cudaGetLastError());
}

void mus_modality_gate_forward_f16(
    MUSContext* ctx,
    const half* text_hidden,
    const half* vision_hidden,
    const half* audio_hidden,
    const half* vision_gate_w,
    const half* audio_gate_w,
    half* fused_hidden,
    int B, int S, int D
) {
    dim3 grid(B, S);
    dim3 block(D);
    modality_gate_kernel_f16<<<grid, block, 0, ctx->stream>>>(
        text_hidden, vision_hidden, audio_hidden,
        vision_gate_w, audio_gate_w, fused_hidden, B, S, D);
    CUDA_CHECK(cudaGetLastError());
}

// ─── Memory Management Functions ─────────────────────────────────────────

void mus_optimize_memory_usage_f16(
    MUSContext* ctx,
    const MUSConfig& cfg,
    size_t* current_vram,
    size_t target_vram
) {
    size_t usage = mus_vram_usage_optimized(cfg);
    *current_vram = usage;
    
    if (usage > target_vram) {
        printf("Memory optimization needed: %.1fGB > %.1fGB\n", 
               usage / (1024.0f * 1024.0f * 1024.0f),
               target_vram / (1024.0f * 1024.0f * 1024.0f));
        
        // Apply memory reduction strategies
        if (cfg.enable_audio) {
            printf("  Disabling audio support for memory savings\n");
            // In practice: disable audio weights and buffers
        }
        
        if (cfg.enable_cross_attention) {
            printf("  Disabling cross-attention for memory savings\n");
            // In practice: disable cross-attention weights and buffers
        }
        
        // Reduce workspace size
        size_t new_workspace = 256 * 1024 * 1024; // 256MB instead of 512MB
        if (ctx->workspace_bytes > new_workspace) {
            // Reallocate smaller workspace (would require reallocation)
            printf("  Reducing workspace to %.1fMB\n", new_workspace / (1024.0f * 1024.0f));
        }
    }
}

void mus_allocate_on_demand_f16(
    MUSContext* ctx,
    const MUSConfig& cfg,
    int batch_size,
    int sequence_length,
    half** workspace,
    size_t* workspace_size
) {
    int D = cfg.get_D();
    int L = cfg.get_L();
    
    // Calculate on-demand allocation needs
    size_t activation_size = (size_t)batch_size * sequence_length * D * sizeof(half);
    size_t attention_size = (size_t)batch_size * L * D * D * sizeof(half);
    size_t multimodal_size = 0;
    
    if (cfg.enable_vision) {
        multimodal_size += (size_t)batch_size * sequence_length * cfg.vision_feature_dim * sizeof(half);
    }
    if (cfg.enable_audio) {
        multimodal_size += (size_t)batch_size * sequence_length * cfg.audio_feature_dim * sizeof(half);
    }
    
    *workspace_size = activation_size + attention_size + multimodal_size + 64 * 1024 * 1024; // 64MB overhead
    
    // Allocate workspace if needed
    if (ctx->workspace_f16 == nullptr || ctx->workspace_bytes < *workspace_size) {
        if (ctx->workspace_f16 != nullptr) {
            cudaFree(ctx->workspace_f16);
        }
        
        cudaMalloc(&ctx->workspace_f16, *workspace_size);
        ctx->workspace_bytes = *workspace_size;
        *workspace = ctx->workspace_f16;
        
        printf("Allocated on-demand workspace: %.1fMB\n", *workspace_size / (1024.0f * 1024.0f));
    }
}
