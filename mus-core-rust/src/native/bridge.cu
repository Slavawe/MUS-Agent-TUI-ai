#include "mus_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════
//  C FFI bridge — wraps 49 C++ CUDA functions for Rust linkage
// ═══════════════════════════════════════════════════════════════

#ifdef __cplusplus
extern "C" {
#endif

// ─── Context ──────────────────────────────────────────────────
MUSContext* mus_create_context_ffi(size_t workspace_bytes) {
    return mus_create_context(workspace_bytes);
}
void mus_destroy_context_ffi(MUSContext* ctx) {
    mus_destroy_context(ctx);
}

// ─── Config constructors ──────────────────────────────────────
MUSConfig* mus_config_500m_ffi() {
    MUSConfig* cfg = new MUSConfig(get_500m_gtx1660ti_config());
    return cfg;
}
MUSConfig* mus_config_800m_ffi() {
    MUSConfig* cfg = new MUSConfig();
    *cfg = get_800m_gtx1660ti_config();
    return cfg;
}
void mus_config_destroy_ffi(MUSConfig* cfg) { delete cfg; }
int mus_config_get_D_ffi(const MUSConfig* cfg) { return cfg->get_D(); }
int mus_config_get_L_ffi(const MUSConfig* cfg) { return cfg->get_L(); }
int mus_config_get_H_ffi(const MUSConfig* cfg) { return cfg->get_H(); }
int mus_config_get_D_ff_ffi(const MUSConfig* cfg) { return cfg->get_D_ff(); }
int mus_config_get_V_ffi(const MUSConfig* cfg) { return cfg->get_V(); }
int mus_config_ckpt_ffi(const MUSConfig* cfg) { return cfg->checkpoint_interval; }

// ─── Memory management ────────────────────────────────────────
void* mus_malloc_ffi(size_t bytes) {
    void* ptr;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e != cudaSuccess) return NULL;
    return ptr;
}
void mus_free_ffi(void* ptr) { cudaFree(ptr); }
int mus_memcpy_h2d_ffi(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int mus_memcpy_d2h_ffi(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int mus_memset_ffi(void* ptr, int val, size_t bytes) {
    return (int)cudaMemset(ptr, val, bytes);
}
int mus_memcpy_d2d_ffi(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}
int mus_stream_sync_ffi(MUSContext* ctx) {
    return (int)cudaStreamSynchronize(ctx->stream);
}
size_t mus_vram_usage_ffi(const MUSConfig* cfg) {
    return mus_vram_usage_optimized(*cfg);
}
half* mus_get_workspace_ffi(MUSContext* ctx) {
    return ctx->workspace_f16;
}

// ─── Weight table (CPU) ───────────────────────────────────────
void build_weight_table_ffi(const MUSConfig* cfg, float* weights, int V) {
    build_weight_table(*cfg, weights, V);
}

// ─── GEMM (cuBLAS FP16) ──────────────────────────────────────
void mus_gemm_f16_ffi(MUSContext* ctx, int opA, int opB,
    int m, int n, int k, const half* A, int lda, const half* B, int ldb,
    half* C, int ldc, float alpha, float beta) {
    cublasOperation_t oA = (cublasOperation_t)opA;
    cublasOperation_t oB = (cublasOperation_t)opB;
    mus_gemm_f16(ctx, oA, oB, m, n, k, A, lda, B, ldb, C, ldc, alpha, beta);
}

// ─── FP32 training functions ─────────────────────────────────
float mus_ce_forward_ffi(MUSContext* ctx, const float* logits,
    const int64_t* labels, const float* weights, float* loss,
    int B, int S, int V) {
    return mus_weighted_ce_forward(ctx, logits, labels, weights, loss, B, S, V);
}
void mus_ce_backward_ffi(MUSContext* ctx, const float* logits,
    const int64_t* labels, const float* weights, float scale,
    float* dldx, int B, int S, int V) {
    mus_weighted_ce_backward(ctx, logits, labels, weights, scale, dldx, B, S, V);
}
void mus_rmsnorm_forward_ffi(MUSContext* ctx, const float* x,
    const float* w, float* y, int rows, int cols) {
    mus_rmsnorm_forward(ctx, x, w, y, rows, cols);
}
void mus_rmsnorm_backward_ffi(MUSContext* ctx, const float* x,
    const float* w, const float* dy, float* dx, float* dw,
    int rows, int cols) {
    mus_rmsnorm_backward(ctx, x, w, dy, dx, dw, rows, cols);
}
void mus_add_vectors_ffi(MUSContext* ctx, float* a, const float* b, int rows, int D) {
    mus_add_vectors(ctx, a, b, rows, D);
}
void mus_swiglu_forward_ffi(MUSContext* ctx, const float* x,
    const float* gate_w, const float* up_w, const float* down_w,
    float* out, int rows, int dim) {
    mus_swiglu_forward(ctx, x, gate_w, up_w, down_w, out, rows, dim);
}
void mus_swiglu_backward_ffi(MUSContext* ctx, const float* x,
    const float* gate_w, const float* up_w, const float* down_w,
    const float* d_out, float* d_x, float* d_gate_w, float* d_up_w, float* d_down_w,
    int rows, int dim) {
    mus_swiglu_backward(ctx, x, gate_w, up_w, down_w, d_out, d_x, d_gate_w, d_up_w, d_down_w, rows, dim);
}
void mus_attention_forward_ffi(MUSContext* ctx, const float* x,
    const float* qkv_w, const float* o_w, const int64_t* pos,
    float* out, int B, int S, int D, int H) {
    mus_attention_forward(ctx, x, qkv_w, o_w, pos, out, B, S, D, H);
}
void mus_attention_backward_ffi(MUSContext* ctx, const float* x,
    const float* qkv_w, const float* o_w, const int64_t* pos,
    const float* d_out, float* d_x, float* d_qkv_w, float* d_o_w,
    int B, int S, int D, int H) {
    mus_attention_backward(ctx, x, qkv_w, o_w, pos, d_out, d_x, d_qkv_w, d_o_w, B, S, D, H);
}
void mus_transformer_forward_ffi(MUSContext* ctx, const float* x_in, float* x_out,
    const float* attn_qkv_w, const float* attn_o_w,
    const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w,
    const float* rn1_w, const float* rn2_w, const int64_t* pos,
    int B, int S, int D, int H) {
    mus_transformer_block_forward(ctx, x_in, x_out, attn_qkv_w, attn_o_w,
        mlp_gate_w, mlp_up_w, mlp_down_w, rn1_w, rn2_w, pos, B, S, D, H);
}
void mus_transformer_backward_ffi(MUSContext* ctx, const float* x_in, const float* d_out,
    const float* attn_qkv_w, const float* attn_o_w,
    const float* mlp_gate_w, const float* mlp_up_w, const float* mlp_down_w,
    const float* rn1_w, const float* rn2_w, const int64_t* pos,
    float* d_x_in, float* d_attn_qkv_w, float* d_attn_o_w,
    float* d_mlp_gate_w, float* d_mlp_up_w, float* d_mlp_down_w,
    float* d_rn1_w, float* d_rn2_w, int B, int S, int D, int H) {
    mus_transformer_block_backward(ctx, x_in, d_out, attn_qkv_w, attn_o_w,
        mlp_gate_w, mlp_up_w, mlp_down_w, rn1_w, rn2_w, pos, d_x_in,
        d_attn_qkv_w, d_attn_o_w, d_mlp_gate_w, d_mlp_up_w, d_mlp_down_w,
        d_rn1_w, d_rn2_w, B, S, D, H);
}
void mus_clip_gradients_ffi(MUSContext* ctx, float* g, int n, float max_norm) {
    mus_clip_gradients(ctx, g, n, max_norm);
}
void mus_adamw_step_ffi(MUSContext* ctx, float* p, float* m, float* v,
    const float* g, int n, float lr, float b1, float b2,
    float eps, float wd, int step) {
    mus_adamw_step(ctx, p, m, v, g, n, lr, b1, b2, eps, wd, step);
}

// ─── FP16 training functions ─────────────────────────────────
void mus_convert_f16_to_f32_ffi(MUSContext* ctx, const half* src, float* dst, int n) {
    mus_convert_f16_to_f32(ctx, src, dst, n);
}
void mus_convert_f32_to_f16_ffi(MUSContext* ctx, const float* src, half* dst, int n) {
    mus_convert_f32_to_f16(ctx, src, dst, n);
}

void mus_embed_forward_ffi(MUSContext* ctx, const half* table,
    const int* input_ids, half* out, int B, int S, int D, int V) {
    int rows = B * S;
    embed_fwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(table, input_ids, out, B, S, D, V);
}
void mus_embed_backward_ffi(MUSContext* ctx, const half* d_output,
    const int* input_ids, float* d_embed, int B, int S, int D, int V) {
    int rows = B * S;
    embed_bwd_kernel_f16<<<(rows*D+255)/256, 256, 0, ctx->stream>>>(d_output, input_ids, d_embed, B, S, D, V);
}

float mus_ce_forward_f16_ffi(MUSContext* ctx, const half* logits,
    const int64_t* labels, const float* weights, float* loss,
    int B, int S, int V) {
    return mus_weighted_ce_forward_f16(ctx, logits, labels, weights, loss, B, S, V);
}
void mus_ce_backward_f16_ffi(MUSContext* ctx, const half* logits,
    const int64_t* labels, const float* weights, float scale,
    half* dldx, int B, int S, int V) {
    mus_weighted_ce_backward_f16(ctx, logits, labels, weights, scale, dldx, B, S, V);
}

void mus_rmsnorm_forward_f16_ffi(MUSContext* ctx, const half* x,
    const float* w, half* y, int rows, int cols) {
    mus_rmsnorm_forward_f16(ctx, x, w, y, rows, cols);
}
void mus_rmsnorm_backward_f16_ffi(MUSContext* ctx, const half* x,
    const float* w, const half* dy, half* dx, float* dw,
    int rows, int cols) {
    mus_rmsnorm_backward_f16(ctx, x, w, dy, dx, dw, rows, cols);
}
void mus_add_vectors_f16_ffi(MUSContext* ctx, half* a, const half* b, int rows, int D) {
    mus_add_vectors_f16(ctx, a, b, rows, D);
}

void mus_swiglu_forward_f16_ffi(MUSContext* ctx, const half* x,
    const half* gate_w, const half* up_w, const half* down_w,
    half* out, int rows, int dim, int d_ff) {
    mus_swiglu_forward_f16(ctx, x, gate_w, up_w, down_w, out, rows, dim, d_ff);
}
void mus_swiglu_backward_f16_ffi(MUSContext* ctx, const half* x,
    const half* gate_w, const half* up_w, const half* down_w,
    const half* d_out, half* d_x, half* d_gate_w, half* d_up_w, half* d_down_w,
    int rows, int dim, int d_ff) {
    mus_swiglu_backward_f16(ctx, x, gate_w, up_w, down_w, d_out, d_x, d_gate_w, d_up_w, d_down_w, rows, dim, d_ff);
}

void mus_attention_forward_f16_ffi(MUSContext* ctx, const half* x,
    const half* qkv_w, const half* o_w, const int64_t* pos,
    half* out, int B, int S, int D, int H) {
    mus_attention_forward_f16(ctx, x, qkv_w, o_w, pos, out, B, S, D, H);
}
void mus_attention_backward_f16_ffi(MUSContext* ctx, const half* x,
    const half* qkv_w, const half* o_w, const int64_t* pos,
    const half* d_out, half* d_x, half* d_qkv_w, half* d_o_w,
    int B, int S, int D, int H) {
    mus_attention_backward_f16(ctx, x, qkv_w, o_w, pos, d_out, d_x, d_qkv_w, d_o_w, B, S, D, H);
}

void mus_transformer_forward_f16_ffi(MUSContext* ctx, const half* x_in, half* x_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rn1_w, const float* rn2_w, const int64_t* pos,
    int B, int S, int D, int H, int d_ff) {
    mus_transformer_block_forward_f16(ctx, x_in, x_out, attn_qkv_w, attn_o_w,
        mlp_gate_w, mlp_up_w, mlp_down_w, rn1_w, rn2_w, pos, B, S, D, H, d_ff);
}
void mus_transformer_backward_f16_ffi(MUSContext* ctx, const half* x_in, const half* d_out,
    const half* attn_qkv_w, const half* attn_o_w,
    const half* mlp_gate_w, const half* mlp_up_w, const half* mlp_down_w,
    const float* rn1_w, const float* rn2_w, const int64_t* pos,
    half* d_x_in, half* d_attn_qkv_w, half* d_attn_o_w,
    half* d_mlp_gate_w, half* d_mlp_up_w, half* d_mlp_down_w,
    float* d_rn1_w, float* d_rn2_w, int B, int S, int D, int H, int d_ff) {
    mus_transformer_block_backward_f16(ctx, x_in, d_out, attn_qkv_w, attn_o_w,
        mlp_gate_w, mlp_up_w, mlp_down_w, rn1_w, rn2_w, pos, d_x_in,
        d_attn_qkv_w, d_attn_o_w, d_mlp_gate_w, d_mlp_up_w, d_mlp_down_w,
        d_rn1_w, d_rn2_w, B, S, D, H, d_ff);
}

void mus_clip_gradients_f16_ffi(MUSContext* ctx, half* g, int n, float max_norm) {
    mus_clip_gradients_f16(ctx, g, n, max_norm);
}
void mus_unscale_gradients_f16_ffi(MUSContext* ctx, half* g, int n, float inv_scale) {
    mus_unscale_gradients_f16(ctx, g, n, inv_scale);
}
void mus_adamw_step_f16_ffi(MUSContext* ctx, half* p, half* m, half* v,
    const half* g, int n, float lr, float b1, float b2,
    float eps, float wd, int step) {
    mus_adamw_step_f16(ctx, p, m, v, g, n, lr, b1, b2, eps, wd, step);
}

#ifdef __cplusplus
}
#endif
