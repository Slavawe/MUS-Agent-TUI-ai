// ═══════════════════════════════════════════════════════════════
//  MUS-Core Rust — safe FFI bindings for CUDA training on 1660Ti
// ═══════════════════════════════════════════════════════════════

use std::ffi::c_void;

#[allow(non_snake_case, non_camel_case_types, dead_code)]
mod ffi {
    use std::ffi::c_void;

    pub enum MusContextOpaque {}
    pub enum MusConfigOpaque {}

    extern "C" {
        pub fn mus_create_context_ffi(ws: usize) -> *mut MusContextOpaque;
        pub fn mus_destroy_context_ffi(ctx: *mut MusContextOpaque);
        pub fn mus_get_workspace_ffi(ctx: *mut MusContextOpaque) -> *mut c_void;

        pub fn mus_config_500m_ffi() -> *mut MusConfigOpaque;
        pub fn mus_config_destroy_ffi(cfg: *mut MusConfigOpaque);
        pub fn mus_config_get_D_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_config_get_L_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_config_get_H_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_config_get_D_ff_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_config_get_V_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_config_ckpt_ffi(cfg: *const MusConfigOpaque) -> i32;
        pub fn mus_vram_usage_ffi(cfg: *const MusConfigOpaque) -> usize;

        pub fn mus_malloc_ffi(bytes: usize) -> *mut c_void;
        pub fn mus_free_ffi(ptr: *mut c_void);
        pub fn mus_memcpy_h2d_ffi(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn mus_memcpy_d2h_ffi(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn mus_memset_ffi(ptr: *mut c_void, val: i32, bytes: usize) -> i32;
        pub fn mus_stream_sync_ffi(ctx: *mut MusContextOpaque) -> i32;
        pub fn mus_memcpy_d2d_ffi(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;

        pub fn build_weight_table_ffi(cfg: *const MusConfigOpaque, w: *mut f32, v: i32);

        pub fn mus_gemm_f16_ffi(ctx: *mut MusContextOpaque,
            opA: i32, opB: i32, m: i32, n: i32, k: i32,
            A: *const c_void, lda: i32, B: *const c_void, ldb: i32,
            C: *mut c_void, ldc: i32, alpha: f32, beta: f32);

        pub fn mus_embed_forward_ffi(ctx: *mut MusContextOpaque,
            table: *const c_void, input_ids: *const i32, out: *mut c_void,
            B: i32, S: i32, D: i32, V: i32);
        pub fn mus_embed_backward_ffi(ctx: *mut MusContextOpaque,
            d_output: *const c_void, input_ids: *const i32, d_embed: *mut f32,
            B: i32, S: i32, D: i32, V: i32);

        pub fn mus_ce_forward_f16_ffi(ctx: *mut MusContextOpaque,
            logits: *const c_void, labels: *const i64, weights: *const f32,
            loss: *mut f32, B: i32, S: i32, V: i32) -> f32;
        pub fn mus_ce_backward_f16_ffi(ctx: *mut MusContextOpaque,
            logits: *const c_void, labels: *const i64, weights: *const f32,
            scale: f32, dldx: *mut c_void, B: i32, S: i32, V: i32);

        pub fn mus_rmsnorm_forward_f16_ffi(ctx: *mut MusContextOpaque,
            x: *const c_void, w: *const f32, y: *mut c_void,
            rows: i32, cols: i32);
        pub fn mus_rmsnorm_backward_f16_ffi(ctx: *mut MusContextOpaque,
            x: *const c_void, w: *const f32, dy: *const c_void,
            dx: *mut c_void, dw: *mut f32, rows: i32, cols: i32);

        pub fn mus_add_vectors_f16_ffi(ctx: *mut MusContextOpaque,
            a: *mut c_void, b: *const c_void, rows: i32, D: i32);

        pub fn mus_swiglu_forward_f16_ffi(ctx: *mut MusContextOpaque,
            x: *const c_void, gw: *const c_void, uw: *const c_void, dw: *const c_void,
            out: *mut c_void, rows: i32, dim: i32, d_ff: i32);

        pub fn mus_attention_forward_f16_ffi(ctx: *mut MusContextOpaque,
            x: *const c_void, qkv_w: *const c_void, o_w: *const c_void,
            pos: *const i64, out: *mut c_void, B: i32, S: i32, D: i32, H: i32);

        pub fn mus_transformer_forward_f16_ffi(ctx: *mut MusContextOpaque,
            x_in: *const c_void, x_out: *mut c_void,
            aqkv: *const c_void, ao: *const c_void,
            mgw: *const c_void, muw: *const c_void, mdw: *const c_void,
            rn1: *const f32, rn2: *const f32, pos: *const i64,
            B: i32, S: i32, D: i32, H: i32, d_ff: i32);
        pub fn mus_transformer_backward_f16_ffi(ctx: *mut MusContextOpaque,
            x_in: *const c_void, d_out: *const c_void,
            aqkv: *const c_void, ao: *const c_void,
            mgw: *const c_void, muw: *const c_void, mdw: *const c_void,
            rn1: *const f32, rn2: *const f32, pos: *const i64,
            d_x: *mut c_void, d_aqkv: *mut c_void, d_ao: *mut c_void,
            d_mgw: *mut c_void, d_muw: *mut c_void, d_mdw: *mut c_void,
            d_rn1: *mut f32, d_rn2: *mut f32,
            B: i32, S: i32, D: i32, H: i32, d_ff: i32);

        pub fn mus_clip_gradients_f16_ffi(ctx: *mut MusContextOpaque,
            g: *mut c_void, n: i32, max_norm: f32);
        pub fn mus_unscale_gradients_f16_ffi(ctx: *mut MusContextOpaque,
            g: *mut c_void, n: i32, inv_scale: f32);
        pub fn mus_adamw_step_f16_ffi(ctx: *mut MusContextOpaque,
            p: *mut c_void, m: *mut c_void, v: *mut c_void,
            g: *const c_void, n: i32, lr: f32, b1: f32, b2: f32,
            eps: f32, wd: f32, step: i32);

        pub fn mus_convert_f16_to_f32_ffi(ctx: *mut MusContextOpaque,
            src: *const c_void, dst: *mut f32, n: i32);
        pub fn mus_convert_f32_to_f16_ffi(ctx: *mut MusContextOpaque,
            src: *const f32, dst: *mut c_void, n: i32);
    }
}

// ─── Safe wrappers ──────────────────────────────────────────

pub struct MusConfig { ptr: *mut ffi::MusConfigOpaque }
impl MusConfig {
    pub fn new_500m() -> Self {
        let ptr = unsafe { ffi::mus_config_500m_ffi() };
        assert!(!ptr.is_null()); MusConfig { ptr }
    }
    pub fn D(&self) -> i32 { unsafe { ffi::mus_config_get_D_ffi(self.ptr) } }
    pub fn L(&self) -> i32 { unsafe { ffi::mus_config_get_L_ffi(self.ptr) } }
    pub fn H(&self) -> i32 { unsafe { ffi::mus_config_get_H_ffi(self.ptr) } }
    pub fn D_ff(&self) -> i32 { unsafe { ffi::mus_config_get_D_ff_ffi(self.ptr) } }
    pub fn V(&self) -> i32 { unsafe { ffi::mus_config_get_V_ffi(self.ptr) } }
    pub fn ckpt(&self) -> i32 { unsafe { ffi::mus_config_ckpt_ffi(self.ptr) } }
    pub fn vram_usage(&self) -> usize { unsafe { ffi::mus_vram_usage_ffi(self.ptr) } }
    pub fn as_ptr(&self) -> *const ffi::MusConfigOpaque { self.ptr }
    pub fn build_weight_table(&self, w: &mut [f32]) {
        assert_eq!(w.len() as i32, self.V());
        unsafe { ffi::build_weight_table_ffi(self.ptr, w.as_mut_ptr(), self.V()); }
    }
}
impl Drop for MusConfig { fn drop(&mut self) { unsafe { ffi::mus_config_destroy_ffi(self.ptr); } } }

pub struct MusContext { ptr: *mut ffi::MusContextOpaque }
impl MusContext {
    pub fn new(ws: usize) -> Self {
        let ptr = unsafe { ffi::mus_create_context_ffi(ws) };
        assert!(!ptr.is_null()); MusContext { ptr }
    }
    pub fn as_ptr(&self) -> *mut ffi::MusContextOpaque { self.ptr }
    pub fn sync(&self) { unsafe { ffi::mus_stream_sync_ffi(self.ptr); } }
    pub fn workspace(&self) -> *mut u16 {
        unsafe { ffi::mus_get_workspace_ffi(self.ptr) as *mut u16 }
    }
}
impl Drop for MusContext { fn drop(&mut self) { unsafe { ffi::mus_destroy_context_ffi(self.ptr); } } }

pub struct DeviceBuffer { ptr: *mut c_void, bytes: usize }
impl DeviceBuffer {
    pub fn alloc(n: usize, elem_size: usize) -> Self {
        let bytes = n * elem_size;
        let ptr = unsafe { ffi::mus_malloc_ffi(bytes) };
        assert!(!ptr.is_null(), "alloc {} bytes failed", bytes);
        DeviceBuffer { ptr, bytes }
    }
    pub fn alloc_half(n: usize) -> Self { Self::alloc(n, 2) }
    pub fn alloc_float(n: usize) -> Self { Self::alloc(n, 4) }
    pub fn alloc_int(n: usize) -> Self { Self::alloc(n, 4) }
    pub fn alloc_i64(n: usize) -> Self { Self::alloc(n, 8) }
    pub fn copy_from<T>(&self, src: &[T]) {
        let bytes = src.len() * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "copy_from: {} > {}", bytes, self.bytes);
        unsafe { ffi::mus_memcpy_h2d_ffi(self.ptr, src.as_ptr() as *const c_void, bytes); }
    }
    pub fn copy_to<T>(&self, dst: &mut [T]) {
        let bytes = dst.len() * std::mem::size_of::<T>();
        assert_eq!(bytes, self.bytes);
        unsafe { ffi::mus_memcpy_d2h_ffi(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes); }
    }
    pub fn memset(&self, val: i32) { unsafe { ffi::mus_memset_ffi(self.ptr, val, self.bytes); } }
    pub fn as_mut(&self) -> *mut c_void { self.ptr }
    pub fn as_half(&self) -> *mut u16 { self.ptr as *mut u16 }
    pub fn as_float(&self) -> *mut f32 { self.ptr as *mut f32 }
    pub fn num_bytes(&self) -> usize { self.bytes }
}
impl Drop for DeviceBuffer { fn drop(&mut self) { unsafe { ffi::mus_free_ffi(self.ptr); } } }

pub fn memcpy_d2d(dst: *mut c_void, src: *const c_void, bytes: usize) {
    unsafe { ffi::mus_memcpy_d2d_ffi(dst, src, bytes); }
}

pub fn device_memset(ptr: *mut c_void, val: i32, bytes: usize) {
    unsafe { ffi::mus_memset_ffi(ptr, val, bytes); }
}

pub struct WeightBuf {
    pub w: DeviceBuffer, pub g: DeviceBuffer,
    pub m: DeviceBuffer, pub v: DeviceBuffer,
}
impl WeightBuf {
    pub fn alloc_f16(n: usize, stddev: f32, seed: u64) -> Self {
        use std::f32::consts::TAU;
        let w = DeviceBuffer::alloc_half(n);
        let g = DeviceBuffer::alloc_half(n);
        let m = DeviceBuffer::alloc_half(n);
        let v = DeviceBuffer::alloc_half(n);
        g.memset(0); m.memset(0); v.memset(0);
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut hw = vec![0u16; n];
        for i in 0..n {
            let u1 = rng.f32().max(1e-10f32);
            let u2 = rng.f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let val = (r * (TAU * u2).cos() * stddev) as f32;
            hw[i] = f32_to_f16(val);
        }
        w.copy_from(&hw);
        WeightBuf { w, g, m, v }
    }
    pub fn zero_grad(&self) { self.g.memset(0); }
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let nexp = exp - 127 + 15;
    if exp == 0 { return sign as u16; }
    if exp == 255 { return (sign | 0x7c00 | ((bits & 0x7fffff) >> 13) as u32) as u16; }
    if nexp >= 31 { return (sign | 0x7c00) as u16; }
    if nexp <= 0 { return sign as u16; }
    (sign | ((nexp as u32) << 10) | ((bits & 0x7fffff) >> 13)) as u16
}

// ─── Public wrapper functions ───────────────────────────────

pub mod gemm_op {
    pub const N: i32 = 0;
    pub const T: i32 = 1;
}

pub fn gemm_f16(ctx: &MusContext, opA: i32, opB: i32,
    m: i32, n: i32, k: i32, A: *const u16, lda: i32,
    B: *const u16, ldb: i32, C: *mut u16, ldc: i32,
    alpha: f32, beta: f32) {
    unsafe { ffi::mus_gemm_f16_ffi(ctx.as_ptr(), opA, opB, m, n, k,
        A as *const c_void, lda, B as *const c_void, ldb,
        C as *mut c_void, ldc, alpha, beta); }
}

pub fn embed_fwd(ctx: &MusContext, table: *const u16, ids: *const i32,
    out: *mut u16, B: i32, S: i32, D: i32, V: i32) {
    unsafe { ffi::mus_embed_forward_ffi(ctx.as_ptr(),
        table as *const c_void, ids, out as *mut c_void, B, S, D, V); }
}
pub fn embed_bwd(ctx: &MusContext, d_out: *const u16, ids: *const i32,
    d_embed: *mut f32, B: i32, S: i32, D: i32, V: i32) {
    unsafe { ffi::mus_embed_backward_ffi(ctx.as_ptr(),
        d_out as *const c_void, ids, d_embed, B, S, D, V); }
}

pub fn rmsnorm_fwd(ctx: &MusContext, x: *const u16, w: *const f32, y: *mut u16, rows: i32, cols: i32) {
    unsafe { ffi::mus_rmsnorm_forward_f16_ffi(ctx.as_ptr(),
        x as *const c_void, w, y as *mut c_void, rows, cols); }
}
pub fn rmsnorm_bwd(ctx: &MusContext, x: *const u16, w: *const f32,
    dy: *const u16, dx: *mut u16, dw: *mut f32, rows: i32, cols: i32) {
    unsafe { ffi::mus_rmsnorm_backward_f16_ffi(ctx.as_ptr(),
        x as *const c_void, w, dy as *const c_void, dx as *mut c_void, dw, rows, cols); }
}

pub fn ce_fwd(ctx: &MusContext, logits: *const u16, labels: *const i64,
    weights: *const f32, loss: *mut f32, B: i32, S: i32, V: i32) -> f32 {
    unsafe { ffi::mus_ce_forward_f16_ffi(ctx.as_ptr(),
        logits as *const c_void, labels, weights, loss, B, S, V) }
}
pub fn ce_bwd(ctx: &MusContext, logits: *const u16, labels: *const i64,
    weights: *const f32, scale: f32, dldx: *mut u16, B: i32, S: i32, V: i32) {
    unsafe { ffi::mus_ce_backward_f16_ffi(ctx.as_ptr(),
        logits as *const c_void, labels, weights, scale, dldx as *mut c_void, B, S, V); }
}

pub fn add_vecs(ctx: &MusContext, a: *mut u16, b: *const u16, rows: i32, D: i32) {
    unsafe { ffi::mus_add_vectors_f16_ffi(ctx.as_ptr(),
        a as *mut c_void, b as *const c_void, rows, D); }
}

pub fn swiglu_fwd(ctx: &MusContext, x: *const u16, gw: *const u16, uw: *const u16,
    dw: *const u16, out: *mut u16, rows: i32, dim: i32, d_ff: i32) {
    unsafe { ffi::mus_swiglu_forward_f16_ffi(ctx.as_ptr(),
        x as *const c_void, gw as *const c_void, uw as *const c_void,
        dw as *const c_void, out as *mut c_void, rows, dim, d_ff); }
}

pub fn attn_fwd(ctx: &MusContext, x: *const u16, qkv_w: *const u16,
    o_w: *const u16, pos: *const i64, out: *mut u16, B: i32, S: i32, D: i32, H: i32) {
    unsafe { ffi::mus_attention_forward_f16_ffi(ctx.as_ptr(),
        x as *const c_void, qkv_w as *const c_void, o_w as *const c_void,
        pos, out as *mut c_void, B, S, D, H); }
}

pub fn transformer_fwd(ctx: &MusContext, x_in: *const u16, x_out: *mut u16,
    aqkv: *const u16, ao: *const u16, mgw: *const u16, muw: *const u16, mdw: *const u16,
    rn1: *const f32, rn2: *const f32, pos: *const i64,
    B: i32, S: i32, D: i32, H: i32, d_ff: i32) {
    unsafe { ffi::mus_transformer_forward_f16_ffi(ctx.as_ptr(),
        x_in as *const c_void, x_out as *mut c_void,
        aqkv as *const c_void, ao as *const c_void,
        mgw as *const c_void, muw as *const c_void, mdw as *const c_void,
        rn1, rn2, pos, B, S, D, H, d_ff); }
}

pub fn transformer_bwd(ctx: &MusContext, x_in: *const u16, d_out: *const u16,
    aqkv: *const u16, ao: *const u16, mgw: *const u16, muw: *const u16, mdw: *const u16,
    rn1: *const f32, rn2: *const f32, pos: *const i64,
    d_x: *mut u16, d_aqkv: *mut u16, d_ao: *mut u16,
    d_mgw: *mut u16, d_muw: *mut u16, d_mdw: *mut u16,
    d_rn1: *mut f32, d_rn2: *mut f32,
    B: i32, S: i32, D: i32, H: i32, d_ff: i32) {
    unsafe { ffi::mus_transformer_backward_f16_ffi(ctx.as_ptr(),
        x_in as *const c_void, d_out as *const c_void,
        aqkv as *const c_void, ao as *const c_void,
        mgw as *const c_void, muw as *const c_void, mdw as *const c_void,
        rn1, rn2, pos,
        d_x as *mut c_void, d_aqkv as *mut c_void, d_ao as *mut c_void,
        d_mgw as *mut c_void, d_muw as *mut c_void, d_mdw as *mut c_void,
        d_rn1, d_rn2, B, S, D, H, d_ff); }
}

pub fn adamw_step(ctx: &MusContext, p: *mut u16, m: *mut u16, v: *mut u16,
    g: *const u16, n: i32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: i32) {
    unsafe { ffi::mus_adamw_step_f16_ffi(ctx.as_ptr(),
        p as *mut c_void, m as *mut c_void, v as *mut c_void,
        g as *const c_void, n, lr, b1, b2, eps, wd, step); }
}

pub fn clip_grads(ctx: &MusContext, g: *mut u16, n: i32, max_norm: f32) {
    unsafe { ffi::mus_clip_gradients_f16_ffi(ctx.as_ptr(),
        g as *mut c_void, n, max_norm); }
}
pub fn unscale_grads(ctx: &MusContext, g: *mut u16, n: i32, inv_scale: f32) {
    unsafe { ffi::mus_unscale_gradients_f16_ffi(ctx.as_ptr(),
        g as *mut c_void, n, inv_scale); }
}
pub fn cvt_f16_f32(ctx: &MusContext, src: *const u16, dst: *mut f32, n: i32) {
    unsafe { ffi::mus_convert_f16_to_f32_ffi(ctx.as_ptr(),
        src as *const c_void, dst, n); }
}
pub fn cvt_f32_f16(ctx: &MusContext, src: *const f32, dst: *mut u16, n: i32) {
    unsafe { ffi::mus_convert_f32_to_f16_ffi(ctx.as_ptr(),
        src, dst as *mut c_void, n); }
}
