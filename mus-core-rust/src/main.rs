use mus_core_rust::*;
use std::ffi::c_void;


fn main() {
    let data_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../data/train_cache.bin".to_string());

    let cfg = MusConfig::new_500m();
    let D = cfg.D(); let L = cfg.L(); let H = cfg.H();
    let d_ff = cfg.D_ff(); let V = cfg.V();
    let B: i32 = 1; let S: i32 = 256; let rows = B * S;
    let ckpt = cfg.ckpt();
    let num_ckpt = L / ckpt + 2;
    let ws_size: usize = 64 * 1024 * 1024;

    println!("  MUS-Core Rust  D={} L={} H={} FF={} V={}  B={} S={}",
        D, L, H, d_ff, V, B, S);
    println!("  VRAM est: {:.2} / ~6 GB", cfg.vram_usage() as f64 / 1e9);

    let ctx = MusContext::new(ws_size);

    let data_file = std::fs::File::open(&data_path).expect("Cannot open data");
    let data_mmap = unsafe { memmap2::Mmap::map(&data_file).expect("mmap failed") };
    let N = data_mmap.len() / ((S as usize) * 4);
    println!("  Loaded {} samples", N);

    // ─── Model weights (FP16) ──────────────────────────────
    let w_embed = WeightBuf::alloc_f16((V * D) as usize, 0.02, 42);
    // Embedding gradient accumulates in FP32 (atomicAdd)
    let d_embed_f32 = DeviceBuffer::alloc_float((V * D) as usize);

    let mut w_qkv  = Vec::new(); let mut w_o   = Vec::new();
    let mut w_gate = Vec::new(); let mut w_up  = Vec::new(); let mut w_down = Vec::new();
    let mut w_rn1  = Vec::new(); let mut w_rn2  = Vec::new();
    let mut g_rn1  = Vec::new(); let mut g_rn2  = Vec::new();

    for l in 0..L {
        w_rn1.push(DeviceBuffer::alloc_float(D as usize));
        w_rn2.push(DeviceBuffer::alloc_float(D as usize));
        g_rn1.push(DeviceBuffer::alloc_float(D as usize));
        g_rn2.push(DeviceBuffer::alloc_float(D as usize));
        let li = l as usize;
        w_rn1[li].copy_from(&vec![1.0f32; D as usize]);
        w_rn2[li].copy_from(&vec![1.0f32; D as usize]);
        g_rn1[li].memset(0);
        g_rn2[li].memset(0);
        w_qkv .push(WeightBuf::alloc_f16((D * 3 * D) as usize, 0.02, 42 + l as u64));
        w_o   .push(WeightBuf::alloc_f16((D * D) as usize,      0.01, 43 + l as u64));
        w_gate.push(WeightBuf::alloc_f16((D * d_ff) as usize,   0.02, 44 + l as u64));
        w_up  .push(WeightBuf::alloc_f16((D * d_ff) as usize,   0.02, 45 + l as u64));
        w_down.push(WeightBuf::alloc_f16((d_ff * D) as usize,   0.01, 46 + l as u64));
    }

    let fn_w = DeviceBuffer::alloc_float(D as usize);
    fn_w.copy_from(&vec![1.0f32; D as usize]);
    let fn_g = DeviceBuffer::alloc_float(D as usize);
    fn_g.memset(0);

    let d_weights = DeviceBuffer::alloc_float(V as usize);
    {
        let mut hw = vec![0.0f32; V as usize];
        cfg.build_weight_table(&mut hw);
        d_weights.copy_from(&hw);
    }

    // Buffers
    let d_input_ids  = DeviceBuffer::alloc_int((B * S) as usize);
    let d_labels64   = DeviceBuffer::alloc_i64((B * S) as usize);
    let d_logits     = DeviceBuffer::alloc_half((rows * V) as usize);
    let d_trace      = DeviceBuffer::alloc_half((num_ckpt * rows * D) as usize);
    let d_fn_out     = DeviceBuffer::alloc_half((rows * D) as usize);
    let d_pos        = DeviceBuffer::alloc_i64((B * S) as usize);
    let d_loss       = DeviceBuffer::alloc_float(1);

    {
        let h_pos: Vec<i64> = (0..B*S).map(|i| (i % S) as i64).collect();
        d_pos.copy_from(&h_pos);
    }

    let mut h_input = vec![0i32; (B * S) as usize];
    let mut h_labels64 = vec![-1i64; (B * S) as usize];

    // ─── Training loop ──────────────────────────────────────────
    let num_epochs: i32 = 3;
    let steps_per_epoch = N as i32 / B;
    let mut global_step: i32 = 0;

    let base_lr: f32 = 3e-4;
    let warmup_steps: i32 = 500;
    let loss_scale: f32 = 1024.0;
    let weight_decay: f32 = 0.1;
    let inv_scale = 1.0 / loss_scale;

    println!("\n  ─── Training ───────────────────────────────");
    println!("  Epochs: {}  Steps/epoch: {}", num_epochs, steps_per_epoch);
    println!("  lr={:.0e}  warmup={}  wd={}  ls={:.0}\n", base_lr, warmup_steps, weight_decay, loss_scale);

    let mut best_loss = 1e10f64;

    for epoch in 0..num_epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss: f64 = 0.0;
        let mut total_valid: i32 = 0;

        for step in 0..steps_per_epoch {
            let start_idx = step * B;

            for b in 0..B {
                let idx = (start_idx + b) % (N as i32);
                let offset = (idx as usize) * (S as usize);
                let src: &[i32] = unsafe {
                    std::slice::from_raw_parts(
                        data_mmap.as_ptr().add(offset * 4) as *const i32, S as usize)
                };
                for s in 0..(S - 1) {
                    let pos = (b * S + s) as usize;
                    h_input[pos] = src[s as usize];
                    h_labels64[pos] = src[(s + 1) as usize] as i64;
                }
                let last = (b * S + (S - 1)) as usize;
                h_input[last] = 0;
                h_labels64[last] = -100;
            }

            d_input_ids.copy_from(&h_input);
            d_labels64.copy_from(&h_labels64);

            let ws_ptr = ctx.workspace() as *mut u16;
            let ws_size_half = ws_size / 2;

            // ── Forward ────────────────────────────────────────
            let ckpt_slot = |layer: i32| -> *mut u16 {
                let slot = (layer / ckpt) as usize;
                unsafe { d_trace.as_half().add(slot * (rows as usize) * (D as usize)) }
            };
            let last_slot = || -> *mut u16 {
                unsafe { d_trace.as_half().add((num_ckpt as usize - 1) * (rows as usize) * (D as usize)) }
            };

            embed_fwd(&ctx, w_embed.w.as_half(), d_input_ids.as_mut() as *const i32,
                ckpt_slot(0), B, S, D, V);

            let mut prev = ckpt_slot(0);
            for l in 0..L {
                let curr = if l == L - 1 {
                    last_slot()
                } else if l % ckpt == ckpt - 1 {
                    ckpt_slot(l + 1)
                } else {
                    unsafe { ws_ptr.add((l as usize % ckpt as usize) * (rows as usize) * (D as usize)) }
                };
                transformer_fwd(&ctx, prev, curr,
                    w_qkv[l as usize].w.as_half(),  w_o[l as usize].w.as_half(),
                    w_gate[l as usize].w.as_half(), w_up[l as usize].w.as_half(),  w_down[l as usize].w.as_half(),
                    w_rn1[l as usize].as_float() as *const f32,
                    w_rn2[l as usize].as_float() as *const f32,
                    d_pos.as_mut() as *const i64, B, S, D, H, d_ff);
                prev = curr;
            }

            rmsnorm_fwd(&ctx, prev, fn_w.as_float() as *const f32, d_fn_out.as_half(), rows, D);

            let embed_ptr = w_embed.w.as_half();
            gemm_f16(&ctx, gemm_op::N, gemm_op::T, V, rows, D,
                embed_ptr, V, d_fn_out.as_half(), rows, d_logits.as_half(), V, 1.0, 0.0);

            ctx.sync();

            let mut valid: i32 = 0;
            for i in 0..(B * S) as usize {
                if h_labels64[i] != -100 { valid += 1; }
            }

            let loss_val = ce_fwd(&ctx, d_logits.as_half(), d_labels64.as_mut() as *const i64,
                d_weights.as_float(), d_loss.as_float(), B, S, V);
            let step_loss = loss_val / valid.max(1) as f32;
            total_loss += loss_val as f64;
            total_valid += valid;
            global_step += 1;

            let current_lr = if global_step < warmup_steps {
                base_lr * (global_step as f32 / warmup_steps as f32)
            } else {
                base_lr
            };

            // ── Backward ───────────────────────────────────────
            d_embed_f32.memset(0);
            fn_g.memset(0);
            for l in 0..L {
                w_qkv[l as usize].zero_grad();
                w_o[l as usize].zero_grad();
                w_gate[l as usize].zero_grad();
                w_up[l as usize].zero_grad();
                w_down[l as usize].zero_grad();
                g_rn1[l as usize].memset(0);
                g_rn2[l as usize].memset(0);
            }

            let scale = loss_scale / valid.max(1) as f32;
            ce_bwd(&ctx, d_logits.as_half(), d_labels64.as_mut() as *const i64,
                d_weights.as_float(), scale, d_logits.as_half(), B, S, V);

            // LM head gradient: d_embed = d_logits @ fn_out^T
            let d_embed_temp = ws_ptr;
            gemm_f16(&ctx, gemm_op::N, gemm_op::T, V, D, rows,
                d_logits.as_half(), V, d_fn_out.as_half(), D,
                d_embed_temp, V, 1.0, 0.0);
            // Convert LM head gradient FP16 → FP32 into d_embed_f32
            cvt_f16_f32(&ctx, d_embed_temp, d_embed_f32.as_float(), V * D);

            device_memset(ws_ptr as *mut c_void, 0, ws_size);

            gemm_f16(&ctx, gemm_op::T, gemm_op::N, D, rows, V,
                embed_ptr, V, d_logits.as_half(), V, d_fn_out.as_half(), D, 1.0, 0.0);

            rmsnorm_bwd(&ctx, prev, fn_w.as_float() as *const f32,
                d_fn_out.as_half(), d_fn_out.as_half(), fn_g.as_float() as *mut f32, rows, D);
            let mut d_prev = d_fn_out.as_half();

            // Backward through layers (reverse, checkpoint replay)
            let ws_end = unsafe { ws_ptr.add(ws_size_half) };
            let tmp_area = unsafe { ws_end.sub(3 * rows as usize * D as usize) };
            let skip = tmp_area;
            let norm1_buf = unsafe { tmp_area.add(rows as usize * D as usize) };
            let attn_buf = unsafe { tmp_area.add(2 * rows as usize * D as usize) };
            let rows_d_bytes = (rows as usize) * (D as usize) * 2;

            for l in (0..L).rev() {
                let ckpt_base = (l / ckpt) * ckpt;
                let layer_in = ckpt_slot(ckpt_base);

                if l > ckpt_base {
                    for rp in ckpt_base..l {
                        let rp_in = if rp == ckpt_base { layer_in } else { skip };
                        memcpy_d2d(
                            skip as *mut _,
                            rp_in as *const _,
                            rows_d_bytes);
                        rmsnorm_fwd(&ctx, skip, w_rn1[rp as usize].as_float() as *const f32,
                            norm1_buf, rows, D);
                        attn_fwd(&ctx, norm1_buf,
                            w_qkv[rp as usize].w.as_half(), w_o[rp as usize].w.as_half(),
                            d_pos.as_mut() as *const i64, attn_buf, B, S, D, H);
                        add_vecs(&ctx, attn_buf, skip, rows, D);
                        rmsnorm_fwd(&ctx, attn_buf, w_rn2[rp as usize].as_float() as *const f32,
                            norm1_buf, rows, D);
                        swiglu_fwd(&ctx, norm1_buf,
                            w_gate[rp as usize].w.as_half(), w_up[rp as usize].w.as_half(),
                            w_down[rp as usize].w.as_half(), norm1_buf, rows, D, d_ff);
                        add_vecs(&ctx, norm1_buf, attn_buf, rows, D);
                    }
                }

                clip_grads(&ctx, d_prev, rows * D, 16.0 * loss_scale);
                transformer_bwd(&ctx, layer_in, d_prev,
                    w_qkv[l as usize].w.as_half(), w_o[l as usize].w.as_half(),
                    w_gate[l as usize].w.as_half(), w_up[l as usize].w.as_half(), w_down[l as usize].w.as_half(),
                    w_rn1[l as usize].as_float() as *const f32,
                    w_rn2[l as usize].as_float() as *const f32,
                    d_pos.as_mut() as *const i64,
                    d_fn_out.as_half(),
                    w_qkv[l as usize].g.as_half(), w_o[l as usize].g.as_half(),
                    w_gate[l as usize].g.as_half(), w_up[l as usize].g.as_half(), w_down[l as usize].g.as_half(),
                    g_rn1[l as usize].as_float() as *mut f32, g_rn2[l as usize].as_float() as *mut f32,
                    B, S, D, H, d_ff);
                clip_grads(&ctx, d_fn_out.as_half(), rows * D, 16.0 * loss_scale);
                d_prev = d_fn_out.as_half();
            }

            // Embedding backward — adds to d_embed_f32
            embed_bwd(&ctx, d_prev, d_input_ids.as_mut() as *const i32,
                d_embed_f32.as_float() as *mut f32, B, S, D, V);

            // Convert FP32 embed grad → FP16
            cvt_f32_f16(&ctx, d_embed_f32.as_float(), w_embed.g.as_half(), V * D);

            // ── Optimizer (FP16 AdamW with fixed v-accumulation) ─
            unscale_grads(&ctx, w_embed.g.as_half(), V * D, inv_scale);
            adamw_step(&ctx,
                w_embed.w.as_half(), w_embed.m.as_half(), w_embed.v.as_half(),
                w_embed.g.as_half(), V * D, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

            for l in 0..L as usize {
                let n_qkv = D * 3 * D;
                unscale_grads(&ctx, w_qkv[l].g.as_half(), n_qkv, inv_scale);
                adamw_step(&ctx,
                    w_qkv[l].w.as_half(), w_qkv[l].m.as_half(), w_qkv[l].v.as_half(),
                    w_qkv[l].g.as_half(), n_qkv, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                let n_o = D * D;
                unscale_grads(&ctx, w_o[l].g.as_half(), n_o, inv_scale);
                adamw_step(&ctx,
                    w_o[l].w.as_half(), w_o[l].m.as_half(), w_o[l].v.as_half(),
                    w_o[l].g.as_half(), n_o, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                let n_gate = D * d_ff;
                unscale_grads(&ctx, w_gate[l].g.as_half(), n_gate, inv_scale);
                adamw_step(&ctx,
                    w_gate[l].w.as_half(), w_gate[l].m.as_half(), w_gate[l].v.as_half(),
                    w_gate[l].g.as_half(), n_gate, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                let n_up = D * d_ff;
                unscale_grads(&ctx, w_up[l].g.as_half(), n_up, inv_scale);
                adamw_step(&ctx,
                    w_up[l].w.as_half(), w_up[l].m.as_half(), w_up[l].v.as_half(),
                    w_up[l].g.as_half(), n_up, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                let n_down = d_ff * D;
                unscale_grads(&ctx, w_down[l].g.as_half(), n_down, inv_scale);
                adamw_step(&ctx,
                    w_down[l].w.as_half(), w_down[l].m.as_half(), w_down[l].v.as_half(),
                    w_down[l].g.as_half(), n_down, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);
            }

            // ── Logging ────────────────────────────────────────
            let avg = total_loss / total_valid.max(1) as f64;
            println!("  ep {:2}/{} step {:4}  loss={:.4}  step={:.4}  lr={:.2e}",
                epoch + 1, num_epochs, step, avg, step_loss, current_lr);
        }

        let et = epoch_start.elapsed().as_secs_f64();
        let avg = total_loss / total_valid.max(1) as f64;
        if avg < best_loss { best_loss = avg; }
        println!("  ── epoch {:2}/{}  loss={:.4}  {:.1}s  best={:.4}\n",
            epoch + 1, num_epochs, avg, et, best_loss);
    }

    println!("  Done! Best loss: {:.4}\n", best_loss);
}
