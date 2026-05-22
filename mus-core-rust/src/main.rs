// ═══════════════════════════════════════════════════════════════
//  MUS-Core Rust — training binary
// ═══════════════════════════════════════════════════════════════

use mus_core_rust::*;
use std::ffi::c_void;
use std::io::{Read, Write};

fn alloc_f32_gpu(n: i32, val: f32) -> (DeviceBuffer, DeviceBuffer) {
    let d = DeviceBuffer::alloc_float(n as usize);
    let h: Vec<f32> = vec![val; n as usize];
    d.copy_from(&h);
    let g = DeviceBuffer::alloc_float(n as usize);
    g.memset(0);
    (d, g)
}

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

    // Load data via mmap
    let data_file = std::fs::File::open(&data_path).expect("Cannot open data");
    let data_mmap = unsafe { memmap2::Mmap::map(&data_file).expect("mmap failed") };
    let N = data_mmap.len() / ((S as usize) * 4);
    println!("  Loaded {} samples", N);

    // Model weights
    let w_embed = WeightBuf::alloc_f16((V * D) as usize, 0.02, 42);
    let d_embed_grad_f32 = DeviceBuffer::alloc_float((V * D) as usize);

    let mut w_qkv  = Vec::new(); let mut w_o   = Vec::new();
    let mut w_gate = Vec::new(); let mut w_up  = Vec::new(); let mut w_down = Vec::new();
    let mut w_rn1  = Vec::new(); let mut w_rn2  = Vec::new();
    let mut g_rn1  = Vec::new(); let mut g_rn2  = Vec::new();
    let mut rn1_m  = Vec::new(); let mut rn1_v  = Vec::new();
    let mut rn2_m  = Vec::new(); let mut rn2_v  = Vec::new();

    for l in 0..L {
        let (w, g) = alloc_f32_gpu(D, 1.0); w_rn1.push(w); g_rn1.push(g);
        let (w, g) = alloc_f32_gpu(D, 1.0); w_rn2.push(w); g_rn2.push(g);
        let m1 = DeviceBuffer::alloc_float(D as usize); m1.memset(0); rn1_m.push(m1);
        let v1 = DeviceBuffer::alloc_float(D as usize); v1.memset(0); rn1_v.push(v1);
        let m2 = DeviceBuffer::alloc_float(D as usize); m2.memset(0); rn2_m.push(m2);
        let v2 = DeviceBuffer::alloc_float(D as usize); v2.memset(0); rn2_v.push(v2);
        w_qkv .push(WeightBuf::alloc_f16((D * 3 * D) as usize, 0.02, 42 + l as u64));
        w_o   .push(WeightBuf::alloc_f16((D * D) as usize,      0.01, 43 + l as u64));
        w_gate.push(WeightBuf::alloc_f16((D * d_ff) as usize,   0.02, 44 + l as u64));
        w_up  .push(WeightBuf::alloc_f16((D * d_ff) as usize,   0.02, 45 + l as u64));
        w_down.push(WeightBuf::alloc_f16((d_ff * D) as usize,   0.01, 46 + l as u64));
    }

    let fn_m = DeviceBuffer::alloc_float(D as usize); fn_m.memset(0);
    let fn_v = DeviceBuffer::alloc_float(D as usize); fn_v.memset(0);

    let d_weights = DeviceBuffer::alloc_float(V as usize);
    {
        let mut hw = vec![0.0f32; V as usize];
        cfg.build_weight_table(&mut hw);
        d_weights.copy_from(&hw);
    }

    let (fn_w, fn_g) = alloc_f32_gpu(D, 1.0);

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

    // Cached host buffers
    let mut h_input = vec![0i32; (B * S) as usize];
    let mut h_labels64 = vec![-1i64; (B * S) as usize];

    // ─── Training loop ──────────────────────────────────────────
    let num_epochs: i32 = 3;
    let steps_per_epoch = N as i32 / B;
    let mut global_step: i32 = 0;

    let base_lr: f32 = 1e-4;
    let warmup_steps: i32 = 200;
    let loss_scale: f32 = 128.0;
    let weight_decay: f32 = 0.01;

    let ckpt_dir = std::env::var("MUS_CKPT_DIR").unwrap_or_else(|_| ".".to_string());
    let ckpt_path = format!("{}/mus_checkpoint.bin", ckpt_dir);
    if std::path::Path::new(&ckpt_path).exists() {
        let loaded_step = load_checkpoint(&ckpt_path, &ctx,
            &w_embed, &w_qkv, &w_o, &w_gate, &w_up, &w_down,
            &w_rn1, &w_rn2, &fn_w,
            L, D, d_ff, V);
        global_step = loaded_step;
        println!("  Resumed from step {}", global_step);
    }

    println!("\n  ─── Training ───────────────────────────────");
    println!("  Epochs: {}  Steps/epoch: {}", num_epochs, steps_per_epoch);
    println!("  lr=1e-4  warmup=200  wd=0.01  ls=128  ckpt={}\n", ckpt_path);

    let mut best_loss = 1e10f64;

    for epoch in 0..num_epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss: f64 = 0.0;
        let mut total_valid: i32 = 0;

        for step in 0..steps_per_epoch {
            let start_idx = step * B;

            // Prepare batch
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
            w_embed.zero_grad();
            d_embed_grad_f32.memset(0);
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
            cvt_f16_f32(&ctx, d_embed_temp, d_embed_grad_f32.as_float(), V * D);

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

                clip_grads(&ctx, d_prev, rows * D, 1.0 * loss_scale);
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
                clip_grads(&ctx, d_fn_out.as_half(), rows * D, 1.0 * loss_scale);
                d_prev = d_fn_out.as_half();
            }

            // Embedding backward
            embed_bwd(&ctx, d_prev, d_input_ids.as_mut() as *const i32,
                d_embed_grad_f32.as_float() as *mut f32, B, S, D, V);
            cvt_f32_f16(&ctx, d_embed_grad_f32.as_float() as *const f32, w_embed.g.as_half(), V * D);

            // ── Optimizer ─────────────────────────────────────────
            let inv_scale = 1.0 / loss_scale;

            unscale_grads(&ctx, w_embed.g.as_half(), V * D, inv_scale);
            clip_grads(&ctx, w_embed.g.as_half(), V * D, 1.0);
            adamw_step(&ctx, w_embed.w.as_half(), w_embed.m.as_half(), w_embed.v.as_half(),
                w_embed.g.as_half(), V * D, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

            for l in 0..L as usize {
                unscale_grads(&ctx, w_qkv[l].g.as_half(), D * 3 * D, inv_scale);
                clip_grads(&ctx, w_qkv[l].g.as_half(), D * 3 * D, 1.0);
                adamw_step(&ctx, w_qkv[l].w.as_half(), w_qkv[l].m.as_half(), w_qkv[l].v.as_half(),
                    w_qkv[l].g.as_half(), D * 3 * D, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                unscale_grads(&ctx, w_o[l].g.as_half(), D * D, inv_scale);
                clip_grads(&ctx, w_o[l].g.as_half(), D * D, 1.0);
                adamw_step(&ctx, w_o[l].w.as_half(), w_o[l].m.as_half(), w_o[l].v.as_half(),
                    w_o[l].g.as_half(), D * D, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                unscale_grads(&ctx, w_gate[l].g.as_half(), D * d_ff, inv_scale);
                clip_grads(&ctx, w_gate[l].g.as_half(), D * d_ff, 1.0);
                adamw_step(&ctx, w_gate[l].w.as_half(), w_gate[l].m.as_half(), w_gate[l].v.as_half(),
                    w_gate[l].g.as_half(), D * d_ff, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                unscale_grads(&ctx, w_up[l].g.as_half(), D * d_ff, inv_scale);
                clip_grads(&ctx, w_up[l].g.as_half(), D * d_ff, 1.0);
                adamw_step(&ctx, w_up[l].w.as_half(), w_up[l].m.as_half(), w_up[l].v.as_half(),
                    w_up[l].g.as_half(), D * d_ff, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);

                unscale_grads(&ctx, w_down[l].g.as_half(), d_ff * D, inv_scale);
                clip_grads(&ctx, w_down[l].g.as_half(), d_ff * D, 1.0);
                adamw_step(&ctx, w_down[l].w.as_half(), w_down[l].m.as_half(), w_down[l].v.as_half(),
                    w_down[l].g.as_half(), d_ff * D, current_lr, 0.9, 0.999, 1e-8, weight_decay, global_step);
            }

            let avg = total_loss / total_valid.max(1) as f64;
            if global_step % 500 == 0 {
                save_checkpoint(&ckpt_path, global_step, &ctx,
                    &w_embed, &w_qkv, &w_o, &w_gate, &w_up, &w_down,
                    &w_rn1, &w_rn2, &fn_w,
                    L, D, d_ff, V);
            }
            println!("  ep {:2}/{} step {:4}  loss={:.4}  step={:.4}  lr={:.2e}",
                epoch + 1, num_epochs, step, avg, step_loss, current_lr);
        }

        let et = epoch_start.elapsed().as_secs_f64();
        let avg = total_loss / total_valid.max(1) as f64;
        if avg < best_loss { best_loss = avg; }
        save_checkpoint(&ckpt_path, global_step, &ctx,
            &w_embed, &w_qkv, &w_o, &w_gate, &w_up, &w_down,
            &w_rn1, &w_rn2, &fn_w,
            L, D, d_ff, V);
        println!("  ── epoch {:2}/{}  loss={:.4}  {:.1}s  best={:.4}\n",
            epoch + 1, num_epochs, avg, et, best_loss);
    }

    println!("  Done! Best loss: {:.4}\n", best_loss);
}

// ─── Checkpoint save/load ──────────────────────────────────────

fn save_tensor<W: std::io::Write>(f: &mut W, name: &str, data: &[u8]) -> std::io::Result<()> {
    let name_bytes = name.as_bytes();
    f.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    f.write_all(name_bytes)?;
    f.write_all(&(data.len() as u64).to_le_bytes())?;
    f.write_all(data)?;
    Ok(())
}

fn load_tensor<R: std::io::Read>(f: &mut R, expected_name: &str) -> std::io::Result<Vec<u8>> {
    let mut name_len_buf = [0u8; 4];
    f.read_exact(&mut name_len_buf)?;
    let name_len = u32::from_le_bytes(name_len_buf) as usize;
    let mut name_buf = vec![0u8; name_len];
    f.read_exact(&mut name_buf)?;
    let name = String::from_utf8(name_buf).unwrap_or_default();
    assert_eq!(name, expected_name, "checkpoint tensor name mismatch: {} vs {}", name, expected_name);
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)?;
    let data_len = u64::from_le_bytes(size_buf) as usize;
    let mut data = vec![0u8; data_len];
    f.read_exact(&mut data)?;
    Ok(data)
}

fn save_checkpoint(path: &str, step: i32,
    ctx: &MusContext,
    w_embed: &WeightBuf,
    w_qkv: &[WeightBuf], w_o: &[WeightBuf],
    w_gate: &[WeightBuf], w_up: &[WeightBuf], w_down: &[WeightBuf],
    w_rn1: &[DeviceBuffer], w_rn2: &[DeviceBuffer],
    fn_w: &DeviceBuffer,
    L: i32, D: i32, d_ff: i32, V: i32)
{
    let tmp = format!("{}.tmp", path);
    let mut f = std::fs::File::create(&tmp).expect("Cannot create checkpoint");
    f.write_all(b"MUSCKPT").expect("write magic failed");
    f.write_all(&1u32.to_le_bytes()).expect("write version failed");
    f.write_all(&(step as i32).to_le_bytes()).expect("write step failed");

    let num_tensors: u32 = 3 + L as u32 * 7; // embed + fn_w + fn_g + layers × 7
    f.write_all(&num_tensors.to_le_bytes()).expect("write count failed");

    // Embedding
    eprintln!("    [save] embedding...");
    let n = (V * D) as usize;
    let mut host = vec![0u16; n];
    w_embed.w.copy_to(&mut host);
    save_tensor(&mut f, "embed.w", unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, n * 2)
    }).unwrap();

    // RMSNorm weights
    eprintln!("    [save] fn_w...");
    let n_fn = D as usize;
    let mut host_fn = vec![0f32; n_fn];
    fn_w.copy_to(&mut host_fn);
    save_tensor(&mut f, "fn_w", unsafe {
        std::slice::from_raw_parts(host_fn.as_ptr() as *const u8, n_fn * 4)
    }).unwrap();

    // Per-layer weights
    for layer in 0..L as usize {
        eprintln!("    [save] layer {}/{}...", layer + 1, L);
        let prefix = format!("l{}", layer);

        for (name, wb) in [("qkv", &w_qkv[layer]), ("o", &w_o[layer]),
            ("gate", &w_gate[layer]), ("up", &w_up[layer]), ("down", &w_down[layer])]
        {
            let n = wb.w.num_bytes() / 2; // half elements
            let mut host = vec![0u16; n];
            wb.w.copy_to(&mut host);
            save_tensor(&mut f, &format!("{}.{}", prefix, name), unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, n * 2)
            }).unwrap();
        }

        for (name, db) in [("rn1", &w_rn1[layer]), ("rn2", &w_rn2[layer])] {
            let n = db.num_bytes() / 4; // float elements
            let mut host = vec![0f32; n];
            db.copy_to(&mut host);
            save_tensor(&mut f, &format!("{}.{}", prefix, name), unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, n * 4)
            }).unwrap();
        }
    }

    std::fs::rename(&tmp, path).expect("checkpoint finalize failed");
    eprintln!("    [save] checkpoint saved to {}", path);
}

fn load_checkpoint(path: &str,
    ctx: &MusContext,
    w_embed: &WeightBuf,
    w_qkv: &[WeightBuf], w_o: &[WeightBuf],
    w_gate: &[WeightBuf], w_up: &[WeightBuf], w_down: &[WeightBuf],
    w_rn1: &[DeviceBuffer], w_rn2: &[DeviceBuffer],
    fn_w: &DeviceBuffer,
    L: i32, D: i32, d_ff: i32, V: i32) -> i32
{
    eprintln!("    [load] loading checkpoint from {}", path);
    let mut f = std::fs::File::open(path).expect("Cannot open checkpoint");

    let mut magic = [0u8; 7];
    f.read_exact(&mut magic).expect("read magic failed");
    assert_eq!(&magic, b"MUSCKPT", "bad checkpoint magic");

    let mut version_buf = [0u8; 4];
    f.read_exact(&mut version_buf).unwrap();
    let _version = u32::from_le_bytes(version_buf);

    let mut step_buf = [0u8; 4];
    f.read_exact(&mut step_buf).unwrap();
    let step = i32::from_le_bytes(step_buf);

    let mut count_buf = [0u8; 4];
    f.read_exact(&mut count_buf).unwrap();
    let _num = u32::from_le_bytes(count_buf);

    // Embedding
    let data = load_tensor(&mut f, "embed.w").unwrap();
    let n = (V * D) as usize * 2;
    assert_eq!(data.len(), n);
    w_embed.w.copy_from(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u16, n / 2)
    });

    // fn_w
    let data = load_tensor(&mut f, "fn_w").unwrap();
    let n_fn = D as usize * 4;
    assert_eq!(data.len(), n_fn);
    fn_w.copy_from(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, n_fn / 4)
    });

    // Per-layer
    for layer in 0..L as usize {
        let prefix = format!("l{}", layer);

        for (name, wb) in [("qkv", &w_qkv[layer]), ("o", &w_o[layer]),
            ("gate", &w_gate[layer]), ("up", &w_up[layer]), ("down", &w_down[layer])]
        {
            let data = load_tensor(&mut f, &format!("{}.{}", prefix, name)).unwrap();
            wb.w.copy_from(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
            });
        }

        for (name, db) in [("rn1", &w_rn1[layer]), ("rn2", &w_rn2[layer])] {
            let data = load_tensor(&mut f, &format!("{}.{}", prefix, name)).unwrap();
            db.copy_from(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            });
        }
    }

    eprintln!("    [load] loaded checkpoint at step {}", step);
    step
}
