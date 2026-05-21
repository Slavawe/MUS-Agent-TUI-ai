#pragma once

#include "mus_cuda.h"
#include <string>
#include <cstdio>

// ══════════════════════════════════════════════════════════════════════
//  Модельные пресеты — архитектура, память, рекомендации по GPU
// ══════════════════════════════════════════════════════════════════════

struct ModelPreset {
    std::string name;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int head_dim;
    int ffn_dim;        // SwiGLU intermediate
    int vocab_size;     // полный словарь (BPE + AER + ASCII + multimodal)
    float base_lr;      // рекомендуемый стартовый LR
    int warmup_steps;
    float init_std;     // stddev для инициализации весов

    // Параметры обучения
    int batch_size;     // для 6GB VRAM
    int seq_len;
    float loss_scale_init;

    // Память (FP16, приблизительно)
    double param_count() const {
        double embed = (double)vocab_size * hidden_dim;
        double per_layer = 4.0 * hidden_dim * hidden_dim
                         + 3.0 * hidden_dim * ffn_dim
                         + 2.0 * hidden_dim;
        double total = embed + num_layers * per_layer + hidden_dim;
        return total;
    }

    double mem_weights_mb() const { return param_count() * 2.0 / 1e6; }
    double mem_optim_mb() const { return param_count() * 2.0 * 2.0 / 1e6; } // m+v in FP16
    double mem_total_mb() const { return mem_weights_mb() * 2 + mem_optim_mb() + 512; }

    void print() const {
        printf("  Preset: %s\n", name.c_str());
        printf("    D=%d L=%d H=%d hd=%d D_ff=%d V=%d\n",
               hidden_dim, num_layers, num_heads, head_dim, ffn_dim, vocab_size);
        printf("    Parameters: %.0fM\n", param_count() / 1e6);
        printf("    Memory (FP16): weights=%.0fMB grads=%.0fMB optim=%.0fMB total≈%.0fMB\n",
               mem_weights_mb(), mem_weights_mb(), mem_optim_mb(), mem_total_mb());
        printf("    Recommended GPU VRAM: >=%.0fGB\n", mem_total_mb() / 1024 + 0.5);
    }
};

// ─── Пресеты ──────────────────────────────────────────────────────────

inline ModelPreset preset_500m() {
    return {
        "MUS-500M-FP16",
        1280,   // hidden_dim
        22,     // num_layers
        20,     // num_heads
        64,     // head_dim
        3456,   // ffn_dim (8/3 × 1280 ≈ 3413 → rounded to 3456)
        48000,  // vocab_size
        1e-4f,  // base_lr
        200,    // warmup_steps
        0.02f,  // init_std
        1,      // batch_size (6GB)
        256,    // seq_len
        128.0f  // loss_scale_init
    };
}

inline ModelPreset preset_1b() {
    return {
        "MUS-1B-FP16",
        1600,   // hidden_dim
        30,     // num_layers
        25,     // num_heads
        64,     // head_dim
        4352,   // ffn_dim (8/3 × 1600 ≈ 4267 → rounded to 4352)
        48000,  // vocab_size
        8e-5f,  // base_lr (чуть ниже для большего размера)
        400,    // warmup_steps (дольше для стабильности)
        0.015f, // init_std (чуть ниже для глубокой сети)
        1,      // batch_size (≈9.5GB в FP16)
        256,    // seq_len
        256.0f  // loss_scale_init
    };
}

// Применить пресет к MUSConfig
inline MUSConfig apply_preset(const ModelPreset& p) {
    MUSConfig cfg;
    cfg.hidden_dim = p.hidden_dim;
    cfg.num_layers = p.num_layers;
    cfg.num_heads = p.num_heads;
    cfg.head_dim = p.head_dim;
    cfg.ffn_dim = p.ffn_dim;
    cfg.vocab_size = p.vocab_size;
    return cfg;
}

// ══════════════════════════════════════════════════════════════════════
//  Мультимедийные токен-диапазоны (общие для всех моделей)
// ══════════════════════════════════════════════════════════════════════

// Токенизация:
//   [0, 2000]     AER (Audio/Event/Registry) — ID сущностей
//   [2001, 2077]  ASCII-Vision — символы яркостной палитры (77 chars)
//   [2078, 2100]  Audio-токены — дискретные кванты звуковой волны (23)
//   [2101, 2200]  Multimodal-теги — маркеры модальности (100 тегов)
//   [2201, 48000] BPE-словарь — основной текстовый словарь (≈45800 токенов)

constexpr int AER_START          = 0;
constexpr int AER_END            = 2000;

constexpr int ASCII_START        = 2001;  // ' ' (space)
constexpr int ASCII_END          = 2077;  // '~' (last printable)

constexpr int AUDIO_START        = 2078;  // дискретные аудио-кванты
constexpr int AUDIO_END          = 2100;  // 23 quanta

constexpr int MM_START           = 2101;  // мультимодальные теги
constexpr int MM_END             = 2200;

// ─── Мультимодальные теги ─────────────────────────────────────────────
constexpr int TAG_VISION_START   = 2101;  // <vision_start>
constexpr int TAG_VISION_END     = 2102;  // <vision_end>
constexpr int TAG_FRAME_SEP      = 2103;  // <frame_sep> (для видео)
constexpr int TAG_MODE_PHOTO     = 2104;  // <mode:photo>
constexpr int TAG_MODE_GRAPH     = 2105;  // <mode:graph>
constexpr int TAG_AUDIO_START    = 2110;  // <audio_start>
constexpr int TAG_AUDIO_END      = 2111;  // <audio_end>
constexpr int TAG_SYSTEM         = 2150;  // <system>
constexpr int TAG_USER           = 2151;  // <user>
constexpr int TAG_ASSISTANT      = 2152;  // <assistant>
constexpr int TAG_THINK          = 2160;  // <think>
constexpr int TAG_ACTION         = 2170;  // <action> — AER action trigger

// ─── ASCII палитра (стандартная 70 символов) ───────────────────────────
inline const char* default_ascii_palette() {
    return " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#8&@%";
}

inline int palette_len() { return 70; }
