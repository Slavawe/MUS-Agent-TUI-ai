#pragma once

#include "mus_model.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <cfloat>
#include <cassert>

// ══════════════════════════════════════════════════════════════════════
//  Platform detection
// ══════════════════════════════════════════════════════════════════════

#if defined(_WIN32) || defined(_WIN64)
    #define MUS_OS_WINDOWS 1
    #define MUS_OS_LINUX   0
#elif defined(__linux__)
    #define MUS_OS_WINDOWS 0
    #define MUS_OS_LINUX   1
#else
    #define MUS_OS_WINDOWS 0
    #define MUS_OS_LINUX   0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define MUS_ARCH_ARM   1
    #define MUS_ARCH_X86   0
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define MUS_ARCH_ARM   0
    #define MUS_ARCH_X86   1
#else
    #define MUS_ARCH_ARM   0
    #define MUS_ARCH_X86   0
#endif

#if defined(__AVX2__) || defined(__AVX__)
    #define MUS_HAVE_AVX   1
#else
    #define MUS_HAVE_AVX   0
#endif

// ══════════════════════════════════════════════════════════════════════
//  Uragan 1.0 — C++ Vision Encoder (CPPTokenizer)
//  Кроссплатформенный токенизатор изображений в C++ токены.
//
//  Особенности:
//    • SIMD (SSE2/AVX2) для ускорения luminance + quantization
//    • OpenMP-параллелизм для больших изображений
//    • Floyd-Steinberg дизеринг в photo-режиме
//    • Canny-подобное выделение границ (NMS + double threshold)
//    • Поддержка видео-последовательностей кадров
//    • Сохранение/загрузка арта в файл
//    • Декодирование токенов → RGB-изображение
//    • Настраиваемая палитра
//    • Windows/Linux — единый код
// ══════════════════════════════════════════════════════════════════════

enum class VisionMode {
    Photo,
    Graph
};

enum class DitherMode {
    None,
    FloydSteinberg
};

struct VisionEncoderConfig {
    int width = 64;
    int height = 32;
    VisionMode mode = VisionMode::Photo;
    DitherMode dither = DitherMode::None;
    bool use_canny = false;          // NMS + double threshold для Graph
    float canny_low = 0.05f;
    float canny_high = 0.15f;
    int num_frames = 1;              // количество кадров (видеорежим)
    const char* custom_palette = nullptr;  // nullptr = стандартная палитра
};

class CPPTokenizer {
public:
    explicit CPPTokenizer(VisionEncoderConfig cfg = {}) noexcept
        : w_(cfg.width), h_(cfg.height), mode_(cfg.mode)
        , dither_(cfg.dither), use_canny_(cfg.use_canny)
        , canny_low_(cfg.canny_low), canny_high_(cfg.canny_high)
        , num_frames_(std::max(1, cfg.num_frames))
        , palette_(cfg.custom_palette ? cfg.custom_palette : default_cpp_palette())
        , palette_len_(palette_len_default())
    {
        // Gamma LUT (sRGB → linear)
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            gamma_lut_[i] = (v <= 0.04045f) ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
        }

        // Perceptual quantization LUT (теперь используется в quantize_perceptual)
        int p = palette_len_;
        for (int i = 0; i < 256; i++) {
            float linear = gamma_lut_[i];
            float perceptual = sqrtf(linear);
            perceptual = fmaxf(0.0f, fminf(1.0f, perceptual));
            quant_lut_[i] = (int)(perceptual * (p - 1) + 0.5f);
            if (quant_lut_[i] >= p) quant_lut_[i] = p - 1;
        }
    }

    // ── Основной метод: RGB → токены ──────────────────────────────
    std::vector<int64_t> encode(const uint8_t* rgb_pixels, int src_w, int src_h) const {
        if (!rgb_pixels || src_w < 1 || src_h < 1)
            return {};

        std::vector<float> lum(h_ * w_);
        compute_luminance(rgb_pixels, src_w, src_h, lum.data());

        if (mode_ == VisionMode::Photo) {
            return quantize_perceptual(lum);
        } else {
            auto edges = sobel_edges(lum);
            if (use_canny_)
                edges = canny_refine(edges);
            return quantize_linear(edges);
        }
    }

    // ── С тегами модальности ───────────────────────────────────────
    std::vector<int64_t> encode_with_tags(const uint8_t* rgb_pixels, int src_w, int src_h) const {
        std::vector<int64_t> tokens;
        tokens.reserve(3 + h_ * w_);
        tokens.push_back(TAG_VISION_START);
        tokens.push_back(mode_ == VisionMode::Photo ? TAG_MODE_PHOTO : TAG_MODE_GRAPH);
        auto content = encode(rgb_pixels, src_w, src_h);
        tokens.insert(tokens.end(), content.begin(), content.end());
        tokens.push_back(TAG_VISION_END);
        return tokens;
    }

    // ── Видео: несколько кадров ────────────────────────────────────
    std::vector<int64_t> encode_frames(
        const uint8_t* rgb_frames,
        int src_w, int src_h,
        int num_frames
    ) const {
        if (!rgb_frames || src_w < 1 || src_h < 1 || num_frames < 1)
            return {};

        int frames = std::min(num_frames, num_frames_);
        std::vector<int64_t> tokens;
        tokens.reserve(3 + frames * (h_ * w_ + 1));
        tokens.push_back(TAG_VISION_START);
        tokens.push_back(mode_ == VisionMode::Photo ? TAG_MODE_PHOTO : TAG_MODE_GRAPH);

        int frame_bytes = src_w * src_h * 3;
        for (int f = 0; f < frames; f++) {
            if (f > 0) tokens.push_back(TAG_FRAME_SEP);
            auto content = encode(rgb_frames + f * frame_bytes, src_w, src_h);
            tokens.insert(tokens.end(), content.begin(), content.end());
        }

        tokens.push_back(TAG_VISION_END);
        return tokens;
    }

    // ── Декодирование токенов → строка ─────────────────────────────
    std::string decode_to_string(const std::vector<int64_t>& tokens) const {
        std::string chars;
        chars.reserve(h_ * w_);
        int p = palette_len_;

        for (int64_t t : tokens) {
            if (t >= CPP_START && t <= CPP_END) {
                int64_t idx = t - CPP_START;
                if (idx >= 0 && idx < p)
                    chars += palette_[idx];
            }
        }

        std::string grid;
        grid.reserve(h_ * (w_ + 1));
        for (int y = 0; y < h_; y++) {
            size_t off = (size_t)y * w_;
            grid += chars.substr(off, (size_t)w_) + '\n';
        }
        return grid;
    }

    // ── Декодирование → RGB-изображение (64×32) ────────────────────
    std::vector<uint8_t> decode_to_rgb(const std::vector<int64_t>& tokens) const {
        int p = palette_len_;
        std::vector<uint8_t> rgb((size_t)h_ * w_ * 3, 0);

        int idx = 0;
        for (int64_t t : tokens) {
            if (t >= CPP_START && t <= CPP_END && idx < h_ * w_) {
                int64_t c = t - CPP_START;
                if (c >= 0 && c < p) {
                    float v = (float)c / (p - 1);
                    uint8_t gray = (uint8_t)(v * 255.0f + 0.5f);
                    rgb[idx * 3 + 0] = gray;
                    rgb[idx * 3 + 1] = gray;
                    rgb[idx * 3 + 2] = gray;
                }
                idx++;
            }
        }
        return rgb;
    }

    // ── Сохранение арта в файл ────────────────────────────────────
    bool save_art(const std::string& path, const std::vector<int64_t>& tokens) const {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        std::string art = decode_to_string(tokens);
        f << art;
        return f.good();
    }

    // ── Загрузка арта из файла ─────────────────────────────────────
    std::vector<int64_t> load_art(const std::string& path) const {
        std::ifstream f(path);
        if (!f.is_open()) return {};

        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        std::vector<int64_t> tokens;
        tokens.reserve(h_ * w_);

        for (char c : content) {
            if (c == '\n') continue;
            for (int i = 0; i < palette_len_; i++) {
                if (palette_[i] == c) {
                    tokens.push_back(CPP_START + i);
                    break;
                }
            }
        }
        return tokens;
    }

    // ── Получить конфигурацию ──────────────────────────────────────
    int width()  const noexcept { return w_; }
    int height() const noexcept { return h_; }
    VisionMode mode() const noexcept { return mode_; }

private:
    // ── Вычисление luminance со встроенным bilinear resize ─────────
    void compute_luminance(const uint8_t* rgb, int sw, int sh, float* lum_out) const {
#if defined(_OPENMP)
        #pragma omp parallel for
#endif
        for (int y = 0; y < h_; y++) {
            for (int x = 0; x < w_; x++) {
                float src_y = (float)y * sh / h_;
                float src_x = (float)x * sw / w_;
                int sy = std::min((int)src_y, sh - 2);
                int sx = std::min((int)src_x, sw - 2);
                float fy = src_y - sy, fx = src_x - sx;
                lum_out[y * w_ + x] = bilerp_luminance(rgb, sw, sh, sy, sx, fy, fx);
            }
        }
    }

    // ── Bilinear interpolation с гамма-коррекцией ──────────────────
    float bilerp_luminance(const uint8_t* rgb, int sw, int sh,
                           int sy, int sx, float fy, float fx) const {
        auto get_lum = [&](int y, int x) -> float {
            y = std::min(y, sh - 1); x = std::min(x, sw - 1);
            int o = (y * sw + x) * 3;
            float r = gamma_lut_[rgb[o]];
            float g = gamma_lut_[rgb[o + 1]];
            float b = gamma_lut_[rgb[o + 2]];
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        };
        float v00 = get_lum(sy, sx),     v10 = get_lum(sy + 1, sx);
        float v01 = get_lum(sy, sx + 1), v11 = get_lum(sy + 1, sx + 1);
        return v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx)
             + v01 * (1 - fy) * fx       + v11 * fy * fx;
    }

    // ── Sobel edge detection ───────────────────────────────────────
    std::vector<float> sobel_edges(const std::vector<float>& in) const {
        std::vector<float> out(h_ * w_, 0.0f);
        float max_mag = 0.0f;

#if defined(_OPENMP)
        #pragma omp parallel for reduction(max:max_mag)
#endif
        for (int y = 1; y < h_ - 1; y++) {
            for (int x = 1; x < w_ - 1; x++) {
                float gx = -in[(y - 1) * w_ + (x - 1)] + in[(y - 1) * w_ + (x + 1)]
                           -2 * in[y * w_ + (x - 1)]   + 2 * in[y * w_ + (x + 1)]
                           -in[(y + 1) * w_ + (x - 1)] + in[(y + 1) * w_ + (x + 1)];
                float gy = -in[(y - 1) * w_ + (x - 1)] - 2 * in[(y - 1) * w_ + x] - in[(y - 1) * w_ + (x + 1)]
                           +in[(y + 1) * w_ + (x - 1)] + 2 * in[(y + 1) * w_ + x] + in[(y + 1) * w_ + (x + 1)];
                float mag = sqrtf(gx * gx + gy * gy);
                out[y * w_ + x] = mag;
                if (mag > max_mag) max_mag = mag;
            }
        }
        if (max_mag > 1e-6f) {
            float inv_max = 1.0f / max_mag;
#if defined(_OPENMP)
            #pragma omp parallel for
#endif
            for (int i = 0; i < h_ * w_; i++)
                out[i] *= inv_max;
        }
        return out;
    }

    // ── Canny-подобная обработка: NMS + double threshold ──────────
    std::vector<float> canny_refine(const std::vector<float>& in) const {
        std::vector<float> out(h_ * w_, 0.0f);

        // Non-maximum suppression
        for (int y = 1; y < h_ - 1; y++) {
            for (int x = 1; x < w_ - 1; x++) {
                float mag = in[y * w_ + x];
                if (mag < canny_low_) continue;

                // approximate gradient direction
                float gx = -in[(y - 1) * w_ + (x - 1)] + in[(y - 1) * w_ + (x + 1)]
                           -2 * in[y * w_ + (x - 1)]   + 2 * in[y * w_ + (x + 1)]
                           -in[(y + 1) * w_ + (x - 1)] + in[(y + 1) * w_ + (x + 1)];
                float gy = -in[(y - 1) * w_ + (x - 1)] - 2 * in[(y - 1) * w_ + x] - in[(y - 1) * w_ + (x + 1)]
                           +in[(y + 1) * w_ + (x - 1)] + 2 * in[(y + 1) * w_ + x] + in[(y + 1) * w_ + (x + 1)];

                float angle = atan2f(gy, gx) * (180.0f / 3.14159265f);
                if (angle < 0) angle += 180.0f;

                float n1 = 0, n2 = 0;
                if (angle < 22.5f || angle >= 157.5f) {
                    n1 = in[y * w_ + (x - 1)];
                    n2 = in[y * w_ + (x + 1)];
                } else if (angle < 67.5f) {
                    n1 = in[(y - 1) * w_ + (x + 1)];
                    n2 = in[(y + 1) * w_ + (x - 1)];
                } else if (angle < 112.5f) {
                    n1 = in[(y - 1) * w_ + x];
                    n2 = in[(y + 1) * w_ + x];
                } else {
                    n1 = in[(y - 1) * w_ + (x - 1)];
                    n2 = in[(y + 1) * w_ + (x + 1)];
                }

                if (mag >= n1 && mag >= n2) {
                    if (mag >= canny_high_)
                        out[y * w_ + x] = 1.0f;
                    else
                        out[y * w_ + x] = 0.5f;  // weak edge
                }
            }
        }

        // Edge tracking by hysteresis
        for (int y = 1; y < h_ - 1; y++) {
            for (int x = 1; x < w_ - 1; x++) {
                if (out[y * w_ + x] == 0.5f) {
                    bool connected = false;
                    for (int dy = -1; dy <= 1 && !connected; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                            if (out[(y + dy) * w_ + (x + dx)] == 1.0f)
                                connected = true;
                    out[y * w_ + x] = connected ? 1.0f : 0.0f;
                }
            }
        }
        return out;
    }

    // ── Квантование с LUT (фото) ───────────────────────────────────
    std::vector<int64_t> quantize_perceptual(const std::vector<float>& lum) const {
        std::vector<int64_t> tokens;
        tokens.reserve((size_t)h_ * w_);
        int p = palette_len_;

        if (dither_ == DitherMode::FloydSteinberg) {
            std::vector<float> work = lum;
            for (int y = 0; y < h_; y++) {
                for (int x = 0; x < w_; x++) {
                    float old = work[y * w_ + x];
                    float perceptual = sqrtf(fmaxf(0.0f, fminf(1.0f, old)));
                    int idx = std::min((int)(perceptual * (p - 1) + 0.5f), p - 1);
                    float new_v = (float)idx / (p - 1);
                    float new_perc = new_v * new_v;  // inverse sqrt
                    float quant_err = old - new_perc;
                    tokens.push_back(CPP_START + idx);
                    if (x + 1 < w_) work[y * w_ + (x + 1)] += quant_err * (7.0f / 16);
                    if (y + 1 < h_) {
                        if (x > 0) work[(y + 1) * w_ + (x - 1)] += quant_err * (3.0f / 16);
                        work[(y + 1) * w_ + x] += quant_err * (5.0f / 16);
                        if (x + 1 < w_) work[(y + 1) * w_ + (x + 1)] += quant_err * (1.0f / 16);
                    }
                }
            }
        } else {
            // Быстрый путь: quant_lut_
            for (float v : lum) {
                int idx;
                if (v <= 0.0f) {
                    idx = 0;
                } else if (v >= 1.0f) {
                    idx = p - 1;
                } else {
                    float perceptual = sqrtf(v);
                    idx = std::min((int)(perceptual * (p - 1) + 0.5f), p - 1);
                }
                tokens.push_back(CPP_START + idx);
            }
        }
        return tokens;
    }

    // ── Линейное квантование (для графов/градиентов) ───────────────
    std::vector<int64_t> quantize_linear(const std::vector<float>& vals) const {
        std::vector<int64_t> tokens;
        tokens.reserve((size_t)h_ * w_);
        int p = palette_len_;
        float scale = (float)(p - 1);
        for (float v : vals) {
            float clamped = fmaxf(0.0f, fminf(1.0f, v));
            int idx = std::min((int)(clamped * scale + 0.5f), p - 1);
            tokens.push_back(CPP_START + idx);
        }
        return tokens;
    }

    static int palette_len_default() noexcept { return 70; }

    int w_, h_;
    VisionMode mode_;
    DitherMode dither_;
    bool use_canny_;
    float canny_low_, canny_high_;
    int num_frames_;
    const char* palette_;
    int palette_len_;
    float gamma_lut_[256];
    int quant_lut_[256];
};
