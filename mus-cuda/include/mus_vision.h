#pragma once

#include "mus_model.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// ══════════════════════════════════════════════════════════════════════
//  C++ ASCII-Vision Encoder — изображение → последовательность токенов
//
//  Конвейер:  sRGB → gamma correction → BT.709 luminance
//           → bilinear resize → sqrt quantization → ASCII-токены
//
//  Режимы:
//    photo — perceptual (gamma + sqrt) для фотографий
//    graph — Sobel edge detection для схем/графиков
// ══════════════════════════════════════════════════════════════════════

enum class VisionMode {
    Photo,
    Graph
};

struct VisionEncoderConfig {
    int width = 64;
    int height = 32;
    VisionMode mode = VisionMode::Photo;
};

class ASCIITokenizer {
public:
    ASCIITokenizer(VisionEncoderConfig cfg = {})
        : w_(cfg.width), h_(cfg.height), mode_(cfg.mode)
    {
        // Gamma LUT (sRGB → linear)
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            gamma_lut_[i] = (v <= 0.04045f) ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
        }

        // Precompute perceptual quantization LUT
        int p = palette_len();
        for (int i = 0; i < 256; i++) {
            float linear = gamma_lut_[i];
            float perceptual = sqrtf(linear);
            perceptual = fmaxf(0.0f, fminf(1.0f, perceptual));
            quant_lut_[i] = (int)(perceptual * (p - 1) + 0.5f);
            if (quant_lut_[i] >= p) quant_lut_[i] = p - 1;
        }
    }

    // ── Основной метод: RGB пиксели → токены ─────────────────────
    std::vector<int64_t> encode(const uint8_t* rgb_pixels, int src_w, int src_h) const {
        // 1. Преобразование в luminance с гамма-коррекцией
        std::vector<float> lum(h_ * w_);

        if (mode_ == VisionMode::Photo) {
            // Photo: gamma + BT.709
            for (int y = 0; y < h_; y++) {
                for (int x = 0; x < w_; x++) {
                    float src_y = (float)y * src_h / h_;
                    float src_x = (float)x * src_w / w_;
                    int sy = std::min((int)src_y, src_h - 2);
                    int sx = std::min((int)src_x, src_w - 2);
                    float fy = src_y - sy, fx = src_x - sx;
                    lum[y * w_ + x] = bilerp_luminance(rgb_pixels, src_w, src_h, sy, sx, fy, fx);
                }
            }
            // Perceptual quantization
            return quantize_perceptual(lum);
        } else {
            // Graph: raw luminance + Sobel edges
            std::vector<float> raw(h_ * w_);
            for (int y = 0; y < h_; y++) {
                for (int x = 0; x < w_; x++) {
                    float src_y = (float)y * src_h / h_;
                    float src_x = (float)x * src_w / w_;
                    int sy = std::min((int)src_y, src_h - 2);
                    int sx = std::min((int)src_x, src_w - 2);
                    float fy = src_y - sy, fx = src_x - sx;
                    raw[y * w_ + x] = bilerp_luminance_raw(rgb_pixels, src_w, src_h, sy, sx, fy, fx);
                }
            }
            auto edges = sobel_edges(raw);
            return quantize_linear(edges);
        }
    }

    // ── Токенизированное представление ────────────────────────────
    std::vector<int64_t> encode_with_tags(const uint8_t* rgb_pixels, int src_w, int src_h) const {
        std::vector<int64_t> tokens;
        tokens.push_back(TAG_VISION_START);
        tokens.push_back(mode_ == VisionMode::Photo ? TAG_MODE_PHOTO : TAG_MODE_GRAPH);

        auto content = encode(rgb_pixels, src_w, src_h);
        tokens.insert(tokens.end(), content.begin(), content.end());

        tokens.push_back(TAG_VISION_END);
        return tokens;
    }

    // ── Декодирование токенов → ASCII-изображение ─────────────────
    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string result;
        const char* palette = default_ascii_palette();
        int p = palette_len();

        for (int64_t t : tokens) {
            if (t >= ASCII_START && t <= ASCII_END) {
                int idx = (int)(t - ASCII_START);
                if (idx >= 0 && idx < p) {
                    result += palette[idx];
                }
            } else if (t == TAG_FRAME_SEP) {
                result += '\n';
            }
        }

        // Format as grid
        std::string grid;
        for (int y = 0; y < h_ && y * w_ < (int)result.size(); y++) {
            grid += result.substr(y * w_, w_) + '\n';
        }
        return grid;
    }

private:
    // ── Bilinear interpolation with gamma-corrected luminance ────
    float bilerp_luminance(const uint8_t* rgb, int sw, int sh, int sy, int sx, float fy, float fx) const {
        auto get_lum = [&](int y, int x) {
            y = std::min(y, sh - 1); x = std::min(x, sw - 1);
            int o = (y * sw + x) * 3;
            float r = gamma_lut_[rgb[o]], g = gamma_lut_[rgb[o+1]], b = gamma_lut_[rgb[o+2]];
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        };
        float v00 = get_lum(sy, sx), v10 = get_lum(sy+1, sx);
        float v01 = get_lum(sy, sx+1), v11 = get_lum(sy+1, sx+1);
        return v00 * (1-fy) * (1-fx) + v10 * fy * (1-fx)
             + v01 * (1-fy) * fx + v11 * fy * fx;
    }

    float bilerp_luminance_raw(const uint8_t* rgb, int sw, int sh, int sy, int sx, float fy, float fx) const {
        auto get_lum_raw = [&](int y, int x) {
            y = std::min(y, sh - 1); x = std::min(x, sw - 1);
            int o = (y * sw + x) * 3;
            return 0.2126f * rgb[o] / 255.0f + 0.7152f * rgb[o+1] / 255.0f + 0.0722f * rgb[o+2] / 255.0f;
        };
        float v00 = get_lum_raw(sy, sx), v10 = get_lum_raw(sy+1, sx);
        float v01 = get_lum_raw(sy, sx+1), v11 = get_lum_raw(sy+1, sx+1);
        return v00 * (1-fy) * (1-fx) + v10 * fy * (1-fx)
             + v01 * (1-fy) * fx + v11 * fy * fx;
    }

    // ── Sobel edge detection ─────────────────────────────────────
    std::vector<float> sobel_edges(const std::vector<float>& in) const {
        std::vector<float> out(h_ * w_, 0.0f);
        float max_mag = 0.0f;
        for (int y = 1; y < h_ - 1; y++) {
            for (int x = 1; x < w_ - 1; x++) {
                float gx = -in[(y-1)*w_ + (x-1)] + in[(y-1)*w_ + (x+1)]
                           -2*in[y*w_ + (x-1)] + 2*in[y*w_ + (x+1)]
                           -in[(y+1)*w_ + (x-1)] + in[(y+1)*w_ + (x+1)];
                float gy = -in[(y-1)*w_ + (x-1)] - 2*in[(y-1)*w_ + x] - in[(y-1)*w_ + (x+1)]
                           +in[(y+1)*w_ + (x-1)] + 2*in[(y+1)*w_ + x] + in[(y+1)*w_ + (x+1)];
                float mag = sqrtf(gx*gx + gy*gy);
                out[y*w_ + x] = mag;
                if (mag > max_mag) max_mag = mag;
            }
        }
        if (max_mag > 1e-6f)
            for (auto& v : out) v /= max_mag;
        return out;
    }

    // ── Quantization ─────────────────────────────────────────────
    std::vector<int64_t> quantize_perceptual(const std::vector<float>& lum) const {
        std::vector<int64_t> tokens;
        tokens.reserve(h_ * w_);
        int p = palette_len();
        for (float v : lum) {
            float perceptual = sqrtf(fmaxf(0.0f, fminf(1.0f, v)));
            int idx = std::min((int)(perceptual * (p - 1) + 0.5f), p - 1);
            tokens.push_back(ASCII_START + idx);
        }
        return tokens;
    }

    std::vector<int64_t> quantize_linear(const std::vector<float>& vals) const {
        std::vector<int64_t> tokens;
        tokens.reserve(h_ * w_);
        int p = palette_len();
        for (float v : vals) {
            int idx = std::min((int)(fmaxf(0.0f, fminf(1.0f, v)) * (p - 1) + 0.5f), p - 1);
            tokens.push_back(ASCII_START + idx);
        }
        return tokens;
    }

    int w_, h_;
    VisionMode mode_;
    float gamma_lut_[256];
    int quant_lut_[256];
};
