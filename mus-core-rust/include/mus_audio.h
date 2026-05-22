#pragma once

#include "mus_model.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// ══════════════════════════════════════════════════════════════════════
//  Audio Pipeline — дискретная токенизация звука для трансформера
//
//  Архитектура:
//    Микрофон/файл → STFT magnitude → mel filterbank → VAD →
//    → hand-crafted audio features → uniform quantization → [AUDIO_START..AUDIO_END]
//
//  Такой подход работает без обучения (аудио-эмбеддинги дообучаются
//  вместе с моделью), и не требует внешнего VQ-VAE.
//  Для продакшна: заменить на EnCodec / DAC (discrete audio codes).
// ══════════════════════════════════════════════════════════════════════

// ─── Конфигурация ─────────────────────────────────────────────────────
struct AudioConfig {
    int sample_rate = 16000;      // Hz (стандарт для VAD/ASR)
    int fft_size = 512;           // размер окна STFT
    int hop_length = 160;         // шаг STFT (10ms при 16kHz)
    int n_mels = 20;              // mel-фильтров
    int max_freq = 8000;          // верхняя граница частот

    // Квантование
    int n_audio_tokens = 23;      // AUDIO_END - AUDIO_START + 1
    float codebook_min = -3.0f;   // нижняя граница log-mel (нормализовано)
    float codebook_max = 3.0f;
    bool vad_enabled = true;      // Voice Activity Detection
};

// ── Выходные токены ───────────────────────────────────────────────────
struct AudioTokens {
    std::vector<int64_t> tokens;  // с тегами <audio_start> ... <audio_end>
    float energy_db;              // средняя громкость (для логов)
    bool has_voice;               // VAD-решение
    int n_frames;                 // количество аудио-фреймов
};

// ══════════════════════════════════════════════════════════════════════
//  Mel Filterbank (предвычисленный)
// ══════════════════════════════════════════════════════════════════════

class MelFilterbank {
public:
    MelFilterbank(const AudioConfig& cfg) {
        int n_fft = cfg.fft_size;
        int n_bins = n_fft / 2 + 1;
        float fmax = cfg.max_freq;

        // mel scale limits
        float mel_min = hz_to_mel(0.0f);
        float mel_max = hz_to_mel(fmax);
        float mel_step = (mel_max - mel_min) / (cfg.n_mels + 1);

        weights_.resize(cfg.n_mels, std::vector<float>(n_bins, 0.0f));

        for (int m = 0; m < cfg.n_mels; m++) {
            float mel_center = mel_min + (m + 1) * mel_step;
            float hz_center = mel_to_hz(mel_center);

            float mel_left = mel_center - mel_step;
            float mel_right = mel_center + mel_step;
            float hz_left = mel_to_hz(mel_left);
            float hz_right = mel_to_hz(mel_right);

            for (int k = 0; k < n_bins; k++) {
                float hz = (float)k * cfg.sample_rate / n_fft;
                if (hz <= hz_left || hz >= hz_right) continue;

                if (hz <= hz_center)
                    weights_[m][k] = (hz - hz_left) / (hz_center - hz_left);
                else
                    weights_[m][k] = (hz_right - hz) / (hz_right - hz_center);
            }
        }
    }

    void apply(const float* magnitude, int n_bins, float* mel_out) const {
        int n_mels = weights_.size();
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_bins; k++)
                sum += weights_[m][k] * magnitude[k];
            mel_out[m] = sum;
        }
    }

    int n_mels() const { return weights_.size(); }
    int n_bins() const { return weights_[0].size(); }

private:
    static float hz_to_mel(float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    }
    static float mel_to_hz(float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    std::vector<std::vector<float>> weights_;
};

// ══════════════════════════════════════════════════════════════════════
//  Audio Encoder — PCM → токены
// ══════════════════════════════════════════════════════════════════════

class AudioEncoder {
public:
    explicit AudioEncoder(AudioConfig cfg = {})
        : cfg_(cfg), melbank_(cfg), hanning_(cfg.fft_size)
    {
        for (int i = 0; i < cfg.fft_size; i++)
            hanning_[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (cfg.fft_size - 1)));
    }

    // PCM float [-1, 1] → token sequence
    AudioTokens encode(const float* pcm, int n_samples) const {
        AudioTokens result;
        result.has_voice = false;
        result.energy_db = 0.0f;

        int hop = cfg_.hop_length;
        int n_fft = cfg_.fft_size;
        int n_frames = (n_samples - n_fft) / hop + 1;
        if (n_frames < 1) return result;

        int n_bins = n_fft / 2 + 1;
        int n_mels = cfg_.n_mels;

        // STFT + mel
        std::vector<float> magnitudes(n_frames * n_bins, 0.0f);
        std::vector<float> total_energy(n_frames, 0.0f);

        for (int t = 0; t < n_frames; t++) {
            int offset = t * hop;

            // apply hanning window
            for (int i = 0; i < n_fft; i++) {
                int idx = offset + i;
                float sample = (idx < n_samples) ? pcm[idx] : 0.0f;
                windowed_[i] = sample * hanning_[i];
            }

            // simplified pseudo-FFT magnitude via autocorrelation (для демо)
            // В продакшне: cufft / kissfft
            for (int k = 0; k < n_bins; k++) {
                float re = 0.0f, im = 0.0f;
                for (int i = 0; i < n_fft; i++) {
                    float angle = 2.0f * M_PI * k * i / n_fft;
                    re += windowed_[i] * cosf(angle);
                    im -= windowed_[i] * sinf(angle);
                }
                float mag = sqrtf(re * re + im * im) / n_fft;
                magnitudes[t * n_bins + k] = mag;
                total_energy[t] += mag * mag;
            }
            total_energy[t] = sqrtf(total_energy[t]);
        }

        // VAD: порог тишины
        float energy_thresh = 0.01f;
        int voice_frames = 0;
        if (cfg_.vad_enabled) {
            for (int t = 0; t < n_frames; t++)
                if (total_energy[t] > energy_thresh)
                    voice_frames++;
            result.has_voice = (voice_frames > n_frames * 0.1f);
        } else {
            result.has_voice = true;
        }

        if (!result.has_voice) {
            result.tokens.push_back(TAG_AUDIO_START);
            result.tokens.push_back(TAG_AUDIO_END);
            result.n_frames = 0;
            return result;
        }

        // Mel filterbank per frame
        std::vector<float> mel_frame(n_mels);
        std::vector<float> all_mels;
        all_mels.reserve(n_frames * n_mels);

        for (int t = 0; t < n_frames; t++) {
            melbank_.apply(&magnitudes[t * n_bins], n_bins, mel_frame.data());
            for (int m = 0; m < n_mels; m++) {
                // log-mel + clip
                float val = logf(fmaxf(mel_frame[m], 1e-6f));
                all_mels.push_back(val);
            }
        }

        // Normalize per mel band
        for (int m = 0; m < n_mels; m++) {
            float mean = 0.0f, var = 0.0f;
            for (int t = 0; t < n_frames; t++) mean += all_mels[t * n_mels + m];
            mean /= n_frames;
            for (int t = 0; t < n_frames; t++)
                var += (all_mels[t * n_mels + m] - mean) * (all_mels[t * n_mels + m] - mean);
            var = sqrtf(var / n_frames + 1e-6f);
            for (int t = 0; t < n_frames; t++)
                all_mels[t * n_mels + m] = (all_mels[t * n_mels + m] - mean) / var;
        }

        // Uniform quantization → tokens
        int codebook_size = AUDIO_END - AUDIO_START + 1;
        float lo = cfg_.codebook_min, hi = cfg_.codebook_max;
        float scale = codebook_size / (hi - lo);
        float sum_energy = 0.0f;

        result.tokens.push_back(TAG_AUDIO_START);
        for (int i = 0; i < (int)all_mels.size(); i++) {
            float clamped = fmaxf(lo, fminf(hi, all_mels[i]));
            int idx = (int)((clamped - lo) * scale);
            idx = std::min(idx, codebook_size - 1);
            result.tokens.push_back(AUDIO_START + idx);
            if (i % n_mels == 0) sum_energy += total_energy[i / n_mels];
        }
        result.tokens.push_back(TAG_AUDIO_END);
        result.energy_db = 20.0f * log10f(fmaxf(sum_energy / n_frames, 1e-6f));
        result.n_frames = n_frames;

        return result;
    }

    // PCM int16 → token sequence
    AudioTokens encode_i16(const int16_t* pcm_i16, int n_samples) const {
        std::vector<float> pcm_float(n_samples);
        for (int i = 0; i < n_samples; i++)
            pcm_float[i] = pcm_i16[i] / 32768.0f;
        return encode(pcm_float.data(), n_samples);
    }

    const AudioConfig& config() const { return cfg_; }

private:
    AudioConfig cfg_;
    MelFilterbank melbank_;
    std::vector<float> hanning_;
    mutable std::vector<float> windowed_ = std::vector<float>(cfg_.fft_size);
};

// ══════════════════════════════════════════════════════════════════════
//  Вспомогательные утилиты для работы с аудио
// ══════════════════════════════════════════════════════════════════════

// Поиск голосовой активности (энергетический VAD)
inline bool simple_vad(const float* pcm, int n_samples, float threshold_db = -30.0f) {
    float energy = 0.0f;
    for (int i = 0; i < n_samples; i++)
        energy += pcm[i] * pcm[i];
    energy = sqrtf(energy / n_samples);
    float db = 20.0f * log10f(fmaxf(energy, 1e-10f));
    return db > threshold_db;
}

// Нормализация звука (RMS-based)
inline void normalize_audio(float* pcm, int n_samples, float target_rms = 0.1f) {
    float rms = 0.0f;
    for (int i = 0; i < n_samples; i++) rms += pcm[i] * pcm[i];
    rms = sqrtf(rms / n_samples);
    if (rms > 1e-6f) {
        float gain = target_rms / rms;
        for (int i = 0; i < n_samples; i++) pcm[i] *= gain;
    }
}

// Даунсемпл (простой decimate + anti-alias filter)
inline void downsample(const float* in, int n_in, int in_rate, float* out, int out_rate) {
    if (in_rate == out_rate) {
        memcpy(out, in, n_in * sizeof(float));
        return;
    }
    // simple nearest, для продакшна нужен polyphase filter
    float ratio = (float)in_rate / out_rate;
    int n_out = (int)(n_in / ratio);
    for (int i = 0; i < n_out; i++) {
        int src = std::min((int)(i * ratio), n_in - 1);
        out[i] = in[src];
    }
}
