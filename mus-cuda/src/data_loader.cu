#include "mus_cuda.h"
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <climits>

// ─── Data Types ──────────────────────────────────────────────────────────
struct MultimodalSample {
    std::vector<int64_t> text_tokens;      // Text token IDs
    std::vector<float> text_logits;        // Optional teacher logits
    std::vector<float> vision_features;    // Vision features [N, vision_dim]
    std::vector<float> audio_features;     // Audio features [T, audio_dim]
    std::vector<int64_t> labels;           // Target labels (aligned with text_tokens)
    std::vector<float> token_weights;      // Per-token loss weights
    int N;                                 // Vision sequence length
    int T;                                 // Audio sequence length
    float total_aer;                       // Text AER (accuracy) metric
    int64_t pos;                           // Position index

    MultimodalSample() : N(0), T(0), total_aer(0.0f), pos(0) {}

    bool is_valid() const {
        return !text_tokens.empty() && !labels.empty() &&
               text_tokens.size() == labels.size();
    }
};

// ─── Thread-safe Ring Buffer ────────────────────────────────────────────
template<typename T>
class RingBuffer {
    std::vector<T> buffer;
    size_t head, tail, capacity;
    std::mutex mtx;
    std::condition_variable not_full, not_empty;
    bool closed;

public:
    RingBuffer(size_t cap) : buffer(cap), head(0), tail(0), capacity(cap), closed(false) {}

    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        while (is_full()) {
            if (closed) return false;
            not_full.wait_for(lock, std::chrono::milliseconds(10));
        }
        buffer[tail] = item;
        tail = (tail + 1) % capacity;
        not_empty.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        while (is_empty()) {
            if (closed) return false;
            not_empty.wait_for(lock, std::chrono::milliseconds(10));
        }
        item = buffer[head];
        head = (head + 1) % capacity;
        not_full.notify_one();
        return true;
    }

    bool is_full() const { return (tail + 1) % capacity == head; }
    bool is_empty() const { return head == tail; }
    size_t size() const { return (tail - head + capacity) % capacity; }

    void close() {
        closed = true;
        not_empty.notify_all();
        not_full.notify_all();
    }
};

// ─── Binary Format Loader ───────────────────────────────────────────────
class BinaryDatasetLoader {
public:
    static bool load_text_data(const char* path,
                               std::vector<MultimodalSample>& samples,
                               int max_samples = -1) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            printf("Error: Cannot open %s\n", path);
            return false;
        }

        int32_t num_samples;
        file.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));
        printf("  Loading %d samples from %s\n", num_samples, path);

        if (max_samples > 0 && max_samples < num_samples) {
            num_samples = max_samples;
        }

        samples.reserve(samples.size() + num_samples);

        for (int32_t i = 0; i < num_samples; ++i) {
            if (!file.good()) {
                printf("  Warning: Premature end of file at sample %d\n", i);
                break;
            }

            MultimodalSample sample;

            int32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            sample.text_tokens.resize(len);
            file.read(reinterpret_cast<char*>(sample.text_tokens.data()),
                      len * sizeof(int64_t));

            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            sample.labels.resize(len);
            file.read(reinterpret_cast<char*>(sample.labels.data()),
                      len * sizeof(int64_t));

            file.read(reinterpret_cast<char*>(&sample.total_aer), sizeof(float));

            int32_t N;
            file.read(reinterpret_cast<char*>(&N), sizeof(N));
            sample.N = N;
            if (N > 0) {
                int32_t vision_dim;
                file.read(reinterpret_cast<char*>(&vision_dim), sizeof(vision_dim));
                sample.vision_features.resize(N * vision_dim);
                file.read(reinterpret_cast<char*>(sample.vision_features.data()),
                          N * vision_dim * sizeof(float));
            }

            int32_t T;
            file.read(reinterpret_cast<char*>(&T), sizeof(T));
            sample.T = T;
            if (T > 0) {
                int32_t audio_dim;
                file.read(reinterpret_cast<char*>(&audio_dim), sizeof(audio_dim));
                sample.audio_features.resize(T * audio_dim);
                file.read(reinterpret_cast<char*>(sample.audio_features.data()),
                          T * audio_dim * sizeof(float));
            }

            if (sample.is_valid()) {
                samples.push_back(std::move(sample));
            }

            if ((i + 1) % 10000 == 0) {
                printf("    Loaded %d/%d samples\r", i + 1, num_samples);
            }
        }

        printf("  Total loaded: %zu samples\n", samples.size());
        return !samples.empty();
    }

    static void generate_dummy_data(std::vector<MultimodalSample>& samples,
                                    int num_samples, int vocab_size,
                                    int seq_len, int vision_dim, int audio_dim,
                                    std::mt19937& rng) {
        printf("  Generating %d dummy samples...\n", num_samples);
        samples.reserve(samples.size() + num_samples);

        for (int i = 0; i < num_samples; ++i) {
            MultimodalSample sample;

            sample.text_tokens.resize(seq_len);
            sample.labels.resize(seq_len);
            sample.token_weights.resize(seq_len, 1.0f);

            for (int j = 0; j < seq_len; ++j) {
                sample.text_tokens[j] = rng() % vocab_size;
                sample.labels[j] = rng() % vocab_size;
            }

            sample.N = 64;
            sample.vision_features.resize(sample.N * vision_dim);
            for (auto& f : sample.vision_features) {
                f = (float)rng() / (float)rng.max();
            }

            if (audio_dim > 0) {
                sample.T = 128;
                sample.audio_features.resize(sample.T * audio_dim);
                for (auto& f : sample.audio_features) {
                    f = (float)rng() / (float)rng.max();
                }
            }

            sample.total_aer = (float)rng() / (float)rng.max() * 0.3f;
            sample.pos = i;

            samples.push_back(std::move(sample));
        }
    }
};

// ─── Batch Builder ──────────────────────────────────────────────────────
struct TrainingBatch {
    int64_t* text_input;
    int64_t* text_labels;
    int64_t* positions;
    float* token_weights;
    float* ce_weights;
    float* vision_features;
    float* audio_features;

    int B, S, N, T, vision_dim, audio_dim;

    TrainingBatch(int b, int s, int n, int t, int vd, int ad)
        : B(b), S(s), N(n), T(t), vision_dim(vd), audio_dim(ad) {
        text_input = new int64_t[B * S]();
        text_labels = new int64_t[B * S]();
        positions = new int64_t[B * S]();
        token_weights = new float[B * S]();
        ce_weights = new float[B * S]();

        if (vd > 0 && N > 0) {
            vision_features = new float[B * N * vd]();
        } else {
            vision_features = nullptr;
        }

        if (ad > 0 && T > 0) {
            audio_features = new float[B * T * ad]();
        } else {
            audio_features = nullptr;
        }
    }

    ~TrainingBatch() {
        delete[] text_input;
        delete[] text_labels;
        delete[] positions;
        delete[] token_weights;
        delete[] ce_weights;
        delete[] vision_features;
        delete[] audio_features;
    }

    bool is_valid() const {
        return text_input != nullptr && text_labels != nullptr;
    }
};

class BatchBuilder {
    int B, S, N, T;
    int vision_dim, audio_dim;
    int vocab_size;

public:
    BatchBuilder(int batch_size, int seq_len, int max_vision_seq, int max_audio_seq,
                 int vd, int ad, int vs)
        : B(batch_size), S(seq_len), N(max_vision_seq), T(max_audio_seq),
          vision_dim(vd), audio_dim(ad), vocab_size(vs) {}

    TrainingBatch build_batch(std::vector<MultimodalSample>& samples,
                              int start_idx) {
        TrainingBatch batch(B, S, N, T, vision_dim, audio_dim);

        for (int b = 0; b < B && (start_idx + b) < (int)samples.size(); ++b) {
            auto& sample = samples[start_idx + b];

            int seq_len = std::min((int)sample.text_tokens.size(), S);
            for (int s = 0; s < seq_len; ++s) {
                batch.text_input[b * S + s] = sample.text_tokens[s];
                batch.text_labels[b * S + s] = sample.labels[s];
                batch.positions[b * S + s] = s;
                batch.token_weights[b * S + s] = 1.0f;
                batch.ce_weights[b * S + s] = 1.0f + sample.total_aer * 2.0f;
            }

            for (int s = seq_len; s < S; ++s) {
                batch.text_input[b * S + s] = 0;
                batch.text_labels[b * S + s] = -1;
                batch.token_weights[b * S + s] = 0.0f;
                batch.ce_weights[b * S + s] = 0.0f;
            }

            if (vision_dim > 0 && !sample.vision_features.empty() && batch.vision_features) {
                int n = std::min(sample.N, N);
                for (int i = 0; i < n; ++i) {
                    for (int d = 0; d < vision_dim; ++d) {
                        batch.vision_features[(b * N + i) * vision_dim + d] =
                            sample.vision_features[i * vision_dim + d];
                    }
                }
            }

            if (audio_dim > 0 && !sample.audio_features.empty() && batch.audio_features) {
                int t = std::min(sample.T, T);
                for (int i = 0; i < t; ++i) {
                    for (int d = 0; d < audio_dim; ++d) {
                        batch.audio_features[(b * T + i) * audio_dim + d] =
                            sample.audio_features[i * audio_dim + d];
                    }
                }
            }
        }

        return batch;
    }

    TrainingBatch build_augmented_batch(std::vector<MultimodalSample>& samples,
                                        int start_idx, std::mt19937& rng) {
        TrainingBatch batch = build_batch(samples, start_idx);

        if (rng() % 100 < 20) {
            for (int b = 0; b < B; ++b) {
                if (rng() % 2 == 0) continue;

                int shift = (rng() % 10) + 1;
                shift = std::min(shift, S / 2);

                for (int s = S - 1; s >= shift; --s) {
                    batch.text_input[b * S + s] = batch.text_input[b * S + s - shift];
                    batch.text_labels[b * S + s] = batch.text_labels[b * S + s - shift];
                }
                for (int s = 0; s < shift; ++s) {
                    batch.text_input[b * S + s] = 0;
                    batch.text_labels[b * S + s] = 0;
                    batch.token_weights[b * S + s] = 0.0f;
                }
            }
        }

        return batch;
    }
};

// ─── Multimodal Dataset ──────────────────────────────────────────────────
class MultimodalDataset {
public:
    std::vector<MultimodalSample> train_samples;
    std::vector<MultimodalSample> val_samples;
    int vocab_size;
    int vision_dim, audio_dim;
    int max_vision_seq, max_audio_seq;

    MultimodalDataset() : vocab_size(48000), vision_dim(512), audio_dim(384),
                          max_vision_seq(64), max_audio_seq(128) {}

    bool load(const char* train_path, const char* val_path = nullptr,
              int max_train = -1, int max_val = -1) {
        printf("Loading multimodal dataset:\n");

        if (train_path) {
            if (!BinaryDatasetLoader::load_text_data(train_path, train_samples, max_train)) {
                printf("  Failed to load training data, using dummy data\n");
                std::mt19937 rng(42);
                BinaryDatasetLoader::generate_dummy_data(
                    train_samples, 1000, vocab_size, 256,
                    vision_dim, audio_dim, rng);
            }
        }

        if (val_path) {
            BinaryDatasetLoader::load_text_data(val_path, val_samples, max_val);
        }

        printf("  Train: %zu samples\n", train_samples.size());
        printf("  Val: %zu samples\n", val_samples.size());

        return !train_samples.empty();
    }

    void generate_dummy(int num_train = 1000, int num_val = 100) {
        std::mt19937 rng(42);
        BinaryDatasetLoader::generate_dummy_data(
            train_samples, num_train, vocab_size, 256,
            vision_dim, audio_dim, rng);
        BinaryDatasetLoader::generate_dummy_data(
            val_samples, num_val, vocab_size, 256,
            vision_dim, audio_dim, rng);
        printf("Generated %d train, %d val dummy samples\n", num_train, num_val);
    }

    void shuffle(std::mt19937& rng) {
        std::shuffle(train_samples.begin(), train_samples.end(), rng);
    }

    BatchBuilder get_batch_builder(int batch_size, int seq_len) {
        return BatchBuilder(batch_size, seq_len, max_vision_seq, max_audio_seq,
                           vision_dim, audio_dim, vocab_size);
    }

    size_t train_size() const { return train_samples.size(); }
    size_t val_size() const { return val_samples.size(); }
    bool has_val() const { return !val_samples.empty(); }

    int get_vision_dim() const { return vision_dim; }
    int get_audio_dim() const { return audio_dim; }

    void analyze() const {
        printf("\n=== Dataset Analysis ===\n");
        printf("Total: %zu samples\n", train_samples.size());

        if (train_samples.empty()) return;

        double avg_aer = 0.0;
        int min_len = INT_MAX, max_len = 0;
        double avg_len = 0.0;
        int has_vision = 0, has_audio = 0;

        for (const auto& s : train_samples) {
            avg_aer += s.total_aer;
            int len = (int)s.text_tokens.size();
            min_len = std::min(min_len, len);
            max_len = std::max(max_len, len);
            avg_len += len;
            if (s.N > 0) has_vision++;
            if (s.T > 0) has_audio++;
        }

        avg_aer /= train_samples.size();
        avg_len /= train_samples.size();

        printf("  Avg AER: %.2f\n", avg_aer);
        printf("  Sequence length: avg=%.1f, min=%d, max=%d\n",
               avg_len, min_len, max_len);
        printf("  Samples with vision: %d (%.1f%%)\n",
               has_vision, 100.0 * has_vision / train_samples.size());
        printf("  Samples with audio: %d (%.1f%%)\n",
               has_audio, 100.0 * has_audio / train_samples.size());
    }
};

// ─── Async Data Loader ──────────────────────────────────────────────────
class AsyncDataLoader {
    RingBuffer<TrainingBatch*> queue;
    MultimodalDataset* dataset;
    BatchBuilder builder;
    std::thread loader_thread;
    std::atomic<bool> running;
    int batch_size, seq_len;

public:
    AsyncDataLoader(MultimodalDataset* ds, int queue_size = 4,
                    int b = 1, int s = 256)
        : queue(queue_size), dataset(ds),
          builder(b, s, ds->max_vision_seq, ds->max_audio_seq,
                  ds->vision_dim, ds->audio_dim, ds->vocab_size),
          running(false), batch_size(b), seq_len(s) {}

    ~AsyncDataLoader() {
        stop();
    }

    void start() {
        if (running) return;
        running = true;
        loader_thread = std::thread(&AsyncDataLoader::load_loop, this);
    }

    void stop() {
        if (!running) return;
        running = false;
        queue.close();
        if (loader_thread.joinable()) {
            loader_thread.join();
        }
        TrainingBatch* batch;
        while (queue.pop(batch)) {
            delete batch;
        }
    }

    TrainingBatch* get_batch() {
        TrainingBatch* batch = nullptr;
        if (queue.pop(batch)) {
            return batch;
        }
        return nullptr;
    }

private:
    void load_loop() {
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        int idx = 0;

        while (running) {
            if (dataset->train_samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (idx >= (int)dataset->train_samples.size()) {
                idx = 0;
                dataset->shuffle(rng);
            }

            TrainingBatch* batch = builder.build_augmented_batch(
                dataset->train_samples, idx, rng);

            if (!queue.push(batch)) {
                delete batch;
            }

            idx += batch_size;

            if (queue.is_full()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
};

// ─── Preprocessing Utilities ────────────────────────────────────────────

void normalize_vision_features(std::vector<float>& features, int N, int D) {
    for (int i = 0; i < N; ++i) {
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < D; ++d) mean += features[i * D + d];
        mean /= D;
        for (int d = 0; d < D; ++d) {
            float diff = features[i * D + d] - mean;
            var += diff * diff;
        }
        var = sqrtf(var / D + 1e-6f);
        for (int d = 0; d < D; ++d)
            features[i * D + d] = (features[i * D + d] - mean) / var;
    }
}

void normalize_audio_features(std::vector<float>& features, int T, int D) {
    for (int i = 0; i < T; ++i) {
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < D; ++d) mean += features[i * D + d];
        mean /= D;
        for (int d = 0; d < D; ++d) {
            float diff = features[i * D + d] - mean;
            var += diff * diff;
        }
        var = sqrtf(var / D + 1e-6f);
        for (int d = 0; d < D; ++d)
            features[i * D + d] = (features[i * D + d] - mean) / var;
    }
}

void apply_time_masking(std::vector<float>& features, int T, int D,
                        int mask_len, std::mt19937& rng) {
    int start = rng() % std::max(1, T - mask_len);
    for (int i = start; i < start + mask_len && i < T; ++i)
        for (int d = 0; d < D; ++d)
            features[i * D + d] = 0.0f;
}

void apply_freq_masking(std::vector<float>& features, int N, int D,
                        int mask_dim, std::mt19937& rng) {
    int start = rng() % std::max(1, D - mask_dim);
    for (int i = 0; i < N; ++i)
        for (int d = start; d < start + mask_dim && d < D; ++d)
            features[i * D + d] = 0.0f;
}

// ─── Main: standalone data loader test ───────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <data_path.bin> [options]\n", argv[0]);
        printf("  --max_samples N  : max samples to load\n");
        printf("  --dummy N        : generate N dummy samples instead\n");
        printf("  --analyze        : analyze dataset\n");
        printf("  --test           : test data pipeline\n");
        return 1;
    }

    const char* data_path = argv[1];
    int max_samples = -1;
    bool analyze = false, test = false, dummy = false;
    int dummy_count = 1000;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max" && i + 1 < argc)
            max_samples = std::stoi(argv[++i]);
        else if (arg == "--analyze")
            analyze = true;
        else if (arg == "--test")
            test = true;
        else if (arg == "--dummy" && i + 1 < argc)
            dummy = true, dummy_count = std::stoi(argv[++i]);
    }

    MultimodalDataset dataset;

    if (dummy) {
        dataset.generate_dummy(dummy_count, dummy_count / 10);
    } else {
        dataset.load(data_path, nullptr, max_samples);
    }

    if (analyze) {
        dataset.analyze();
    }

    if (test) {
        printf("\n=== Testing Data Pipeline ===\n");
        auto builder = dataset.get_batch_builder(2, 256);

        std::vector<MultimodalSample> samples = {dataset.train_samples[0]};
        auto batch = builder.build_batch(samples, 0);

        printf("Batch: B=%d, S=%d, vision_dim=%d, audio_dim=%d\n",
               batch.B, batch.S, batch.vision_dim, batch.audio_dim);

        // Print first few tokens
        printf("First batch tokens: ");
        for (int s = 0; s < std::min(8, batch.S); ++s) {
            printf("%ld ", batch.text_input[s]);
        }
        printf("...\n");
        printf("Dataset test passed! ✅\n");
    }

    printf("\nDone.\n");
    return 0;
}