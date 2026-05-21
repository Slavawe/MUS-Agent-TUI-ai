#include "mus_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <float.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(1);                                               \
    }                                                          \
} while(0)

// CPU reference for a single position
float cpu_pos_loss(const float* logits, int64_t label, const float* weights, int V) {
    if (label == -100) return 0.0f;
    float maxv = -FLT_MAX;
    for (int i = 0; i < V; i++) maxv = fmaxf(maxv, logits[i]);
    float denom = 0.0f;
    for (int i = 0; i < V; i++) denom += expf(logits[i] - maxv);
    float sm = expf(logits[label] - maxv) / denom;
    sm = fmaxf(sm, 1e-30f);
    return -logf(sm) * weights[label];
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s\n", prop.name);

    MUSConfig cfg;
    int B = 1, S = 4, V = cfg.vocab_size;
    int num_pos = B * S;

    // Tiny test: 4 positions, each with V=48000 logits
    std::vector<float> h_logits(num_pos * V);
    std::vector<int64_t> h_labels(num_pos);
    std::vector<float> h_weights(V);
    build_weight_table(cfg, h_weights.data(), V);

    std::mt19937 rng(42);
    std::normal_distribution<float> ld(0.0f, 0.5f);
    std::uniform_int_distribution<int> lad(0, V - 1);

    for (int i = 0; i < num_pos * V; i++) h_logits[i] = ld(rng);
    h_labels[0] = 100;      // AER
    h_labels[1] = 2050;     // ASCII
    h_labels[2] = 30000;    // TEXT
    h_labels[3] = -100;     // PADDING

    // CPU reference
    printf("\nCPU per-position:\n");
    float cpu_sum = 0.0f;
    int valid = 0;
    for (int p = 0; p < num_pos; p++) {
        float l = cpu_pos_loss(h_logits.data() + p * V, h_labels[p], h_weights.data(), V);
        printf("  pos %d (label=%4lld): loss = %.10f\n", p, (long long)h_labels[p], l);
        if (h_labels[p] != -100) { cpu_sum += l; valid++; }
    }
    printf("  CPU mean: %.10f (sum=%.10f, valid=%d)\n", cpu_sum/valid, cpu_sum, valid);

    // Copy to GPU
    float *d_logits, *d_weights, *d_loss;
    int64_t *d_labels;
    CUDA_CHECK(cudaMalloc(&d_logits, num_pos * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_pos * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), num_pos * V * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), num_pos * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), V * sizeof(float), cudaMemcpyHostToDevice));

    // GPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float gpu_loss = mus_weighted_ce_forward(d_logits, d_labels, d_weights, d_loss, B, S, V, stream);
    printf("\nGPU sum: %.10f\n", gpu_loss);
    printf("GPU mean (sum/valid): %.10f\n", gpu_loss / valid);
    printf("Diff: %.2e  %s\n", fabsf(gpu_loss/valid - cpu_sum/valid),
           fabsf(gpu_loss/valid - cpu_sum/valid) < 1e-4 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_loss));
    return 0;
}