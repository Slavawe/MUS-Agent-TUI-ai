#include "mus_cuda.h"
#include <stdio.h>

int main() {
    MUSContext* ctx = mus_create_context(512*1024*1024);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 768*768*4);
    cudaMalloc(&d_b, 768*2048*4);
    cudaMalloc(&d_c, 2304*2048*4);
    cudaMemset(d_a, 0, 768*768*4);
    cudaMemset(d_b, 0, 768*2048*4);
    cudaMemset(d_c, 0, 2304*2048*4);
    float alpha=1, beta=0;
    cublasStatus_t st = cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        2304, 2048, 768, &alpha, d_a, 2304, d_b, 768, &beta, d_c, 2304);
    printf("cublas status: %d\n", st);
    cudaError_t ce = cudaDeviceSynchronize();
    printf("sync: %s\n", cudaGetErrorString(ce));
    mus_destroy_context(ctx);
    return 0;
}
