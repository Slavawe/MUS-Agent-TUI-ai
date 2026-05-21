#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CK(c) do { cudaError_t e = c; if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while(0)
#define CB(c) do { cublasStatus_t s = c; if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS err %s:%d: %d\n", __FILE__, __LINE__, s); return 1; } } while(0)

int main() {
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetStream(h, st));

    float *A, *B, *C;
    CK(cudaMalloc(&A, 2304*768*4));
    CK(cudaMalloc(&B, 768*2048*4));
    CK(cudaMalloc(&C, 2304*2048*4));
    CK(cudaMemset(A, 0, 2304*768*4));
    CK(cudaMemset(B, 0, 768*2048*4));
    CK(cudaMemset(C, 0, 2304*2048*4));

    float alpha=1, beta=0;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
        2304, 2048, 768, &alpha, A, 2304, B, 768, &beta, C, 2304));
    CK(cudaStreamSynchronize(st));
    printf("GEMM OK\n");

    // Also test batched
    float *P;
    CK(cudaMalloc(&P, 48*512*512*4));
    CK(cudaMemset(P, 0, 48*512*512*4));
    float scl = 1.0f/sqrtf(64.0f);
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
        512,512,64, &scl, A,64,512*64, B,64,512*64, &beta, P,512,512*512, 48));
    CK(cudaStreamSynchronize(st));
    printf("Batched GEMM OK\n");

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(P);
    cublasDestroy(h); cudaStreamDestroy(st);
    printf("All OK\n");
    return 0;
}
