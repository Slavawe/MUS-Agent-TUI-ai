#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("Global mem: %.1f GB\n", (float)prop.totalGlobalMem / (1024*1024*1024));

    // Simple kernel test
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    cudaMemset(d_data, 0, 1024 * sizeof(float));

    float *h_data = new float[1024];
    cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy OK: h_data[0] = %.1f\n", h_data[0]);

    delete[] h_data;
    cudaFree(d_data);
    printf("PASS\n");
    return 0;
}
