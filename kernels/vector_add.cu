// Kernel 1 of the co-req thread: vector add.
// Exit condition (per curriculum): result matches a CPU reference bitwise.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;  // 1M elements
    const size_t bytes = n * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
        h_ref[i] = h_a[i] + h_b[i];  // CPU reference
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Bitwise comparison against CPU reference: a+b in fp32 is exact same op
    // on both sides, so any mismatch means a real bug, not float noise.
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (memcmp(&h_c[i], &h_ref[i], sizeof(float)) != 0) mismatches++;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("n=%d  mismatches=%d  -> %s\n", n, mismatches,
           mismatches == 0 ? "PASS (bitwise)" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return mismatches == 0 ? 0 : 1;
}
