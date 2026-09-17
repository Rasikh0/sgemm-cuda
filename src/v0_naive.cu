#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Stops the program and prints where, if any CUDA call fails.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// CPU reference: the "known correct" answer we check the GPU against.
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// The naive GPU kernel: one thread computes one element of C.
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {               // guard: don't run off the edge of C
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512;                       // square N x N matrices
    const size_t bytes = (size_t)N * N * sizeof(float);

    // Host (CPU) memory
    float* h_A     = new float[N * N];
    float* h_B     = new float[N * N];
    float* h_C_gpu = new float[N * N];       // result from the GPU
    float* h_C_cpu = new float[N * N];       // result from the CPU reference

    // Fill A and B with reproducible random values in [0, 1)
    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Device (GPU) memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy inputs CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch: 16x16 threads per block, enough blocks to cover N x N
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // did the launch itself fail?
    CUDA_CHECK(cudaDeviceSynchronize());     // wait for the kernel, catch runtime errors

    // Copy result GPU -> CPU
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Compute the reference and compare
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        float rel  = diff / (fabsf(h_C_cpu[i]) + 1e-5f);
        if (rel > max_err) max_err = rel;
    }
    printf("Matrix size: %d x %d\n", N, N);
    printf("Max relative error: %e\n", max_err);
    printf("%s\n", max_err < 1e-3f ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;
    return 0;
}
