#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// v1: one thread per element of C. threadIdx.x selects the column.
__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void run_kernel(const float* dA, const float* dB, float* dC, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul<<<grid, block>>>(dA, dB, dC, N);
}

// cuBLAS is column-major; passing B then A gives the row-major result.
void run_cublas(cublasHandle_t h, const float* dA, const float* dB, float* dC, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, dB, N, dA, N, &beta, dC, N));
}

float max_rel_err(const float* out, const float* ref, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float rel = fabsf(out[i] - ref[i]) / (fabsf(ref[i]) + 1e-5f);
        if (rel > m) m = rel;
    }
    return m;
}

template <typename F>
float time_ms(F launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int sizes[] = {512, 1024, 2048, 4096};
    printf("v1\n");
    printf("%6s | %10s | %12s | %13s | %9s | %s\n",
           "N", "ms", "GFLOPS", "cuBLAS GFLOPS", "% cuBLAS", "check");
    printf("-------+------------+--------------+---------------+-----------+------\n");

    for (int N : sizes) {
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        float* hA = new float[count];
        float* hB = new float[count];
        float* hOut = new float[count];
        float* hCublas = new float[count];
        srand(42);
        for (size_t i = 0; i < count; ++i) {
            hA[i] = (float)rand() / RAND_MAX;
            hB[i] = (float)rand() / RAND_MAX;
        }

        float *dA, *dB, *dOut, *dCublas;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dOut, bytes));
        CUDA_CHECK(cudaMalloc(&dCublas, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

        run_kernel(dA, dB, dOut, N);
        run_cublas(handle, dA, dB, dCublas, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCublas, dCublas, bytes, cudaMemcpyDeviceToHost));

        bool ok;
        if (N <= 512) {
            float* hRef = new float[count];
            matmul_cpu(hA, hB, hRef, N);
            ok = max_rel_err(hOut, hRef, count) < 1e-3f &&
                 max_rel_err(hCublas, hRef, count) < 1e-3f;
            delete[] hRef;
        } else {
            ok = max_rel_err(hOut, hCublas, count) < 1e-3f;
        }

        float k_ms = time_ms([&]{ run_kernel(dA, dB, dOut, N); }, 2, 5);
        float c_ms = time_ms([&]{ run_cublas(handle, dA, dB, dCublas, N); }, 2, 5);

        double flops = 2.0 * N * N * N;
        double k_gflops = flops / (k_ms * 1e-3) / 1e9;
        double c_gflops = flops / (c_ms * 1e-3) / 1e9;

        printf("%6d | %10.3f | %12.1f | %13.1f | %8.1f%% | %s\n",
               N, k_ms, k_gflops, c_gflops, 100.0 * k_gflops / c_gflops,
               ok ? "PASS" : "FAIL");

        cudaFree(dA); cudaFree(dB); cudaFree(dOut); cudaFree(dCublas);
        delete[] hA; delete[] hB; delete[] hOut; delete[] hCublas;
    }

    cublasDestroy(handle);
    return 0;
}
