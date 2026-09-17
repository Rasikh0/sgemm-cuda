#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) { printf("No CUDA devices found.\n"); return 1; }
    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);
        printf("Device %d: %s\n", dev, p.name);
        printf("  Compute capability       : %d.%d  -> compile with -arch=sm_%d%d\n",
               p.major, p.minor, p.major, p.minor);
        printf("  Streaming multiprocessors: %d\n", p.multiProcessorCount);
        printf("  Global memory            : %.2f GB\n", p.totalGlobalMem / 1e9);
        printf("  Memory bus width         : %d-bit\n", p.memoryBusWidth);
        printf("  Memory clock             : %.0f MHz\n", p.memoryClockRate / 1000.0);
        printf("  GPU clock                : %.0f MHz\n", p.clockRate / 1000.0);
        printf("  Shared mem per block     : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  Shared mem per SM        : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per block      : %d\n", p.regsPerBlock);
        printf("  Warp size                : %d\n", p.warpSize);
        double bw = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6;
        printf("  Peak memory bandwidth    : %.1f GB/s\n", bw);
    }
    return 0;
}
