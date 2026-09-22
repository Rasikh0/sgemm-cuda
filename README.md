# SGEMM CUDA Optimization

Optimizing single-precision matrix multiply from a naive kernel toward cuBLAS, profiled at each step.

## Hardware
- GPU: NVIDIA Tesla T4 (compute capability 7.5, `-arch=sm_75`)
- 40 SMs · 15.64 GB · 256-bit bus · 320 GB/s peak bandwidth · ~8.1 TFLOPS FP32 · 48 KB shared mem/block

## Results (Tesla T4, N = 4096)

| Version | GFLOPS | % of cuBLAS |
|---|---|---|
| v0 | 120 | 3.2% |
| v1 | 426 | 10.4% |
| cuBLAS | ~4000 | 100% |
