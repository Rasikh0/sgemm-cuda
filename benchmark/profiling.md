Benchmark scripts and result charts go here.

# Profiling (Nsight Compute, Tesla T4, N = 4096)

| Version | L1 Throughput | Compute (SM) | DRAM | Achieved Occupancy | Load sectors/req | Store sectors/req |
|---|---|---|---|---|---|---|
| v0 | 99.98% | 11.76% | 3.48% | 99.75% | 8.5 | 16.0 |
| v1 | 91.94% | 61.29% | 18.03% | 99.89% | 2.0 | 4.0 |

## v0 → v1: memory coalescing

v0 and v1 issue the same number of memory instructions; the difference is how many memory transactions each one needs. In v0, `threadIdx.x` indexes rows, so the threads in a warp read and write addresses a full row apart (16 KB at N = 4096). Each warp request splits into many separate 32-byte sectors: 8.5 per load and 16 per store. In v1, `threadIdx.x` indexes columns, so neighboring threads touch neighboring floats and each request is served by 2 sectors per load and 4 per store, about 4× less traffic through the L1 cache. In v0 the L1 cache was saturated (~100%) while the arithmetic units sat mostly idle (12% compute throughput). With coalesced access, compute throughput rises to 61%, and the kernel runs 3.5× faster (120 → 426 GFLOPS).
