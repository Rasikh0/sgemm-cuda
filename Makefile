.RECIPEPREFIX = >
NVCC      = nvcc
ARCH      = sm_75
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Wno-deprecated-gpu-targets
LIBS      = -lcublas

SOURCES  = $(wildcard src/*.cu)
BINARIES = $(patsubst src/%.cu,bin/%,$(SOURCES))

all: $(BINARIES)

bin/%: src/%.cu
> @mkdir -p bin
> $(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

clean:
> rm -rf bin

.PHONY: all clean
