// Memory Allocation Benchmark - Compare different CUDA memory types
//
// This benchmark measures transfer speeds for different memory allocation
// methods: pageable, pinned (cudaMallocHost), registered (cudaHostRegister),
// and managed (cudaMallocManaged).
//
// Run this on your system to get actual transfer speeds for the comparison
// table in the documentation.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>

constexpr int WARMUP = 10;
constexpr int ITERATIONS = 100;
constexpr size_t SIZE = 64 * 1024 * 1024;  // 64 MB

struct BenchmarkResult {
  float h2d_ms;
  float d2h_ms;
  float h2d_gbps;
  float d2h_gbps;
};

template <typename Derived>
struct MemoryBenchmark {
  float* h_data;
  float* d_data;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;

  MemoryBenchmark() {
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  virtual ~MemoryBenchmark() {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
  }

  BenchmarkResult run() {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
      cudaMemcpyAsync(d_data, h_data, SIZE, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(h_data, d_data, SIZE, cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    // Benchmark H2D
    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERATIONS; i++) {
      cudaMemcpyAsync(d_data, h_data, SIZE, cudaMemcpyHostToDevice, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float h2d_ms;
    cudaEventElapsedTime(&h2d_ms, start, stop);
    h2d_ms /= ITERATIONS;

    // Benchmark D2H
    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERATIONS; i++) {
      cudaMemcpyAsync(h_data, d_data, SIZE, cudaMemcpyDeviceToHost, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, start, stop);
    d2h_ms /= ITERATIONS;

    float size_gb = static_cast<float>(SIZE) / (1024.0f * 1024.0f * 1024.0f);
    return {h2d_ms, d2h_ms, size_gb / (h2d_ms / 1000.0f), size_gb / (d2h_ms / 1000.0f)};
  }

  void print(const char* name) {
    auto result = run();
    printf("%s:\n", name);
    printf("  H2D: %.2f ms (%.2f GB/s)\n", result.h2d_ms, result.h2d_gbps);
    printf("  D2H: %.2f ms (%.2f GB/s)\n\n", result.d2h_ms, result.d2h_gbps);
  }
};

struct PageableMemory : MemoryBenchmark<PageableMemory> {
  PageableMemory() {
    h_data = new float[SIZE / sizeof(float)];
    cudaMalloc(&d_data, SIZE);
  }
  ~PageableMemory() {
    cudaFree(d_data);
    delete[] h_data;
  }
};

struct PinnedMemory : MemoryBenchmark<PinnedMemory> {
  PinnedMemory() {
    cudaMallocHost(&h_data, SIZE);
    cudaMalloc(&d_data, SIZE);
  }
  ~PinnedMemory() {
    cudaFree(d_data);
    cudaFreeHost(h_data);
  }
};

struct RegisteredMemory : MemoryBenchmark<RegisteredMemory> {
  RegisteredMemory() {
    h_data = new float[SIZE / sizeof(float)];
    cudaHostRegister(h_data, SIZE, cudaHostRegisterDefault);
    cudaMalloc(&d_data, SIZE);
  }
  ~RegisteredMemory() {
    cudaFree(d_data);
    cudaHostUnregister(h_data);
    delete[] h_data;
  }
};

struct ManagedMemory : MemoryBenchmark<ManagedMemory> {
  ManagedMemory() {
    cudaMallocManaged(&h_data, SIZE);
    d_data = h_data;
  }
  ~ManagedMemory() { cudaFree(h_data); }
};

TEST(CUDA, MemoryBenchmark) {
  printf("\n=== CUDA Memory Allocation Benchmark (64 MB) ===\n\n");

  PageableMemory{}.print("Pageable Memory");
  PinnedMemory{}.print("Pinned Memory (cudaMallocHost)");
  RegisteredMemory{}.print("Registered Memory (cudaHostRegister)");
  ManagedMemory{}.print("Managed Memory (cudaMallocManaged)");

  EXPECT_TRUE(true);
}
