// Parallel Reduction - Sum reduction using shared memory
//
// This example demonstrates parallel reduction, a fundamental pattern for
// computing aggregates (sum, min, max) across large arrays. Each block reduces
// its portion to a single value, then results are combined on the host.
//
// Key concepts:
// - Tree-based reduction: each iteration halves active threads
// - Shared memory holds partial sums within each block
// - __syncthreads() ensures all threads complete each reduction step
// - Two-phase reduction: GPU reduces to block count, CPU finishes
//
// Pitfalls:
// - Missing __syncthreads() causes race conditions between reduction steps
// - Block size must be power of 2 for this simple reduction pattern
// - Warp divergence in final iterations (tid < s) reduces efficiency
// - For large arrays, consider multi-pass GPU reduction instead of CPU finish
//
// Optimization notes:
// - Warp-level primitives (__shfl_down_sync) avoid __syncthreads for last 32
// - Sequential addressing (used here) avoids bank conflicts
// - First load can combine two elements to improve occupancy

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum(const float* input, float* output, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

struct ReductionTest {
  static constexpr int n = 1024 * 1024;
  static constexpr size_t size = n * sizeof(float);
  int blocks;

  float *h_input, *h_partial;
  float *d_input, *d_output;
  cudaStream_t stream;
  float result;

  ReductionTest() : blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE), result(0.0f) {
    cudaStreamCreate(&stream);
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_partial, blocks * sizeof(float));
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, blocks * sizeof(float));

    for (int i = 0; i < n; i++) {
      h_input[i] = 1.0f;
    }
  }

  ~ReductionTest() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_partial);
    cudaStreamDestroy(stream);
  }

  void run() {
    cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);

    reduce_sum<<<blocks, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);

    cudaMemcpyAsync(h_partial, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result = 0.0f;
    for (int i = 0; i < blocks; i++) {
      result += h_partial[i];
    }
  }
};

TEST(CUDA, Reduction) {
  ReductionTest test;
  test.run();

  EXPECT_FLOAT_EQ(test.result, static_cast<float>(ReductionTest::n));
}
