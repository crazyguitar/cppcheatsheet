// Cooperative Groups Reduce - Built-in collective operations
//
// Demonstrates cg::reduce() for warp-level reduction without shared memory.
// Much simpler than manual shuffle-based reduction.
//
// Warp-Level Reduction (no shared memory needed):
// ===============================================
//
//   Warp registers (32 threads, each holds one value):
//   +----+----+----+----+----+----+----+----+----+----+----+----+
//   | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | .. | T30| T31|
//   | v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | .. | v30| v31|
//   +----+----+----+----+----+----+----+----+----+----+----+----+
//
//   cg::reduce(warp, val, cg::plus<float>()) internally:
//
//   Step 1: shfl_down by 16    Step 2: shfl_down by 8
//   T0 += T16, T1 += T17...    T0 += T8, T1 += T9...
//   +----+----+----+           +----+----+----+
//   |sum |sum |... |           |sum |sum |... |
//   |0-16|1-17|    |           |0-24|1-25|    |
//   +----+----+----+           +----+----+----+
//
//   Step 3-5: shfl_down by 4, 2, 1
//   +----+
//   | T0 | <- contains sum of all 32 values
//   +----+
//
//   Only T0 (thread_rank == 0) does atomicAdd to global output

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? input[idx] : 0.0f;

  // Warp-level reduction using cooperative groups
  float warp_sum = cg::reduce(warp, val, cg::plus<float>());

  // First thread in each warp accumulates result
  if (warp.thread_rank() == 0) {
    atomicAdd(output, warp_sum);
  }
}

TEST(CUDA, CoopGroupsReduce) {
  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);

  float *h_input, *h_output;
  float *d_input, *d_output;

  cudaMallocHost(&h_input, size);
  cudaMallocHost(&h_output, sizeof(float));
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, sizeof(float));

  for (int i = 0; i < n; i++) {
    h_input[i] = 1.0f;
  }
  *h_output = 0.0f;

  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  warp_reduce_kernel<<<blocks, threads>>>(d_input, d_output, n);

  cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_FLOAT_EQ(*h_output, static_cast<float>(n));

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}
