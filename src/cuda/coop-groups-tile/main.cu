// Cooperative Groups Tile - Manual warp reduction with shfl_down
//
// Shows tiled_partition for warp-level programming with explicit shuffle.
// Useful when you need more control than cg::reduce() provides.
//
// Shuffle-Down Reduction in Registers (32 threads):
// =================================================
//
//   Each thread holds value in register (no shared memory):
//
//   offset=16: Each thread T[i] reads T[i+16]'s register via shuffle
//   +----+----+----+----+----+----+----+----+
//   | T0 | T1 | T2 |... |T15 |T16 |T17 |... |
//   | 1  | 1  | 1  |... | 1  | 1  | 1  |... |
//   +----+----+----+----+----+----+----+----+
//     |    |              |    ^    ^
//     |    +--------------|----|----+  shfl_down(val, 16)
//     +-------------------+----+
//   +----+----+----+----+----+----+----+----+
//   | 2  | 2  | 2  |... | 2  | 1  | 1  |... |  T0-T15 have partial sums
//   +----+----+----+----+----+----+----+----+
//
//   offset=8:  T[i] += T[i+8]   -> T0-T7 have sums of 4
//   offset=4:  T[i] += T[i+4]   -> T0-T3 have sums of 8
//   offset=2:  T[i] += T[i+2]   -> T0-T1 have sums of 16
//   offset=1:  T[i] += T[i+1]   -> T0 has sum of 32
//
//   +----+
//   | 32 | <- T0 writes to output[warp_idx]
//   +----+

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

__global__ void tile_reduce_kernel(const float* input, float* output, int n) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? input[idx] : 0.0f;

  // Manual warp reduction using shuffle
  for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
    val += warp.shfl_down(val, offset);
  }

  // Thread 0 of each warp writes partial result
  if (warp.thread_rank() == 0) {
    int warp_idx = blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank();
    output[warp_idx] = val;
  }
}

TEST(CUDA, CoopGroupsTile) {
  constexpr int n = 256;
  constexpr int threads = 128;
  constexpr int blocks = (n + threads - 1) / threads;
  constexpr int warps_per_block = threads / 32;
  constexpr int total_warps = blocks * warps_per_block;

  float *h_input, *h_output;
  float *d_input, *d_output;

  cudaMallocHost(&h_input, n * sizeof(float));
  cudaMallocHost(&h_output, total_warps * sizeof(float));
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, total_warps * sizeof(float));

  for (int i = 0; i < n; i++) {
    h_input[i] = 1.0f;
  }

  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  tile_reduce_kernel<<<blocks, threads>>>(d_input, d_output, n);

  cudaMemcpy(h_output, d_output, total_warps * sizeof(float), cudaMemcpyDeviceToHost);

  // Each warp sums 32 elements = 32.0f
  float total = 0.0f;
  for (int i = 0; i < total_warps; i++) {
    EXPECT_FLOAT_EQ(h_output[i], 32.0f);
    total += h_output[i];
  }
  EXPECT_FLOAT_EQ(total, static_cast<float>(n));

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}
