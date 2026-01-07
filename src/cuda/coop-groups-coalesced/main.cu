// Cooperative Groups Coalesced - Dynamic thread groups
//
// coalesced_threads() creates a group from currently active threads.
// Useful when threads diverge but still need to cooperate.
//
// Coalesced Group Formation:
// ==========================
//
//   Warp with divergent branches (8 threads shown):
//   +----+----+----+----+----+----+----+----+
//   | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
//   | -1 | +2 |  0 | -4 | +5 |  0 | -7 | +8 |
//   +----+----+----+----+----+----+----+----+
//
//   if (val > 0) branch - active threads form coalesced group:
//   +----+----+----+
//   | T1 | T4 | T7 |  coalesced_group.size() = 3
//   | +2 | +5 | +8 |  thread_rank: 0, 1, 2
//   +----+----+----+
//        |
//        v  cg::reduce() among these 3 threads only
//   +----+
//   | 15 | -> atomicAdd(pos_sum, 15)
//   +----+
//
//   if (val < 0) branch - different coalesced group:
//   +----+----+----+
//   | T0 | T3 | T6 |  coalesced_group.size() = 3
//   | -1 | -4 | -7 |  thread_rank: 0, 1, 2
//   +----+----+----+
//        |
//        v  cg::reduce() among these 3 threads only
//   +----+
//   |-12 | -> atomicAdd(neg_sum, -12)
//   +----+
//
//   T2, T5 (val == 0) skip both branches, not in any coalesced group

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

__global__ void coalesced_kernel(const int* input, int* pos_sum, int* neg_sum, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int val = input[idx];

  if (val > 0) {
    // Group of threads with positive values
    cg::coalesced_group active = cg::coalesced_threads();
    int sum = cg::reduce(active, val, cg::plus<int>());
    if (active.thread_rank() == 0) {
      atomicAdd(pos_sum, sum);
    }
  } else if (val < 0) {
    // Group of threads with negative values
    cg::coalesced_group active = cg::coalesced_threads();
    int sum = cg::reduce(active, val, cg::plus<int>());
    if (active.thread_rank() == 0) {
      atomicAdd(neg_sum, sum);
    }
  }
}

TEST(CUDA, CoopGroupsCoalesced) {
  constexpr int n = 256;

  int *h_input, *h_pos, *h_neg;
  int *d_input, *d_pos, *d_neg;

  cudaMallocHost(&h_input, n * sizeof(int));
  cudaMallocHost(&h_pos, sizeof(int));
  cudaMallocHost(&h_neg, sizeof(int));
  cudaMalloc(&d_input, n * sizeof(int));
  cudaMalloc(&d_pos, sizeof(int));
  cudaMalloc(&d_neg, sizeof(int));

  int expected_pos = 0, expected_neg = 0;
  for (int i = 0; i < n; i++) {
    h_input[i] = (i % 3 == 0) ? -(i + 1) : (i % 3 == 1) ? (i + 1) : 0;
    if (h_input[i] > 0) expected_pos += h_input[i];
    if (h_input[i] < 0) expected_neg += h_input[i];
  }
  *h_pos = 0;
  *h_neg = 0;

  cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, h_pos, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neg, h_neg, sizeof(int), cudaMemcpyHostToDevice);

  coalesced_kernel<<<1, n>>>(d_input, d_pos, d_neg, n);

  cudaMemcpy(h_pos, d_pos, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_neg, d_neg, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(*h_pos, expected_pos);
  EXPECT_EQ(*h_neg, expected_neg);

  cudaFree(d_input);
  cudaFree(d_pos);
  cudaFree(d_neg);
  cudaFreeHost(h_input);
  cudaFreeHost(h_pos);
  cudaFreeHost(h_neg);
}
