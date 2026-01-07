// Cooperative Groups Cheatsheet - Quick reference examples
//
// Demonstrates all major cooperative groups operations in one file.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

__global__ void cheatsheet_kernel(float* output, int n) {
  // ==========================================================================
  // Create Groups
  // ==========================================================================
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(warp);

  // ==========================================================================
  // Group Properties
  // ==========================================================================
  int block_size = block.size();           // Number of threads in block
  int block_rank = block.thread_rank();    // Thread index in block (0 to size-1)
  int warp_size = warp.size();             // Always 32
  int warp_rank = warp.thread_rank();      // Thread index in warp (0 to 31)
  int warp_id = warp.meta_group_rank();    // Which warp in block
  int num_warps = warp.meta_group_size();  // Number of warps in block

  // Thread indexing (equivalent to blockIdx, blockDim, threadIdx)
  dim3 block_idx = block.group_index();
  dim3 block_dim = block.group_dim();
  dim3 thread_idx = block.thread_index();

  // ==========================================================================
  // Synchronization
  // ==========================================================================
  block.sync();  // __syncthreads() equivalent
  warp.sync();   // __syncwarp() equivalent
  tile.sync();   // Sync 4 threads

  // ==========================================================================
  // Collective Reductions (all threads get result)
  // ==========================================================================
  float val = static_cast<float>(warp_rank + 1);

  float sum = cg::reduce(warp, val, cg::plus<float>());         // Sum: 1+2+...+32 = 528
  float max_val = cg::reduce(warp, val, cg::greater<float>());  // Max: 32
  float min_val = cg::reduce(warp, val, cg::less<float>());     // Min: 1

  int bits = 1 << warp_rank;
  int all_or = cg::reduce(warp, bits, cg::bit_or<int>());  // All bits set

  // ==========================================================================
  // Shuffle Operations
  // ==========================================================================
  float broadcasted = warp.shfl(val, 0);        // Broadcast from thread 0
  float from_lane_5 = warp.shfl(val, 5);        // Get value from lane 5
  float shifted_down = warp.shfl_down(val, 1);  // Shift values down by 1
  float shifted_up = warp.shfl_up(val, 1);      // Shift values up by 1
  float xor_swap = warp.shfl_xor(val, 1);       // XOR swap with neighbor

  // ==========================================================================
  // Predicates (Vote Operations)
  // ==========================================================================
  bool is_even = (warp_rank % 2 == 0);
  bool any_even = warp.any(is_even);       // True (some threads are even)
  bool all_even = warp.all(is_even);       // False (not all are even)
  unsigned ballot = warp.ballot(is_even);  // Bitmask: 0x55555555

  // ==========================================================================
  // Coalesced Threads (dynamic grouping)
  // ==========================================================================
  if (warp_rank < 16) {
    cg::coalesced_group active = cg::coalesced_threads();
    int active_count = active.size();                                // 16 threads active
    int active_rank = active.thread_rank();                          // 0 to 15
    float active_sum = cg::reduce(active, 1.0f, cg::plus<float>());  // 16.0
  }

  // ==========================================================================
  // Store results for verification
  // ==========================================================================
  if (block_rank == 0) {
    output[0] = sum;                           // 528
    output[1] = max_val;                       // 32
    output[2] = min_val;                       // 1
    output[3] = broadcasted;                   // 1 (from thread 0)
    output[4] = from_lane_5;                   // 6 (lane 5 has value 6)
    output[5] = static_cast<float>(any_even);  // 1
    output[6] = static_cast<float>(all_even);  // 0
  }
}

TEST(CUDA, CoopGroupsCheatsheet) {
  constexpr int num_outputs = 7;
  float* d_output;
  float h_output[num_outputs];

  cudaMalloc(&d_output, num_outputs * sizeof(float));
  cudaMemset(d_output, 0, num_outputs * sizeof(float));

  cheatsheet_kernel<<<1, 128>>>(d_output, 128);

  cudaMemcpy(h_output, d_output, num_outputs * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_FLOAT_EQ(h_output[0], 528.0f);  // sum of 1+2+...+32
  EXPECT_FLOAT_EQ(h_output[1], 32.0f);   // max
  EXPECT_FLOAT_EQ(h_output[2], 1.0f);    // min
  EXPECT_FLOAT_EQ(h_output[3], 1.0f);    // broadcast from thread 0
  EXPECT_FLOAT_EQ(h_output[4], 6.0f);    // value from lane 5
  EXPECT_FLOAT_EQ(h_output[5], 1.0f);    // any_even = true
  EXPECT_FLOAT_EQ(h_output[6], 0.0f);    // all_even = false

  cudaFree(d_output);
}
