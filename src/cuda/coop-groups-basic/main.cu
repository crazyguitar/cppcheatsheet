// Cooperative Groups Basic - Thread group hierarchy demonstration
//
// Shows how to create and use different thread group types:
// - thread_block: all threads in a block
// - tiled_partition: static partitions (warps, tiles)
// - Group properties: size(), thread_rank(), sync()
//
// Thread Block Layout (256 threads):
// ===================================
//
//   Block (thread_block)
//   +---------------------------------------------------------------+
//   | Warp 0 (tile<32>)      | Warp 1 (tile<32>)      | ...         |
//   | [T0  T1  T2  ... T31 ] | [T32 T33 T34 ... T63 ] | ...         |
//   +---------------------------------------------------------------+
//
//   Warp 0 subdivided into tile<4>:
//   +-------+-------+-------+-------+-------+-------+-------+-------+
//   | T0-T3 | T4-T7 | T8-11 | 12-15 | 16-19 | 20-23 | 24-27 | 28-31 |
//   +-------+-------+-------+-------+-------+-------+-------+-------+
//   tile[0]  tile[1] tile[2] tile[3] tile[4] tile[5] tile[6] tile[7]
//
//   Synchronization:
//   +-------------------+
//   | block.sync()      |  All 256 threads wait at barrier
//   +-------------------+
//            |
//            v
//   +-------------------+
//   | warp.sync()       |  32 threads in same warp wait
//   +-------------------+
//            |
//            v
//   +-------------------+
//   | tile4.sync()      |  4 threads in same tile wait
//   +-------------------+

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

__global__ void hierarchy_kernel(int* block_ranks, int* warp_ranks, int* tile_ranks, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Thread block group
  cg::thread_block block = cg::this_thread_block();

  // Partition into 32-thread warps
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  // Partition into 4-thread tiles
  cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(warp);

  // Store ranks at each level
  block_ranks[idx] = block.thread_rank();
  warp_ranks[idx] = warp.thread_rank();
  tile_ranks[idx] = tile4.thread_rank();

  // Synchronize at different levels
  tile4.sync();
  warp.sync();
  block.sync();
}

TEST(CUDA, CoopGroupsBasic) {
  constexpr int n = 256;
  constexpr size_t size = n * sizeof(int);

  int *h_block, *h_warp, *h_tile;
  int *d_block, *d_warp, *d_tile;

  cudaMallocHost(&h_block, size);
  cudaMallocHost(&h_warp, size);
  cudaMallocHost(&h_tile, size);
  cudaMalloc(&d_block, size);
  cudaMalloc(&d_warp, size);
  cudaMalloc(&d_tile, size);

  hierarchy_kernel<<<1, n>>>(d_block, d_warp, d_tile, n);

  cudaMemcpy(h_block, d_block, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_warp, d_warp, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_tile, d_tile, size, cudaMemcpyDeviceToHost);

  // Verify ranks
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(h_block[i], i);      // 0-255
    EXPECT_EQ(h_warp[i], i % 32);  // 0-31 repeating
    EXPECT_EQ(h_tile[i], i % 4);   // 0-3 repeating
  }

  cudaFree(d_block);
  cudaFree(d_warp);
  cudaFree(d_tile);
  cudaFreeHost(h_block);
  cudaFreeHost(h_warp);
  cudaFreeHost(h_tile);
}
