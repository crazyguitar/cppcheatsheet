#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cuda/barrier>

__global__ void split_arrive_wait_kernel(int* result) {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ int smem[256];
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0) init(&bar, block.size());
  block.sync();

  smem[threadIdx.x] = threadIdx.x * 2;
  auto token = bar.arrive();   // Get token for current phase
  bar.wait(std::move(token));  // Wait until all arrive

  if (threadIdx.x == 0) *result = smem[255];
}

__global__ void arrive_and_drop_kernel(int* result) {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ int sum;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0) {
    init(&bar, block.size());
    sum = 0;
  }
  block.sync();

  for (int i = 0; i < 3; ++i) {
    if (threadIdx.x == 0 && i == 1) {
      bar.arrive_and_drop();  // Exit early, reduce expected count
      return;
    }
    atomicAdd(&sum, 1);
    bar.arrive_and_wait();
  }
  if (threadIdx.x == 1) *result = sum;
}

class AsyncBarrierTest : public ::testing::Test {
 protected:
  int* d_result;
  void SetUp() override { cudaMalloc(&d_result, sizeof(int)); }
  void TearDown() override { cudaFree(d_result); }
};

TEST_F(AsyncBarrierTest, SplitArriveWait) {
  split_arrive_wait_kernel<<<1, 256>>>(d_result);
  int h_result;
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(h_result, 510);  // 255 * 2
}

TEST_F(AsyncBarrierTest, ArriveAndDrop) {
  arrive_and_drop_kernel<<<1, 32>>>(d_result);
  int h_result;
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  // Thread 0 exits after iteration 1, so: 32 + 31 + 31 = 94
  EXPECT_EQ(h_result, 94);
}
