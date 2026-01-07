// libcu++ Latch
// See: docs/notes/cuda/cuda_cpp.rst

#include <cooperative_groups.h>
#include <cuda/latch>
#include <gtest/gtest.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define CUDA_CHECK(exp)                                                                                              \
  do {                                                                                                               \
    cudaError_t err = (exp);                                                                                         \
    if (err != cudaSuccess) {                                                                                        \
      fprintf(stderr, "[%s:%d] %s failed: %s\n", __FILE__, __LINE__, #exp, cudaGetErrorString(err));                 \
      exit(1);                                                                                                       \
    }                                                                                                                \
  } while (0)

__global__ void latch_kernel(int* output, int threads_per_block) {
  // Use placement new for latch initialization
  __shared__ alignas(cuda::latch<cuda::thread_scope_block>) unsigned char lat_storage[sizeof(cuda::latch<cuda::thread_scope_block>)];
  __shared__ int shared_data[256];

  auto block = cg::this_thread_block();
  cuda::latch<cuda::thread_scope_block>* lat;

  if (block.thread_rank() == 0) {
    lat = new (lat_storage) cuda::latch<cuda::thread_scope_block>(threads_per_block);
  }
  block.sync();
  lat = reinterpret_cast<cuda::latch<cuda::thread_scope_block>*>(lat_storage);

  // Each thread writes its ID
  shared_data[threadIdx.x] = threadIdx.x;

  // Count down and wait
  lat->arrive_and_wait();

  // Verify all data is visible
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
      sum += shared_data[i];
    }
    output[blockIdx.x] = sum;
  }
}

TEST(LibcuxxLatch, ArriveAndWait) {
  const int blocks = 2;
  const int threads = 128;
  const int expected_sum = threads * (threads - 1) / 2;  // 0 + 1 + ... + 127

  int* d_output;
  CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(int)));

  latch_kernel<<<blocks, threads>>>(d_output, threads);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_output[blocks];
  CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(int), cudaMemcpyDeviceToHost));

  printf("Latch results: ");
  for (int i = 0; i < blocks; i++) {
    printf("%d ", h_output[i]);
    EXPECT_EQ(h_output[i], expected_sum);
  }
  printf("(expected all %d)\n", expected_sum);

  CUDA_CHECK(cudaFree(d_output));
}
