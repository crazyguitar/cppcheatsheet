// libcu++ Barrier
// See: docs/notes/cuda/cuda_cpp.rst

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cuda/barrier>

namespace cg = cooperative_groups;

#define CUDA_CHECK(exp)                                                                              \
  do {                                                                                               \
    cudaError_t err = (exp);                                                                         \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s:%d] %s failed: %s\n", __FILE__, __LINE__, #exp, cudaGetErrorString(err)); \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

__global__ void barrier_kernel(int* output) {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ int shared_sum;

  auto block = cg::this_thread_block();
  if (block.thread_rank() == 0) {
    init(&bar, block.size());
    shared_sum = 0;
  }
  block.sync();

  // Phase 1: each thread contributes
  atomicAdd(&shared_sum, 1);

  bar.arrive_and_wait();

  // Phase 2: all threads see the sum
  if (block.thread_rank() == 0) {
    output[blockIdx.x] = shared_sum;
  }
}

TEST(LibcuxxBarrier, ArriveAndWait) {
  const int blocks = 4;
  const int threads = 128;

  int* d_output;
  CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(int)));

  barrier_kernel<<<blocks, threads>>>(d_output);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_output[blocks];
  CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(int), cudaMemcpyDeviceToHost));

  printf("Barrier results: ");
  for (int i = 0; i < blocks; i++) {
    printf("%d ", h_output[i]);
    EXPECT_EQ(h_output[i], threads);
  }
  printf("(expected all %d)\n", threads);

  CUDA_CHECK(cudaFree(d_output));
}
