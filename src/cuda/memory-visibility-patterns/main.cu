// Common Patterns: Producer-Consumer
//
// Demonstrates producer-consumer with __threadfence + atomic.
// See memory_visibility.rst "Common Patterns" section.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void producer(int* data, int* flag) {
  *data = 42;
  __threadfence();
  *flag = 1;
}

__global__ void consumer(int* data, int* flag, int* result) {
  while (atomicAdd(flag, 0) == 0);
  __threadfence();
  *result = *data;
}

TEST(Pattern, ProducerConsumer) {
  int *d_data, *d_flag, *d_result;
  int h_result = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_flag, 0, sizeof(int));

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  producer<<<1, 1, 0, s1>>>(d_data, d_flag);
  consumer<<<1, 1, 0, s2>>>(d_data, d_flag, d_result);

  cudaDeviceSynchronize();
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_result, 42);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_result);
}
