// CUDA Synchronization Primitives
//
// Demonstrates __syncthreads, __threadfence, and volatile behavior.
// See memory_visibility.rst "CUDA Synchronization Primitives" section.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

// __syncthreads: barrier + memory fence (block scope)
__global__ void syncthreads_test(int* output) {
  __shared__ int shared_data[32];
  int tid = threadIdx.x;

  if (tid == 0) shared_data[0] = 123;
  __syncthreads();  // All threads wait + memory visible
  if (tid == 1) *output = shared_data[0];
}

// volatile + __threadfence: volatile alone is NOT enough
__global__ void volatile_fence_test(volatile int* vflag, int* data, int* result) {
  int tid = threadIdx.x;
  if (tid == 0) {
    *data = 77;
    __threadfence();  // Hardware fence required!
    *vflag = 1;
  }
  __syncthreads();
  if (tid == 1) {
    while (*vflag == 0);  // volatile: compiler won't cache
    *result = *data;
  }
}

TEST(CUDAPrimitives, SyncThreads) {
  int* d_output;
  int h_output = 0;
  cudaMalloc(&d_output, sizeof(int));

  syncthreads_test<<<1, 32>>>(d_output);
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_output, 123);
  cudaFree(d_output);
}

TEST(CUDAPrimitives, VolatileFence) {
  int *d_data, *d_result;
  volatile int* d_vflag;
  int h_result = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc((void**)&d_vflag, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset((void*)d_vflag, 0, sizeof(int));

  volatile_fence_test<<<1, 32>>>(d_vflag, d_data, d_result);
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_result, 77);
  cudaFree(d_data);
  cudaFree((void*)d_vflag);
  cudaFree(d_result);
}
