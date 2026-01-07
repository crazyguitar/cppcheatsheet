// cuda::std::atomic (libcu++) Cheatsheet
//
// Demonstrates cuda::atomic with thread_scope and memory_order.
// See memory_visibility.rst "cuda::std::atomic (libcu++)" section.
//
// Thread Scope Hierarchy (performance: block > device > system):
//
//   cuda::thread_scope_block  - Threads in same block (fastest)
//   cuda::thread_scope_device - All threads on GPU
//   cuda::thread_scope_system - GPU + CPU (slowest)
//
// Memory Orders (same as C++):
//
//   memory_order_relaxed  - Atomicity only, no ordering
//   memory_order_acquire  - Sees writes before matching release
//   memory_order_release  - Prior writes visible to acquire
//   memory_order_acq_rel  - Both acquire and release
//   memory_order_seq_cst  - Total ordering (default)

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Thread Scopes

__global__ void scope_block_test(int* result) {
  __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;
  if (threadIdx.x == 0) counter.store(0);
  __syncthreads();

  counter.fetch_add(1, cuda::std::memory_order_relaxed);
  __syncthreads();

  if (threadIdx.x == 0) *result = counter.load();
}

__global__ void scope_device_test(cuda::atomic<int, cuda::thread_scope_device>* counter) {
  counter->fetch_add(1, cuda::std::memory_order_relaxed);
}

// Memory Orders

__global__ void producer(int* data, cuda::atomic<int, cuda::thread_scope_device>* flag) {
  *data = 42;
  flag->store(1, cuda::std::memory_order_release);  // Prior writes visible
}

__global__ void consumer(int* data, cuda::atomic<int, cuda::thread_scope_device>* flag,
                         int* result) {
  while (flag->load(cuda::std::memory_order_acquire) == 0);  // Sees prior writes
  *result = *data;
}

// Compare-and-Swap (CAS)

__global__ void cas_test(cuda::atomic<int, cuda::thread_scope_device>* a, int* result) {
  int expected = 0;
  bool success = a->compare_exchange_strong(expected, 100);
  if (success) *result = 1;  // Only one thread succeeds
}

// Tests

TEST(LibcuAtomic, ScopeBlock) {
  int* d_result;
  int h_result = 0;
  cudaMalloc(&d_result, sizeof(int));

  scope_block_test<<<1, 32>>>(d_result);
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_result, 32);
  cudaFree(d_result);
}

TEST(LibcuAtomic, ScopeDevice) {
  cuda::atomic<int, cuda::thread_scope_device>* d_counter;
  cudaMalloc(&d_counter, sizeof(cuda::atomic<int, cuda::thread_scope_device>));
  cudaMemset(d_counter, 0, sizeof(int));

  scope_device_test<<<4, 32>>>(d_counter);

  int h_result;
  cudaMemcpy(&h_result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(h_result, 4 * 32);
  cudaFree(d_counter);
}

TEST(LibcuAtomic, AcquireRelease) {
  int *d_data, *d_result;
  cuda::atomic<int, cuda::thread_scope_device>* d_flag;
  int h_result = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(cuda::atomic<int, cuda::thread_scope_device>));
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

TEST(LibcuAtomic, CompareExchangeStrong) {
  cuda::atomic<int, cuda::thread_scope_device>* d_atomic;
  int* d_result;
  int h_result = 0;

  cudaMalloc(&d_atomic, sizeof(cuda::atomic<int, cuda::thread_scope_device>));
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_atomic, 0, sizeof(int));
  cudaMemset(d_result, 0, sizeof(int));

  cas_test<<<1, 32>>>(d_atomic, d_result);

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(h_result, 1);  // Exactly one thread succeeded

  int h_atomic;
  cudaMemcpy(&h_atomic, d_atomic, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(h_atomic, 100);  // Value was set

  cudaFree(d_atomic);
  cudaFree(d_result);
}
