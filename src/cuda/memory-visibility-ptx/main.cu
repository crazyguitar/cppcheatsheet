// PTX Memory Operations
//
// Demonstrates PTX inline assembly for memory ordering:
// - ld.acquire / st.release (GPU and system scope)
// - Spinlock with atom.acquire.cta.cas / atom.release.cta.exch
// See memory_visibility.rst "PTX Memory Operations" section.
// Reference: https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh

#include <cuda_runtime.h>
#include <gtest/gtest.h>

// --- Load/Store with ordering (GPU scope) ---
__device__ __forceinline__ int ld_acquire_gpu(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_release_gpu(int* ptr, int val) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

// --- Load/Store with ordering (system scope) ---
__device__ __forceinline__ int ld_acquire_sys(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_release_sys(int* ptr, int val) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

// --- PTX Spinlock (shared memory, CTA scope) ---
__device__ __forceinline__ int atomic_cas_acquire(int* addr, int expected, int desired) {
  int ret;
  asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(expected), "r"(desired) : "memory");
  return ret;
}

__device__ __forceinline__ int atomic_exch_release(int* addr, int val) {
  int ret;
  asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(val) : "memory");
  return ret;
}

__device__ __forceinline__ void acquire_lock(int* mutex) { while (atomic_cas_acquire(mutex, 0, 1) != 0); }

__device__ __forceinline__ void release_lock(int* mutex) { atomic_exch_release(mutex, 0); }

__global__ void spinlock_test(int* counter, int iterations) {
  __shared__ int mutex;
  if (threadIdx.x == 0) mutex = 0;
  __syncthreads();

  for (int i = 0; i < iterations; ++i) {
    acquire_lock(&mutex);
    int tmp = *counter;
    *counter = tmp + 1;
    release_lock(&mutex);
  }
}

__global__ void producer_ptx(int* data, int* flag) {
  *data = 100;
  st_release_gpu(flag, 1);
}

__global__ void consumer_ptx(int* data, int* flag, int* result) {
  while (ld_acquire_gpu(flag) == 0);
  *result = *data;
}

TEST(PTX, Spinlock) {
  int* d_counter;
  int h_counter = 0;

  cudaMalloc(&d_counter, sizeof(int));
  cudaMemset(d_counter, 0, sizeof(int));

  spinlock_test<<<1, 32>>>(d_counter, 100);
  cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_counter, 32 * 100);
  cudaFree(d_counter);
}

TEST(PTX, AcquireRelease) {
  int *d_data, *d_flag, *d_result;
  int h_result = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_flag, 0, sizeof(int));

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  producer_ptx<<<1, 1, 0, s1>>>(d_data, d_flag);
  consumer_ptx<<<1, 1, 0, s2>>>(d_data, d_flag, d_result);

  cudaDeviceSynchronize();
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_result, 100);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_result);
}
