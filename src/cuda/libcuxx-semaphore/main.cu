// libcu++ Semaphore
// See: docs/notes/cuda/cuda_cpp.rst

#include <gtest/gtest.h>

#include <cstdio>
#include <cuda/atomic>
#include <cuda/semaphore>

#define CUDA_CHECK(exp)                                                                              \
  do {                                                                                               \
    cudaError_t err = (exp);                                                                         \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s:%d] %s failed: %s\n", __FILE__, __LINE__, #exp, cudaGetErrorString(err)); \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

__global__ void semaphore_kernel(
    cuda::counting_semaphore<cuda::thread_scope_device, 4>* sem,
    cuda::atomic<int, cuda::thread_scope_device>* max_concurrent,
    cuda::atomic<int, cuda::thread_scope_device>* current,
    int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  sem->acquire();

  // Track max concurrent threads in critical section
  int cur = current->fetch_add(1, cuda::memory_order_relaxed) + 1;
  int old_max = max_concurrent->load(cuda::memory_order_relaxed);
  while (cur > old_max && !max_concurrent->compare_exchange_weak(old_max, cur, cuda::memory_order_relaxed));

  // Simulate work
  for (volatile int i = 0; i < 1000; i++);

  current->fetch_sub(1, cuda::memory_order_relaxed);
  sem->release();
}

TEST(LibcuxxSemaphore, CountingSemaphore) {
  const int n = 1000;
  const int max_allowed = 4;

  cuda::counting_semaphore<cuda::thread_scope_device, 4>* d_sem;
  cuda::atomic<int, cuda::thread_scope_device>* d_max;
  cuda::atomic<int, cuda::thread_scope_device>* d_current;

  CUDA_CHECK(cudaMalloc(&d_sem, sizeof(*d_sem)));
  CUDA_CHECK(cudaMalloc(&d_max, sizeof(*d_max)));
  CUDA_CHECK(cudaMalloc(&d_current, sizeof(*d_current)));

  // Initialize semaphore with count 4
  cuda::counting_semaphore<cuda::thread_scope_device, 4> h_sem{4};
  CUDA_CHECK(cudaMemcpy(d_sem, &h_sem, sizeof(h_sem), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_max, 0, sizeof(*d_max)));
  CUDA_CHECK(cudaMemset(d_current, 0, sizeof(*d_current)));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  semaphore_kernel<<<blocks, threads>>>(d_sem, d_max, d_current, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  int max_observed;
  CUDA_CHECK(cudaMemcpy(&max_observed, d_max, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Semaphore: max concurrent = %d (limit %d)\n", max_observed, max_allowed);
  EXPECT_LE(max_observed, max_allowed);

  CUDA_CHECK(cudaFree(d_sem));
  CUDA_CHECK(cudaFree(d_max));
  CUDA_CHECK(cudaFree(d_current));
}
