// Vector Addition - Basic parallel computation pattern
//
// This example shows the most common CUDA pattern: mapping each thread to one
// element of an array. The kernel computes c[i] = a[i] + b[i] in parallel,
// with thousands of threads executing simultaneously.
//
// Key concepts:
// - Thread index calculation: blockIdx.x * blockDim.x + threadIdx.x
// - Bounds checking (i < n) prevents out-of-bounds memory access
// - cudaMallocHost() allocates pinned (page-locked) host memory for faster
//   transfers between host and device
// - cudaMemcpyAsync() performs non-blocking transfers when using streams
//
// Pitfalls:
// - Forgetting bounds check causes undefined behavior when n isn't divisible
//   by block size
// - Using regular malloc() instead of cudaMallocHost() prevents async transfers
//   from overlapping with computation
// - Must synchronize stream before reading results on host

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

struct VectorAddTest {
  static constexpr int n = 1024;
  static constexpr size_t size = n * sizeof(float);

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  cudaStream_t stream;

  VectorAddTest() {
    cudaStreamCreate(&stream);
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    for (int i = 0; i < n; i++) {
      h_a[i] = static_cast<float>(i);
      h_b[i] = static_cast<float>(i * 2);
    }
  }

  ~VectorAddTest() {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaStreamDestroy(stream);
  }

  void run() {
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};

TEST(CUDA, VectorAdd) {
  VectorAddTest test;
  test.run();

  for (int i = 0; i < VectorAddTest::n; i++) {
    EXPECT_FLOAT_EQ(test.h_c[i], test.h_a[i] + test.h_b[i]);
  }
}
