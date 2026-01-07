// Hello World - Basic CUDA kernel launch
//
// This example demonstrates the fundamental CUDA programming model: launching
// a kernel function that executes in parallel across multiple threads. Each
// thread computes its unique index, showing how CUDA organizes threads into
// a grid of blocks.
//
// Key concepts:
// - __global__ declares a kernel function callable from host, runs on device
// - <<<blocks, threads>>> syntax specifies the execution configuration
// - blockIdx.x and threadIdx.x identify each thread's position in the grid
// - cudaStreamSynchronize() blocks until all kernels in the stream complete
//
// Pitfall: Without synchronization, the program may exit before kernel output
// appears. Always synchronize before reading results or exiting.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void hello_kernel(int* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  output[idx] = idx;
}

struct HelloTest {
  static constexpr int blocks = 2;
  static constexpr int threads = 4;
  static constexpr int n = blocks * threads;

  int* h_output;
  int* d_output;
  cudaStream_t stream;

  HelloTest() {
    cudaStreamCreate(&stream);
    cudaMallocHost(&h_output, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
  }

  ~HelloTest() {
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);
  }

  void run() {
    hello_kernel<<<blocks, threads, 0, stream>>>(d_output);
    cudaMemcpyAsync(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};

TEST(CUDA, HelloWorld) {
  HelloTest test;
  test.run();

  for (int i = 0; i < HelloTest::n; i++) {
    EXPECT_EQ(test.h_output[i], i);
  }
}
