// Tiled Matrix Multiplication - Shared memory optimization
//
// This example demonstrates how to use shared memory to optimize matrix
// multiplication. Naive matrix multiplication has poor memory access patterns
// because each thread reads entire rows/columns from global memory. Tiling
// loads small blocks into fast shared memory, reducing global memory bandwidth.
//
// Key concepts:
// - __shared__ declares memory shared by all threads in a block
// - Tiling: divide matrices into TILE_SIZE x TILE_SIZE blocks
// - Each thread loads one element into shared memory, then all threads
//   compute using the shared data
// - __syncthreads() ensures all threads finish loading before computation
//
// Pitfalls:
// - Missing __syncthreads() after loading causes race conditions - threads
//   may read uninitialized shared memory
// - Missing __syncthreads() before next tile load causes threads to overwrite
//   data still being used by other threads
// - Shared memory is limited (~48KB), tile size must fit within this limit
// - Bank conflicts can reduce performance if access patterns are poor

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#define TILE_SIZE 16

__global__ void matrix_mul(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
    if (row < n && t * TILE_SIZE + threadIdx.x < n)
      tile_a[threadIdx.y][threadIdx.x] = a[row * n + t * TILE_SIZE + threadIdx.x];
    else
      tile_a[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < n && t * TILE_SIZE + threadIdx.y < n)
      tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
    else
      tile_b[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

struct MatrixMulTest {
  static constexpr int n = 512;
  static constexpr size_t size = n * n * sizeof(float);

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  cudaStream_t stream;

  MatrixMulTest() {
    cudaStreamCreate(&stream);
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    for (int i = 0; i < n * n; i++) {
      h_a[i] = 1.0f;
      h_b[i] = 2.0f;
    }
  }

  ~MatrixMulTest() {
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    matrix_mul<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};

TEST(CUDA, MatrixMul) {
  MatrixMulTest test;
  test.run();

  float expected = static_cast<float>(MatrixMulTest::n) * 2.0f;
  EXPECT_FLOAT_EQ(test.h_c[0], expected);
  EXPECT_FLOAT_EQ(test.h_c[MatrixMulTest::n * MatrixMulTest::n - 1], expected);
}
