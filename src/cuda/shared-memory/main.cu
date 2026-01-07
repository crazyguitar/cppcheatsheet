// Shared Memory Stencil - Halo region pattern
//
// This example demonstrates the halo (ghost cell) pattern for stencil
// computations. Each output element depends on neighboring input elements,
// requiring threads to load extra "halo" elements at block boundaries.
//
// Key concepts:
// - Stencil operations access neighbors: output[i] = f(input[i-R:i+R])
// - Shared memory tile includes RADIUS extra elements on each side
// - Boundary threads load halo elements from neighboring regions
// - __syncthreads() ensures all data (including halos) is loaded
//
// Pitfalls:
// - Forgetting halo elements causes incorrect results at block boundaries
// - Halo loading logic must handle array boundaries (clamp or zero-pad)
// - Shared memory size = BLOCK_SIZE + 2*RADIUS, must fit in available memory
// - Missing __syncthreads() after halo load causes race conditions
//
// Memory layout in shared memory:
// [halo left][main elements][halo right]
// [0..RADIUS-1][RADIUS..RADIUS+BLOCK_SIZE-1][RADIUS+BLOCK_SIZE..end]

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#define BLOCK_SIZE 256
#define RADIUS 3

__global__ void stencil_1d(const float* input, float* output, int n) {
  __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

  int gindex = blockIdx.x * blockDim.x + threadIdx.x;
  int lindex = threadIdx.x + RADIUS;

  if (gindex < n) {
    temp[lindex] = input[gindex];
  }

  if (threadIdx.x < RADIUS) {
    int left = gindex - RADIUS;
    int right = gindex + BLOCK_SIZE;
    temp[lindex - RADIUS] = (left >= 0) ? input[left] : 0.0f;
    temp[lindex + BLOCK_SIZE] = (right < n) ? input[right] : 0.0f;
  }

  __syncthreads();

  if (gindex < n) {
    float result = 0.0f;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
      result += temp[lindex + offset];
    }
    output[gindex] = result / (2 * RADIUS + 1);
  }
}

struct SharedMemoryTest {
  static constexpr int n = 1024;
  static constexpr size_t size = n * sizeof(float);

  float *h_input, *h_output;
  float *d_input, *d_output;
  cudaStream_t stream;

  SharedMemoryTest() {
    cudaStreamCreate(&stream);
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    for (int i = 0; i < n; i++) {
      h_input[i] = static_cast<float>(i);
    }
  }

  ~SharedMemoryTest() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);
  }

  void run() {
    cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<blocks, BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);

    cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};

TEST(CUDA, SharedMemory) {
  SharedMemoryTest test;
  test.run();

  // Check middle element (index 100) - average of 97..103
  float expected = 0.0f;
  for (int i = -RADIUS; i <= RADIUS; i++) {
    expected += test.h_input[100 + i];
  }
  expected /= (2 * RADIUS + 1);

  EXPECT_NEAR(test.h_output[100], expected, 0.001f);
}
