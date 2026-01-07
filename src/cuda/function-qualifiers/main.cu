// Function Qualifiers: __global__, __device__, __host__
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Device-only helper function
__device__ int square(int x) { return x * x; }

// Dual compilation: works on both host and device
__host__ __device__ int add(int a, int b) { return a + b; }

// Kernel function: called from host, runs on device
__global__ void compute(int* out, int a, int b) {
  out[0] = add(a, b);
  out[1] = square(a);
}

TEST(FunctionQualifiers, DeviceAndHostDevice) {
  int *d_out, h_out[2];
  cudaMalloc(&d_out, 2 * sizeof(int));

  compute<<<1, 1>>>(d_out, 3, 4);
  cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_out[0], 7);  // add(3, 4)
  EXPECT_EQ(h_out[1], 9);  // square(3)

  // __host__ __device__ can also be called on host
  EXPECT_EQ(add(5, 6), 11);

  cudaFree(d_out);
}
