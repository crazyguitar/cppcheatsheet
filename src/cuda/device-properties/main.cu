#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(CUDA, DeviceCount) {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  EXPECT_GT(deviceCount, 0);
}

TEST(CUDA, DeviceProperties) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  EXPECT_GT(prop.totalGlobalMem, 0);
  EXPECT_GT(prop.multiProcessorCount, 0);
  EXPECT_EQ(prop.warpSize, 32);
  EXPECT_GT(prop.maxThreadsPerBlock, 0);
}
