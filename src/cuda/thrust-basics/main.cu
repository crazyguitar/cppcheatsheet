#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

TEST(ThrustBasics, HostToDevice) {
  thrust::host_vector<int> h_vec(4);
  h_vec[0] = 10;
  h_vec[1] = 20;
  h_vec[2] = 30;
  h_vec[3] = 40;

  thrust::device_vector<int> d_vec = h_vec;

  thrust::host_vector<int> result = d_vec;
  EXPECT_EQ(result[0], 10);
  EXPECT_EQ(result[3], 40);
}

TEST(ThrustBasics, DeviceVectorInit) {
  thrust::device_vector<float> d_vec(100, 1.0f);
  EXPECT_EQ(d_vec.size(), 100);

  float sum = thrust::reduce(d_vec.begin(), d_vec.end());
  EXPECT_FLOAT_EQ(sum, 100.0f);
}
