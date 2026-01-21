#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

struct saxpy_functor {
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(float x, float y) const { return a * x + y; }
};

TEST(ThrustCustom, Saxpy) {
  thrust::device_vector<float> x(4, 1.0f);
  thrust::device_vector<float> y(4, 2.0f);

  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(3.0f));

  thrust::host_vector<float> result = y;
  EXPECT_FLOAT_EQ(result[0], 5.0f);  // 3*1 + 2 = 5
  EXPECT_FLOAT_EQ(result[3], 5.0f);
}
