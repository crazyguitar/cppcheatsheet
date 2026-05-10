#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

TEST(ThrustTransform, Negate) {
  thrust::device_vector<int> a(4, 10);
  thrust::device_vector<int> c(4);

  thrust::transform(a.begin(), a.end(), c.begin(), thrust::negate<int>());

  thrust::host_vector<int> result = c;
  EXPECT_EQ(result[0], -10);
}

TEST(ThrustTransform, BinaryAdd) {
  thrust::device_vector<int> a(4, 10);
  thrust::device_vector<int> b(4, 5);
  thrust::device_vector<int> c(4);

  thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<int>());

  thrust::host_vector<int> result = c;
  EXPECT_EQ(result[0], 15);
}

struct square_functor {
  __host__ __device__ int operator()(int x) const { return x * x; }
};

TEST(ThrustTransform, CustomFunctor) {
  thrust::device_vector<int> a(4, 10);
  thrust::device_vector<int> c(4);

  thrust::transform(a.begin(), a.end(), c.begin(), square_functor());

  thrust::host_vector<int> result = c;
  EXPECT_EQ(result[0], 100);
}
