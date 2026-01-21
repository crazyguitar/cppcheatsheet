#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <climits>

TEST(ThrustReduce, Sum) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 1;
  d_vec[1] = 2;
  d_vec[2] = 3;
  d_vec[3] = 4;

  int sum = thrust::reduce(d_vec.begin(), d_vec.end());
  EXPECT_EQ(sum, 10);
}

TEST(ThrustReduce, SumWithInit) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 1;
  d_vec[1] = 2;
  d_vec[2] = 3;
  d_vec[3] = 4;

  int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 100);
  EXPECT_EQ(sum, 110);
}

TEST(ThrustReduce, Product) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 1;
  d_vec[1] = 2;
  d_vec[2] = 3;
  d_vec[3] = 4;

  int product = thrust::reduce(d_vec.begin(), d_vec.end(), 1, thrust::multiplies<int>());
  EXPECT_EQ(product, 24);
}

TEST(ThrustReduce, Maximum) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 1;
  d_vec[1] = 2;
  d_vec[2] = 3;
  d_vec[3] = 4;

  int max_val = thrust::reduce(d_vec.begin(), d_vec.end(), INT_MIN, thrust::maximum<int>());
  EXPECT_EQ(max_val, 4);
}
