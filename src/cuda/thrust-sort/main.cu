#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

TEST(ThrustSort, Ascending) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 30;
  d_vec[1] = 10;
  d_vec[2] = 40;
  d_vec[3] = 20;

  thrust::sort(d_vec.begin(), d_vec.end());

  thrust::host_vector<int> result = d_vec;
  EXPECT_EQ(result[0], 10);
  EXPECT_EQ(result[1], 20);
  EXPECT_EQ(result[2], 30);
  EXPECT_EQ(result[3], 40);
}

TEST(ThrustSort, Descending) {
  thrust::device_vector<int> d_vec(4);
  d_vec[0] = 30;
  d_vec[1] = 10;
  d_vec[2] = 40;
  d_vec[3] = 20;

  thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());

  thrust::host_vector<int> result = d_vec;
  EXPECT_EQ(result[0], 40);
  EXPECT_EQ(result[3], 10);
}

TEST(ThrustSort, ByKey) {
  thrust::device_vector<int> keys(4);
  keys[0] = 3;
  keys[1] = 1;
  keys[2] = 4;
  keys[3] = 2;

  thrust::device_vector<char> vals(4);
  vals[0] = 'c';
  vals[1] = 'a';
  vals[2] = 'd';
  vals[3] = 'b';

  thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());

  thrust::host_vector<int> h_keys = keys;
  thrust::host_vector<char> h_vals = vals;
  EXPECT_EQ(h_keys[0], 1);
  EXPECT_EQ(h_vals[0], 'a');
  EXPECT_EQ(h_keys[3], 4);
  EXPECT_EQ(h_vals[3], 'd');
}
