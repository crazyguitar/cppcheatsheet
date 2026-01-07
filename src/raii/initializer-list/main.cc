#include <gtest/gtest.h>

#include <initializer_list>

template <typename T>
T sum(std::initializer_list<T> values) {
  T result = 0;
  for (const auto& v : values) {
    result += v;
  }
  return result;
}

TEST(InitializerList, SumIntegers) { EXPECT_EQ(sum({1, 2, 3, 4, 5}), 15); }

TEST(InitializerList, SumDoubles) { EXPECT_DOUBLE_EQ(sum({1.5, 2.5, 3.0}), 7.0); }

TEST(InitializerList, EmptyList) { EXPECT_EQ(sum<int>({}), 0); }
