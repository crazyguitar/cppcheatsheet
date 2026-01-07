#include <gtest/gtest.h>

#include <cmath>

#define abs_val(x) (std::is_same_v<decltype(x), int> ? std::abs((int)(x)) : std::fabs((double)(x)))

TEST(Generic, TypeSelection) {
  EXPECT_EQ(std::abs(-5), 5);
  EXPECT_DOUBLE_EQ(std::fabs(-3.14), 3.14);
}
