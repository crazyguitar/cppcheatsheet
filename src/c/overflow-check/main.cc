#include <gtest/gtest.h>

#include <climits>
#include <cstdint>

TEST(OverflowCheck, DetectsOverflow) {
  int a = INT_MAX, b = 1, result;

  EXPECT_TRUE(__builtin_add_overflow(a, b, &result));

  a = 100;
  b = 200;
  EXPECT_FALSE(__builtin_add_overflow(a, b, &result));
  EXPECT_EQ(result, 300);
}
