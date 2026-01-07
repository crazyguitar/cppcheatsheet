#include <gtest/gtest.h>

consteval int square(int x) { return x * x; }

TEST(Consteval, CompileTimeOnly) {
  constexpr int result = square(5);
  EXPECT_EQ(result, 25);
}

TEST(Consteval, UsedInConstexpr) {
  constexpr int a = square(3);
  constexpr int b = square(4);
  EXPECT_EQ(a + b, 25);  // 9 + 16
}
