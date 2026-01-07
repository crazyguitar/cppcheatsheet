#include <gtest/gtest.h>

#include <array>

constexpr int square(int x) { return x * x; }

TEST(ConstexprVariable, ArraySize) {
  constexpr int size = square(4);
  std::array<int, size> arr{};
  EXPECT_EQ(arr.size(), 16);
}

TEST(ConstexprVariable, CompileTimeConstant) {
  constexpr int val = square(5);
  EXPECT_EQ(val, 25);
}
