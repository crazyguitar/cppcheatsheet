#include <gtest/gtest.h>

TEST(NestedFunc, LambdaEquivalent) {
  int multiplier = 10;
  auto multiply = [&](int x) { return x * multiplier; };

  EXPECT_EQ(multiply(5), 50);
  multiplier = 20;
  EXPECT_EQ(multiply(5), 100);
}
