#include <gtest/gtest.h>

TEST(ConstexprLambda, CompileTime) {
  auto square = [](int x) { return x * x; };
  static_assert(square(5) == 25);
  EXPECT_EQ(square(5), 25);
}
