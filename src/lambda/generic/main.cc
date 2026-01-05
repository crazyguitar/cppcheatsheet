#include <gtest/gtest.h>

#include <utility>

TEST(GenericLambda, VariadicSum) {
  auto sum = [](auto&&... args) { return (std::forward<decltype(args)>(args) + ...); };
  EXPECT_EQ(sum(1, 2, 3, 4, 5), 15);
  EXPECT_DOUBLE_EQ(sum(1.5, 2.5), 4.0);
}
