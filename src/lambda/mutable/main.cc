#include <gtest/gtest.h>

TEST(MutableLambda, ModifiesCopy) {
  int counter = 0;
  auto inc = [counter]() mutable { return ++counter; };
  EXPECT_EQ(inc(), 1);
  EXPECT_EQ(inc(), 2);
  EXPECT_EQ(counter, 0);  // Original unchanged
}
