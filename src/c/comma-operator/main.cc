#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

TEST(CommaOperator, ReturnsLastValue) {
  int a = 1, b = 2, c = 3;
  int i = (a, b, c);
  EXPECT_EQ(i, 3);

  i = (a + 5, a + b);
  EXPECT_EQ(i, 3);
}

#pragma GCC diagnostic pop
