#include <gtest/gtest.h>

import math;

TEST(ModulesBasic, Add) {
  EXPECT_EQ(add(1, 2), 3);
  EXPECT_EQ(add(-5, 5), 0);
}

TEST(ModulesBasic, Multiply) {
  EXPECT_EQ(multiply(3, 4), 12);
  EXPECT_EQ(multiply(0, 100), 0);
}
