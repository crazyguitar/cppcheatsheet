#include <gtest/gtest.h>

TEST(BinaryConst, Literals) {
  int a = 0b1010;
  int b = 0b11110000;

  EXPECT_EQ(a, 10);
  EXPECT_EQ(b, 240);
  EXPECT_EQ(0b1111, 15);
}
