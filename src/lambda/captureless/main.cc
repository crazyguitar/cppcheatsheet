#include <gtest/gtest.h>

TEST(Captureless, ToFunctionPointer) {
  int (*add)(int, int) = [](int a, int b) { return a + b; };
  EXPECT_EQ(add(2, 3), 5);
}
