#include <gtest/gtest.h>

TEST(LambdaSyntax, CaptureByValue) {
  int x = 10;
  auto f = [x]() { return x * 2; };
  EXPECT_EQ(f(), 20);
}

TEST(LambdaSyntax, CaptureByRef) {
  int x = 10;
  auto f = [&x]() { x *= 2; };
  f();
  EXPECT_EQ(x, 20);
}
