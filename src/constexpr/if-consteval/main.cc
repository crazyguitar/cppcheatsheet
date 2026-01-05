#include <gtest/gtest.h>

constexpr int compute(int x) {
  if consteval {
    return x * x;  // Compile-time path
  } else {
    return x * x;  // Runtime path
  }
}

TEST(IfConsteval, CompileTime) {
  constexpr int result = compute(5);
  EXPECT_EQ(result, 25);
}

TEST(IfConsteval, Runtime) {
  int x = 5;
  int result = compute(x);
  EXPECT_EQ(result, 25);
}
