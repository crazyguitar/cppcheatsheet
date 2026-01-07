#include <gtest/gtest.h>

constexpr long fib(long n) { return (n < 2) ? n : fib(n - 1) + fib(n - 2); }

TEST(ConstexprFunction, CompileTime) {
  constexpr long result = fib(10);
  EXPECT_EQ(result, 55);
}

TEST(ConstexprFunction, Runtime) {
  long n = 10;
  long result = fib(n);
  EXPECT_EQ(result, 55);
}
