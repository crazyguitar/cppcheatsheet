#include <gtest/gtest.h>

TEST(DeducingThis, RecursiveLambda) {
  auto fib = [](this auto&& self, long n) -> long { return n < 2 ? n : self(n - 1) + self(n - 2); };
  EXPECT_EQ(fib(10), 55);
}
