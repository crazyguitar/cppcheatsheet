#include <gtest/gtest.h>

#include <functional>

TEST(RecursiveLambda, StdFunction) {
  std::function<long(long)> fib = [&](long n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); };
  EXPECT_EQ(fib(10), 55);
}

TEST(RecursiveLambda, SelfPassing) {
  auto fib = [](auto&& self, long n) -> long { return n < 2 ? n : self(self, n - 1) + self(self, n - 2); };
  EXPECT_EQ(fib(fib, 10), 55);
}
