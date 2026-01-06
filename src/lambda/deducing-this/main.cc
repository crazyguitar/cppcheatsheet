#include <gtest/gtest.h>

#include <functional>

TEST(DeducingThis, RecursiveLambda) {
#if __cpp_explicit_this_parameter >= 202110L
  // C++23 deducing this
  auto fib = [](this auto&& self, long n) -> long { return n < 2 ? n : self(n - 1) + self(n - 2); };
#else
  // C++14/17/20 fallback using std::function
  std::function<long(long)> fib = [&fib](long n) -> long { return n < 2 ? n : fib(n - 1) + fib(n - 2); };
#endif
  EXPECT_EQ(fib(10), 55);
}
