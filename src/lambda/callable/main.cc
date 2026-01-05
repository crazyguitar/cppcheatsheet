#include <gtest/gtest.h>

#include <functional>

class Fib {
 public:
  long operator()(long n) const { return (n < 2) ? n : operator()(n - 1) + operator()(n - 2); }
};

TEST(Callable, Functor) {
  Fib fib;
  EXPECT_EQ(fib(10), 55);
}

TEST(Callable, Lambda) {
  std::function<long(long)> fib = [&](long n) { return (n < 2) ? n : fib(n - 1) + fib(n - 2); };
  EXPECT_EQ(fib(10), 55);
}
