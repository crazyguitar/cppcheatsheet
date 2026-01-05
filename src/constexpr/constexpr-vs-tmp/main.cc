#include <gtest/gtest.h>

template <long N>
struct Fib {
  static constexpr long value = Fib<N - 1>::value + Fib<N - 2>::value;
};

template <>
struct Fib<0> {
  static constexpr long value = 0;
};

template <>
struct Fib<1> {
  static constexpr long value = 1;
};

constexpr long fib(long n) { return (n < 2) ? n : fib(n - 1) + fib(n - 2); }

TEST(ConstexprVsTmp, TemplateMetaprogramming) {
  constexpr long result = Fib<20>::value;
  EXPECT_EQ(result, 6765);
}

TEST(ConstexprVsTmp, Constexpr) {
  constexpr long result = fib(20);
  EXPECT_EQ(result, 6765);
}
