#include <gtest/gtest.h>

#include <concepts>

template <typename T>
  requires std::integral<T>
T add(T a, T b) {
  return a + b;
}

TEST(RequiresBasic, IntegralTypes) {
  EXPECT_EQ(add(1, 2), 3);        // OK: int is integral
  EXPECT_EQ(add(10L, 20L), 30L);  // OK: long is integral
  // add(1.0, 2.0);  // Error: double is not integral
}
