#include <gtest/gtest.h>

#include <concepts>
#include <string>

// Either constraint can be satisfied
template <typename T>
  requires std::integral<T> || std::floating_point<T>
T twice(T x) {
  return x + x;
}

TEST(RequiresDisjunction, IntegralOrFloating) {
  EXPECT_EQ(twice(5), 10);            // OK: int is integral
  EXPECT_DOUBLE_EQ(twice(2.5), 5.0);  // OK: double is floating_point
  EXPECT_EQ(twice(100L), 200L);       // OK: long is integral
  // twice(std::string("x"));  // Error: string is neither
}
