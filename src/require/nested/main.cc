#include <gtest/gtest.h>

#include <concepts>
#include <string>

// Ad-hoc constraint: T must support + and * returning T
template <typename T>
  requires requires(T x) {
    { x + x } -> std::convertible_to<T>;
    { x * x } -> std::convertible_to<T>;
  }
T square_sum(T a, T b) {
  return (a + b) * (a + b);
}

TEST(RequiresNested, Arithmetic) {
  EXPECT_EQ(square_sum(2, 3), 25);               // OK: int supports + and *
  EXPECT_DOUBLE_EQ(square_sum(1.5, 2.5), 16.0);  // OK: double supports + and *
  // square_sum(std::string("a"), std::string("b"));  // Error: string has no *
}
