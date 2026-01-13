#include <gtest/gtest.h>

#include <concepts>

// Both constraints must be satisfied
template <typename T>
  requires std::integral<T> && std::signed_integral<T>
T abs_val(T x) {
  return x < 0 ? -x : x;
}

TEST(RequiresConjunction, SignedIntegral) {
  EXPECT_EQ(abs_val(-5), 5);        // OK: int is signed integral
  EXPECT_EQ(abs_val(10), 10);       // OK: int is signed integral
  EXPECT_EQ(abs_val(-100L), 100L);  // OK: long is signed integral
  // abs_val(5u);  // Error: unsigned int is not signed
}
