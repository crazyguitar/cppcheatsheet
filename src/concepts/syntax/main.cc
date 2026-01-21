#include <gtest/gtest.h>

#include <concepts>

// 1. Requires clause after template
template <typename T>
  requires std::integral<T>
T square1(T x) {
  return x * x;
}

// 2. Trailing requires clause
template <typename T>
T square2(T x)
  requires std::integral<T>
{
  return x * x;
}

// 3. Constrained template parameter
template <std::integral T>
T square3(T x) {
  return x * x;
}

// 4. Abbreviated function template
auto square4(std::integral auto x) { return x * x; }

TEST(ConceptsSyntax, AllVariations) {
  EXPECT_EQ(square1(5), 25);
  EXPECT_EQ(square2(5), 25);
  EXPECT_EQ(square3(5), 25);
  EXPECT_EQ(square4(5), 25);
}
