#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept Addable = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
};

template <typename T>
concept Number = Numeric<T> && Addable<T>;

template <Number T>
T add(T a, T b) {
  return a + b;
}

TEST(ConceptsDefining, NumericConcept) {
  static_assert(Numeric<int>);
  static_assert(Numeric<double>);
  static_assert(!Numeric<std::string>);
}

TEST(ConceptsDefining, AddableConcept) {
  static_assert(Addable<int>);
  static_assert(Addable<std::string>);  // string has operator+
}

TEST(ConceptsDefining, CompoundConcept) {
  EXPECT_EQ(add(1, 2), 3);
  EXPECT_DOUBLE_EQ(add(1.5, 2.5), 4.0);
}
