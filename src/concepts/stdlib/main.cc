#include <gtest/gtest.h>

#include <concepts>
#include <functional>
#include <string>

TEST(ConceptsStdlib, CoreConcepts) {
  static_assert(std::same_as<int, int>);
  static_assert(!std::same_as<int, long>);
  static_assert(std::convertible_to<int, double>);
}

TEST(ConceptsStdlib, ComparisonConcepts) {
  static_assert(std::equality_comparable<int>);
  static_assert(std::totally_ordered<double>);
}

TEST(ConceptsStdlib, ObjectConcepts) {
  static_assert(std::movable<std::string>);
  static_assert(std::copyable<int>);
  static_assert(std::regular<int>);
}

template <std::invocable<int> F>
int call_with_42(F&& f) {
  return f(42);
}

TEST(ConceptsStdlib, CallableConcepts) {
  auto square = [](int x) { return x * x; };
  EXPECT_EQ(call_with_42(square), 1764);
}
