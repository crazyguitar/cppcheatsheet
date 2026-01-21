#include <gtest/gtest.h>

#include <concepts>
#include <vector>

template <typename T>
concept Container = requires(T c) {
  c.begin();
  c.end();
  c.size();
  typename T::value_type;
  typename T::iterator;
  { c.size() } -> std::convertible_to<std::size_t>;
};

TEST(ConceptsRequiresExpr, ContainerConcept) {
  static_assert(Container<std::vector<int>>);
  static_assert(!Container<int>);
}
