#include <gtest/gtest.h>

#include <set>

TEST(DefaultConstructible, SetComparator) {
  auto cmp = [](int a, int b) { return a > b; };
  std::set<int, decltype(cmp)> s;
  s.insert({3, 1, 4});
  EXPECT_EQ(*s.begin(), 4);  // Descending order
}
