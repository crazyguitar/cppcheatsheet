#include <gtest/gtest.h>

#include <ranges>
#include <vector>

TEST(Iota, Bounded) {
  std::vector<int> result;
  for (auto i : std::views::iota(1, 6)) {
    result.push_back(i);
  }
  EXPECT_EQ(result, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(Iota, UnboundedWithTake) {
  std::vector<int> result;
  for (auto i : std::views::iota(1) | std::views::take(5)) {
    result.push_back(i);
  }
  EXPECT_EQ(result, (std::vector<int>{1, 2, 3, 4, 5}));
}
