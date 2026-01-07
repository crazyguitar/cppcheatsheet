#include <gtest/gtest.h>

#include <ranges>
#include <vector>

TEST(TakeDrop, Take) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> result;

  for (auto x : v | std::views::take(3)) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{1, 2, 3}));
}

TEST(TakeDrop, Drop) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> result;

  for (auto x : v | std::views::drop(3)) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{4, 5}));
}

TEST(TakeDrop, Slice) {
  std::vector<int> v{1, 2, 3, 4, 5, 6, 7};
  std::vector<int> result;

  for (auto x : v | std::views::drop(2) | std::views::take(3)) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{3, 4, 5}));
}
