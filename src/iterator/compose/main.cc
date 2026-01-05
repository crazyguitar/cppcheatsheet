#include <gtest/gtest.h>

#include <algorithm>
#include <ranges>
#include <vector>

TEST(Compose, FilterTransformTake) {
  std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> result;

  auto view = v | std::views::filter([](int x) { return x % 2 == 0; }) | std::views::transform([](int x) { return x * x; }) | std::views::take(3);

  for (auto x : view) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{4, 16, 36}));
}
