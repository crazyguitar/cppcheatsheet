#include <gtest/gtest.h>

#include <ranges>
#include <vector>

TEST(Transform, Square) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> result;

  for (auto x : v | std::views::transform([](int x) { return x * x; })) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{1, 4, 9, 16, 25}));
}
