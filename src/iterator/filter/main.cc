#include <gtest/gtest.h>

#include <ranges>
#include <vector>

TEST(Filter, EvenNumbers) {
  std::vector<int> v{1, 2, 3, 4, 5, 6};
  std::vector<int> result;

  for (auto x : v | std::views::filter([](int x) { return x % 2 == 0; })) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{2, 4, 6}));
}
