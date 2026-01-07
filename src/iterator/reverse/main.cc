#include <gtest/gtest.h>

#include <ranges>
#include <string>
#include <vector>

TEST(Reverse, Vector) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> result;

  for (auto x : v | std::views::reverse) {
    result.push_back(x);
  }

  EXPECT_EQ(result, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(Reverse, String) {
  std::string s = "hello";
  std::string result;

  for (auto c : s | std::views::reverse) {
    result += c;
  }

  EXPECT_EQ(result, "olleh");
}
