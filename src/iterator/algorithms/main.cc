#include <gtest/gtest.h>

#include <algorithm>
#include <ranges>
#include <vector>

TEST(Algorithms, Sort) {
  std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};
  std::ranges::sort(v);
  EXPECT_EQ(v, (std::vector<int>{1, 1, 2, 3, 4, 5, 6, 9}));
}

TEST(Algorithms, Find) {
  std::vector<int> v{1, 2, 3, 4, 5};
  auto it = std::ranges::find(v, 3);
  EXPECT_NE(it, v.end());
  EXPECT_EQ(*it, 3);
}

TEST(Algorithms, AllOf) {
  std::vector<int> v{2, 4, 6, 8};
  bool all_even = std::ranges::all_of(v, [](int x) { return x % 2 == 0; });
  EXPECT_TRUE(all_even);
}

TEST(Algorithms, CountIf) {
  std::vector<int> v{1, 2, 3, 4, 5, 6};
  auto count = std::ranges::count_if(v, [](int x) { return x % 2 == 0; });
  EXPECT_EQ(count, 3);
}
