#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

TEST(Bounds, LowerBound) {
  std::vector<int> v{1, 2, 3, 4, 5, 7, 10};
  auto it = std::lower_bound(v.begin(), v.end(), 5);
  EXPECT_EQ(*it, 5);
}

TEST(Bounds, UpperBound) {
  std::vector<int> v{1, 2, 3, 4, 5, 7, 10};
  auto it = std::upper_bound(v.begin(), v.end(), 5);
  EXPECT_EQ(*it, 7);
}

TEST(Bounds, InsertSorted) {
  std::vector<int> v{1, 2, 4, 5};
  auto pos = std::upper_bound(v.begin(), v.end(), 3);
  v.insert(pos, 3);
  EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(Bounds, EraseSorted) {
  std::vector<int> v{1, 2, 3, 4, 5};
  auto pos = std::lower_bound(v.begin(), v.end(), 3);
  v.erase(pos);
  EXPECT_EQ(v, (std::vector<int>{1, 2, 4, 5}));
}
