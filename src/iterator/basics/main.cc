#include <gtest/gtest.h>

#include <iterator>
#include <vector>

TEST(Iterator, ForwardIteration) {
  std::vector<int> v{1, 2, 3, 4, 5};
  int sum = 0;
  for (auto it = v.begin(); it != v.end(); ++it) {
    sum += *it;
  }
  EXPECT_EQ(sum, 15);
}

TEST(Iterator, ReverseIteration) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> reversed;
  for (auto it = v.rbegin(); it != v.rend(); ++it) {
    reversed.push_back(*it);
  }
  EXPECT_EQ(reversed, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(Iterator, Advance) {
  std::vector<int> v{1, 2, 3, 4, 5};
  auto it = v.begin();
  std::advance(it, 2);
  EXPECT_EQ(*it, 3);
}

TEST(Iterator, Distance) {
  std::vector<int> v{1, 2, 3, 4, 5};
  auto dist = std::distance(v.begin(), v.end());
  EXPECT_EQ(dist, 5);
}
