#include <gtest/gtest.h>

#include <deque>

TEST(Deque, PushFrontBack) {
  std::deque<int> d{2, 3, 4};

  d.push_front(1);
  EXPECT_EQ(d.front(), 1);

  d.push_back(5);
  EXPECT_EQ(d.back(), 5);

  EXPECT_EQ(d.size(), 5);
}

TEST(Deque, PopFrontBack) {
  std::deque<int> d{1, 2, 3, 4, 5};

  d.pop_front();
  EXPECT_EQ(d.front(), 2);

  d.pop_back();
  EXPECT_EQ(d.back(), 4);
}
