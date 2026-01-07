#include <gtest/gtest.h>

#include <unordered_set>

TEST(UnorderedSet, InsertAndErase) {
  std::unordered_set<int> s{3, 1, 4, 1, 5};

  EXPECT_EQ(s.size(), 4);
  s.insert(2);
  EXPECT_EQ(s.size(), 5);
  s.erase(3);
  EXPECT_EQ(s.count(3), 0);
}

TEST(UnorderedSet, Find) {
  std::unordered_set<int> s{1, 2, 3};

  EXPECT_NE(s.find(2), s.end());
  EXPECT_EQ(s.find(99), s.end());
}
