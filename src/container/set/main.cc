#include <gtest/gtest.h>

#include <set>

TEST(Set, InsertAndErase) {
  std::set<int> s{3, 1, 4, 1, 5};

  EXPECT_EQ(s.size(), 4);  // Duplicates ignored
  s.insert(2);
  EXPECT_EQ(s.size(), 5);
  s.erase(3);
  EXPECT_EQ(s.count(3), 0);
}

TEST(Set, Find) {
  std::set<int> s{1, 2, 3};

  EXPECT_NE(s.find(2), s.end());
  EXPECT_EQ(s.find(99), s.end());
}

TEST(Set, Ordered) {
  std::set<int> s{3, 1, 4, 1, 5};

  auto it = s.begin();
  EXPECT_EQ(*it++, 1);
  EXPECT_EQ(*it++, 3);
  EXPECT_EQ(*it++, 4);
  EXPECT_EQ(*it++, 5);
}
