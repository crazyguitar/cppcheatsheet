#include <gtest/gtest.h>

#include <list>

TEST(List, PushFrontBack) {
  std::list<int> l{2, 3};

  l.push_front(1);
  EXPECT_EQ(l.front(), 1);

  l.push_back(4);
  EXPECT_EQ(l.back(), 4);
}

TEST(List, Insert) {
  std::list<int> l{1, 3};

  auto it = l.begin();
  std::advance(it, 1);
  l.insert(it, 2);

  auto check = l.begin();
  EXPECT_EQ(*check++, 1);
  EXPECT_EQ(*check++, 2);
  EXPECT_EQ(*check++, 3);
}

TEST(List, Remove) {
  std::list<int> l{1, 2, 2, 3};
  l.remove(2);
  EXPECT_EQ(l.size(), 2);
}

TEST(List, Sort) {
  std::list<int> l{3, 1, 2};
  l.sort();
  auto it = l.begin();
  EXPECT_EQ(*it++, 1);
  EXPECT_EQ(*it++, 2);
  EXPECT_EQ(*it++, 3);
}
