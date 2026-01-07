#include <gtest/gtest.h>

#include <vector>

TEST(Vector, Initialization) {
  std::vector<int> v1;
  std::vector<int> v2(5);
  std::vector<int> v3(5, 10);
  std::vector<int> v4{1, 2, 3, 4, 5};

  EXPECT_TRUE(v1.empty());
  EXPECT_EQ(v2.size(), 5);
  EXPECT_EQ(v3[0], 10);
  EXPECT_EQ(v4.size(), 5);
}

TEST(Vector, ElementAccess) {
  std::vector<int> v{1, 2, 3, 4, 5};

  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v.at(0), 1);
  EXPECT_EQ(v.front(), 1);
  EXPECT_EQ(v.back(), 5);
  EXPECT_EQ(v.data()[0], 1);
}

TEST(Vector, Modifiers) {
  std::vector<int> v;

  v.push_back(1);
  EXPECT_EQ(v.back(), 1);

  v.emplace_back(2);
  EXPECT_EQ(v.back(), 2);

  v.pop_back();
  EXPECT_EQ(v.size(), 1);

  v.insert(v.begin(), 0);
  EXPECT_EQ(v.front(), 0);

  v.erase(v.begin());
  EXPECT_EQ(v.front(), 1);

  v.clear();
  EXPECT_TRUE(v.empty());
}

TEST(Vector, Capacity) {
  std::vector<int> v;

  v.reserve(100);
  EXPECT_GE(v.capacity(), 100);
  EXPECT_EQ(v.size(), 0);

  v.resize(50);
  EXPECT_EQ(v.size(), 50);
}
