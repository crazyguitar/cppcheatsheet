#include <gtest/gtest.h>

#include <map>
#include <string>

TEST(Map, InsertAndAccess) {
  std::map<std::string, int> m{{"apple", 1}, {"banana", 2}};

  m["cherry"] = 3;
  EXPECT_EQ(m["cherry"], 3);

  m.insert({"date", 4});
  EXPECT_EQ(m.size(), 4);
}

TEST(Map, Erase) {
  std::map<std::string, int> m{{"apple", 1}, {"banana", 2}};

  m.erase("banana");
  EXPECT_EQ(m.count("banana"), 0);
}

TEST(Map, Find) {
  std::map<std::string, int> m{{"apple", 1}};

  EXPECT_NE(m.find("apple"), m.end());
  EXPECT_EQ(m.find("orange"), m.end());
}

TEST(Map, Ordered) {
  std::map<std::string, int> m{{"cherry", 3}, {"apple", 1}, {"banana", 2}};

  auto it = m.begin();
  EXPECT_EQ(it++->first, "apple");
  EXPECT_EQ(it++->first, "banana");
  EXPECT_EQ(it++->first, "cherry");
}
