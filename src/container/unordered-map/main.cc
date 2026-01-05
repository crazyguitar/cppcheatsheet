#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

TEST(UnorderedMap, InsertAndAccess) {
  std::unordered_map<std::string, int> m{{"apple", 1}, {"banana", 2}};

  m["cherry"] = 3;
  EXPECT_EQ(m["cherry"], 3);

  m.insert({"date", 4});
  EXPECT_EQ(m.size(), 4);
}

TEST(UnorderedMap, Erase) {
  std::unordered_map<std::string, int> m{{"apple", 1}, {"banana", 2}};

  m.erase("banana");
  EXPECT_EQ(m.count("banana"), 0);
}

TEST(UnorderedMap, Find) {
  std::unordered_map<std::string, int> m{{"apple", 1}};

  EXPECT_NE(m.find("apple"), m.end());
  EXPECT_EQ(m.find("orange"), m.end());
}
