#include <gtest/gtest.h>

#include <string>

TEST(CharToString, FillConstructor) {
  std::string s(1, 'a');
  EXPECT_EQ(s, "a");
}

TEST(CharToString, AppendOperator) {
  std::string s;
  s += 'a';
  EXPECT_EQ(s, "a");
}

TEST(CharToString, AssignmentOperator) {
  std::string s;
  s = 'a';
  EXPECT_EQ(s, "a");
}
