#include <gtest/gtest.h>

#include <string>

TEST(Substr, FromStart) {
  std::string s = "Hello World";
  EXPECT_EQ(s.substr(0, 5), "Hello");
}

TEST(Substr, ToEnd) {
  std::string s = "Hello World";
  EXPECT_EQ(s.substr(6), "World");
}

TEST(Substr, Middle) {
  std::string s = "Hello World";
  EXPECT_EQ(s.substr(6, 3), "Wor");
}
