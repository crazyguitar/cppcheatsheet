#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& str, char delim) {
  std::string s = str;
  std::vector<std::string> out;
  size_t pos = 0;

  while ((pos = s.find(delim)) != std::string::npos) {
    out.emplace_back(s.substr(0, pos));
    s.erase(0, pos + 1);
  }
  out.emplace_back(s);
  return out;
}

std::vector<std::string> split_getline(const std::string& str, char delim) {
  std::vector<std::string> out;
  std::string token;
  std::istringstream stream(str);

  while (std::getline(stream, token, delim)) {
    out.emplace_back(token);
  }
  return out;
}

TEST(Split, FindAndSubstr) {
  auto v = split("abc,def,ghi", ',');
  ASSERT_EQ(v.size(), 3);
  EXPECT_EQ(v[0], "abc");
  EXPECT_EQ(v[1], "def");
  EXPECT_EQ(v[2], "ghi");
}

TEST(Split, Getline) {
  auto v = split_getline("abc,def,ghi", ',');
  ASSERT_EQ(v.size(), 3);
  EXPECT_EQ(v[0], "abc");
  EXPECT_EQ(v[1], "def");
  EXPECT_EQ(v[2], "ghi");
}

TEST(Split, SingleElement) {
  auto v = split("abc", ',');
  ASSERT_EQ(v.size(), 1);
  EXPECT_EQ(v[0], "abc");
}
