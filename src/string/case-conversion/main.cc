#include <gtest/gtest.h>

#include <algorithm>
#include <string>

TEST(CaseConversion, ToUpper) {
  std::string s = "Hello World";
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  EXPECT_EQ(s, "HELLO WORLD");
}

TEST(CaseConversion, ToLower) {
  std::string s = "Hello World";
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  EXPECT_EQ(s, "hello world");
}
