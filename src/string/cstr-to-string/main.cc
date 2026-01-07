#include <gtest/gtest.h>

#include <cstring>
#include <string>

TEST(CstrToString, ImplicitConversion) {
  char cstr[] = "hello cstr";
  std::string s = cstr;
  EXPECT_EQ(s, "hello cstr");
}

TEST(CstrToString, ExplicitConversion) {
  const char* cstr = "hello";
  std::string s(cstr);
  EXPECT_EQ(s, "hello");
}
