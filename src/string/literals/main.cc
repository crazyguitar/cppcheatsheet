#include <gtest/gtest.h>

#include <string>
#include <string_view>

using namespace std::literals;

TEST(Literals, CString) {
  auto s = "c string";
  EXPECT_STREQ(s, "c string");
}

TEST(Literals, StdString) {
  auto s = "std::string"s;
  EXPECT_EQ(s, "std::string");
}

TEST(Literals, StringView) {
  auto s = "std::string_view"sv;
  EXPECT_EQ(s, "std::string_view");
}
