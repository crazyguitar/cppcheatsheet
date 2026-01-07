#include <gtest/gtest.h>

#include <string>
#include <string_view>

void take_view(std::string_view sv, std::string& out) { out = sv; }

TEST(StringView, FromString) {
  const std::string s = "foo";
  std::string out;
  take_view(s, out);
  EXPECT_EQ(out, "foo");
}

TEST(StringView, FromCString) {
  std::string out;
  take_view("bar", out);
  EXPECT_EQ(out, "bar");
}

TEST(StringView, ExplicitConversion) {
  std::string_view sv = "foo";
  std::string s(sv);
  EXPECT_EQ(s, "foo");
}

TEST(StringView, NonNullTerminated) {
  char array[3] = {'B', 'a', 'r'};
  std::string_view sv(array, sizeof(array));
  EXPECT_EQ(sv.size(), 3);
  EXPECT_EQ(sv[0], 'B');
  EXPECT_EQ(sv[1], 'a');
  EXPECT_EQ(sv[2], 'r');
}
