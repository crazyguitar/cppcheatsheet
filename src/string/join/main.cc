#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

std::string join(const std::vector<std::string>& v, char delim) {
  std::ostringstream oss;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) oss << delim;
    oss << v[i];
  }
  return oss.str();
}

TEST(Join, MultipleElements) {
  std::vector<std::string> v = {"abc", "def", "ghi"};
  EXPECT_EQ(join(v, ','), "abc,def,ghi");
}

TEST(Join, SingleElement) {
  std::vector<std::string> v = {"abc"};
  EXPECT_EQ(join(v, ','), "abc");
}

TEST(Join, Empty) {
  std::vector<std::string> v;
  EXPECT_EQ(join(v, ','), "");
}
