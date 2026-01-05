#include <gtest/gtest.h>

#include <map>
#include <string>

TEST(Unevaluated, MapComparator) {
  std::map<int, std::string, decltype([](int a, int b) { return a > b; })> m;
  m[3] = "three";
  m[1] = "one";
  EXPECT_EQ(m.begin()->first, 3);  // Descending
}
