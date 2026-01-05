#include <gtest/gtest.h>

#include <string>

TEST(ConcatPerf, AppendChar) {
  std::string s;
  for (int i = 0; i < 1000; ++i) {
    s += 'a';
  }
  EXPECT_EQ(s.size(), 1000);
}

TEST(ConcatPerf, PrependChar) {
  std::string s;
  for (int i = 0; i < 100; ++i) {
    s = std::string(1, 'a') + s;
  }
  EXPECT_EQ(s.size(), 100);
}
