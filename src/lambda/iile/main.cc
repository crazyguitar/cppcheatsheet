#include <gtest/gtest.h>

TEST(IILE, ConstInit) {
  const int value = []() {
    int sum = 0;
    for (int i = 1; i <= 10; ++i) sum += i;
    return sum;
  }();
  EXPECT_EQ(value, 55);
}
