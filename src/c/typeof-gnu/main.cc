#include <gtest/gtest.h>

#define SWAP(a, b)   \
  do {               \
    auto _tmp = (a); \
    (a) = (b);       \
    (b) = _tmp;      \
  } while (0)

TEST(TypeofGnu, SwapMacro) {
  int i = 5, j = 10;
  SWAP(i, j);
  EXPECT_EQ(i, 10);
  EXPECT_EQ(j, 5);

  double x = 3.14, y = 2.71;
  SWAP(x, y);
  EXPECT_DOUBLE_EQ(x, 2.71);
  EXPECT_DOUBLE_EQ(y, 3.14);
}
