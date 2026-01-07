#include <gtest/gtest.h>

#include <climits>
#include <cmath>

TEST(CStyleCast, DoubleTruncation) {
  double x = M_PI;
  int xx = (int)x;
  EXPECT_EQ(xx, 3);
}

TEST(CStyleCast, PointerCast) {
  int x = 42;
  void* vp = (void*)&x;
  int* ip = (int*)vp;
  EXPECT_EQ(*ip, 42);
}
