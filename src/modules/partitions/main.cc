#include <gtest/gtest.h>

import graphics;

TEST(ModulesPartitions, Point) {
  Point p{3.0, 4.0};
  EXPECT_DOUBLE_EQ(p.x, 3.0);
  EXPECT_DOUBLE_EQ(p.y, 4.0);
}

TEST(ModulesPartitions, Color) {
  EXPECT_EQ(Red.r, 255);
  EXPECT_EQ(Red.g, 0);
  EXPECT_EQ(Green.g, 255);
  EXPECT_EQ(Blue.b, 255);
}

TEST(ModulesPartitions, Distance) {
  Point a{0, 0};
  Point b{3, 4};
  EXPECT_DOUBLE_EQ(distance(a, b), 25.0);  // 3^2 + 4^2
}
