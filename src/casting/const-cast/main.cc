#include <gtest/gtest.h>

void modify(const int& x) { const_cast<int&>(x) = 0; }

TEST(ConstCast, RemoveConst) {
  int x = 123;
  modify(x);
  EXPECT_EQ(x, 0);
}

TEST(ConstCast, AddConst) {
  int x = 42;
  const int* cp = const_cast<const int*>(&x);
  EXPECT_EQ(*cp, 42);
}
