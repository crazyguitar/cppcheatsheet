#include <gtest/gtest.h>

#define max(a, b)         \
  ({                      \
    decltype(a) _a = (a); \
    decltype(b) _b = (b); \
    _a > _b ? _a : _b;    \
  })

TEST(StmtExpr, MaxMacro) {
  EXPECT_EQ(max(10, 20), 20);
  EXPECT_EQ(max(30, 5), 30);
  EXPECT_DOUBLE_EQ(max(3.14, 2.71), 3.14);
}
