#include <gtest/gtest.h>

TEST(PredefinedMacros, Exist) {
  EXPECT_NE(__FILE__, nullptr);
  EXPECT_NE(__DATE__, nullptr);
  EXPECT_NE(__TIME__, nullptr);
  EXPECT_GT(__LINE__, 0);
  EXPECT_NE(__func__, nullptr);
}
