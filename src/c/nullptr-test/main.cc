#include <gtest/gtest.h>

#include <cstddef>

TEST(Nullptr, NullPointer) {
  int* p = nullptr;
  EXPECT_EQ(p, nullptr);
  EXPECT_TRUE(p == nullptr);
  EXPECT_FALSE(p != nullptr);
}
