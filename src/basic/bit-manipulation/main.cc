#include <gtest/gtest.h>

#include <bitset>

TEST(BitManipulation, Bitset) {
  std::bitset<4> bits{0b1000};
  EXPECT_EQ(bits.count(), 1);
  EXPECT_TRUE(bits == 8);
}
