#include <gtest/gtest.h>

TEST(Builtins, BitOperations) {
  unsigned int x = 0b10110000;  // 176 = 0xB0, bits: 10110000

  EXPECT_EQ(__builtin_popcount(x), 3);  // 3 bits set
  EXPECT_EQ(__builtin_ctz(x), 4);       // 4 trailing zeros
  EXPECT_EQ(__builtin_ffs(x), 5);       // first set bit at position 5 (1-indexed)
}
