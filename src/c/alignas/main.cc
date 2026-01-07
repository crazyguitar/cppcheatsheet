#include <gtest/gtest.h>

#include <cstdint>

struct alignas(64) CacheAligned {
  char data[64];
};

TEST(Alignas, Alignment) {
  EXPECT_EQ(alignof(int), 4u);
  EXPECT_EQ(alignof(double), 8u);
  EXPECT_EQ(alignof(CacheAligned), 64u);

  alignas(32) int aligned_int = 42;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&aligned_int) % 32, 0u);
}
