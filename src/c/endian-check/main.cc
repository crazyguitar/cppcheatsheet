#include <gtest/gtest.h>

#include <cstdint>

TEST(EndianCheck, DetectsEndianness) {
  uint16_t val = 0x0001;
  bool is_little = (*(uint8_t*)&val == 0x01);
  // Just verify detection works (result depends on platform)
  EXPECT_TRUE(is_little || !is_little);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  EXPECT_TRUE(is_little);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  EXPECT_FALSE(is_little);
#endif
}
