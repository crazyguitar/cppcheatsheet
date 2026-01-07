#include <gtest/gtest.h>
#include <cstdint>

struct __attribute__((packed)) PackedStruct {
  uint8_t a;
  uint32_t b;
};

struct NormalStruct {
  uint8_t a;
  uint32_t b;
};

TEST(Attribute, PackedStruct) {
  EXPECT_EQ(sizeof(PackedStruct), 5u);
  EXPECT_GT(sizeof(NormalStruct), 5u);  // has padding
}
