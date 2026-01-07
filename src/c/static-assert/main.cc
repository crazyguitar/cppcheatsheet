#include <gtest/gtest.h>

#include <cstdint>

static_assert(sizeof(int) >= 4, "int must be at least 32 bits");
static_assert(sizeof(void*) == 8 || sizeof(void*) == 4, "pointer size check");

struct packet {
  uint32_t header;
  uint32_t payload;
};
static_assert(sizeof(packet) == 8, "packet must be 8 bytes");

TEST(StaticAssert, CompileTimeCheck) {
  // If we get here, all static assertions passed
  SUCCEED();
}
