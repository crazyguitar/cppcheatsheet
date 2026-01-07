#include <gtest/gtest.h>

[[nodiscard]] int must_use() { return 42; }

void callback([[maybe_unused]] int unused_param) {}

TEST(Attributes, Nodiscard) {
  int result = must_use();
  EXPECT_EQ(result, 42);
  callback(0);
}
