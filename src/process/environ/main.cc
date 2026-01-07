#include <gtest/gtest.h>

#include <cstdlib>

TEST(Environ, SetGetUnset) {
  setenv("TEST_VAR", "hello", 1);
  EXPECT_STREQ(getenv("TEST_VAR"), "hello");
  unsetenv("TEST_VAR");
  EXPECT_EQ(getenv("TEST_VAR"), nullptr);
}
