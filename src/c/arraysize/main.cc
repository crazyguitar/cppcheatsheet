#include <gtest/gtest.h>

#include <cstring>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

TEST(ArraySize, CalculatesCorrectly) {
  int nums[] = {1, 2, 3, 4, 5};
  const char* strs[] = {"a", "b", "c"};

  EXPECT_EQ(ARRAY_SIZE(nums), 5u);
  EXPECT_EQ(ARRAY_SIZE(strs), 3u);
}
