#include <gtest/gtest.h>

TEST(NegativeArrayIndex, PointerArithmetic) {
  int arr[] = {1, 2, 3};
  int* ptr = &arr[1];

  EXPECT_EQ(ptr[-1], 1);
  EXPECT_EQ(ptr[0], 2);
  EXPECT_EQ(ptr[1], 3);
}
