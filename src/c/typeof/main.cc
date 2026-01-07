#include <gtest/gtest.h>

#include <type_traits>

TEST(Typeof, TypeDeduction) {
  auto x = 42;
  auto pi = 3.14159;
  auto msg = "hello";

  EXPECT_EQ(x, 42);
  EXPECT_DOUBLE_EQ(pi, 3.14159);
  EXPECT_STREQ(msg, "hello");

  int arr[] = {1, 2, 3, 4, 5};
  std::remove_reference_t<decltype(arr[0])> sum = 0;
  for (int i = 0; i < 5; i++) {
    sum += arr[i];
  }
  EXPECT_EQ(sum, 15);
}
