#include <gtest/gtest.h>

#include <memory>

TEST(UniquePtr, ExclusiveOwnership) {
  auto ptr = std::make_unique<int>(42);
  EXPECT_EQ(*ptr, 42);
}

TEST(UniquePtr, MoveTransfersOwnership) {
  auto ptr1 = std::make_unique<int>(42);
  auto ptr2 = std::move(ptr1);
  EXPECT_EQ(ptr1, nullptr);
  EXPECT_EQ(*ptr2, 42);
}

TEST(UniquePtr, ArraySupport) {
  auto arr = std::make_unique<int[]>(5);
  arr[0] = 10;
  EXPECT_EQ(arr[0], 10);
}
