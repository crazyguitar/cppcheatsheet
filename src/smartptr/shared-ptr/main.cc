#include <gtest/gtest.h>

#include <memory>

TEST(SharedPtr, SharedOwnership) {
  auto ptr1 = std::make_shared<int>(42);
  EXPECT_EQ(ptr1.use_count(), 1);

  auto ptr2 = ptr1;
  EXPECT_EQ(ptr1.use_count(), 2);
}

TEST(SharedPtr, DecrementOnDestroy) {
  auto ptr1 = std::make_shared<int>(42);
  {
    auto ptr2 = ptr1;
    EXPECT_EQ(ptr1.use_count(), 2);
  }
  EXPECT_EQ(ptr1.use_count(), 1);
}
