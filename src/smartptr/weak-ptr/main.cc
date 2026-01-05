#include <gtest/gtest.h>

#include <memory>

TEST(WeakPtr, DoesNotExtendLifetime) {
  std::weak_ptr<int> weak;
  {
    auto shared = std::make_shared<int>(42);
    weak = shared;
    EXPECT_FALSE(weak.expired());
  }
  EXPECT_TRUE(weak.expired());
}

TEST(WeakPtr, LockReturnsShared) {
  auto shared = std::make_shared<int>(42);
  std::weak_ptr<int> weak = shared;

  if (auto locked = weak.lock()) {
    EXPECT_EQ(*locked, 42);
    EXPECT_EQ(shared.use_count(), 2);
  }
}
