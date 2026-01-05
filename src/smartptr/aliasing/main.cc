#include <gtest/gtest.h>

#include <memory>

struct Inner {
  int value = 42;
};

struct Outer {
  Inner inner;
};

TEST(Aliasing, SharesOwnershipPointsToDifferent) {
  auto outer = std::make_shared<Outer>();
  std::shared_ptr<Inner> inner(outer, &outer->inner);

  EXPECT_EQ(inner->value, 42);
  EXPECT_EQ(outer.use_count(), 2);
}
