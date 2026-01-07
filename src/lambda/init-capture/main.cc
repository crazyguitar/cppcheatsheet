#include <gtest/gtest.h>

#include <memory>
#include <utility>

TEST(InitCapture, MoveCapture) {
  auto ptr = std::make_unique<int>(42);
  auto f = [p = std::move(ptr)]() { return *p; };
  EXPECT_EQ(f(), 42);
  EXPECT_EQ(ptr, nullptr);
}

TEST(InitCapture, Expression) {
  int x = 10;
  auto f = [y = x * 2]() { return y; };
  EXPECT_EQ(f(), 20);
}
