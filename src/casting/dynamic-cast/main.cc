#include <gtest/gtest.h>

struct Base {
  virtual ~Base() = default;
};

struct Derived : public Base {
  int value = 42;
};

TEST(DynamicCast, FailedDowncast) {
  Base b;
  Base* bp = &b;  // Use pointer to avoid compile-time type knowledge
  auto d = dynamic_cast<Derived*>(bp);
  EXPECT_EQ(d, nullptr);
}

TEST(DynamicCast, SuccessfulDowncast) {
  Derived d;
  Base* b = &d;
  auto d2 = dynamic_cast<Derived*>(b);
  EXPECT_NE(d2, nullptr);
  EXPECT_EQ(d2->value, 42);
}

TEST(DynamicCast, Upcast) {
  Derived d;
  auto b = dynamic_cast<Base*>(&d);
  EXPECT_NE(b, nullptr);
}
