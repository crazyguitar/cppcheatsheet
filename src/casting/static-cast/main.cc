#include <gtest/gtest.h>

struct Base {
  virtual ~Base() = default;
};

struct Derived : public Base {
  int value = 42;
};

TEST(StaticCast, NumericConversion) {
  double d = 3.14;
  int i = static_cast<int>(d);
  EXPECT_EQ(i, 3);
}

TEST(StaticCast, Upcast) {
  Derived d;
  Base* b = static_cast<Base*>(&d);
  EXPECT_NE(b, nullptr);
}

TEST(StaticCast, Downcast) {
  Derived d;
  Base* b = &d;
  auto d2 = static_cast<Derived*>(b);
  EXPECT_EQ(d2->value, 42);
}
