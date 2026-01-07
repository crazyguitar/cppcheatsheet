#include <gtest/gtest.h>

struct Foo {
  static int constructs;
  Foo() { ++constructs; }
  Foo(const Foo&) { ++constructs; }
  Foo(Foo&&) noexcept { ++constructs; }
  static void reset() { constructs = 0; }
};

int Foo::constructs = 0;

Foo ConditionalRVO(bool flag) { return flag ? Foo() : Foo(); }

TEST(Conditional, RVOStillApplies) {
  Foo::reset();
  Foo f = ConditionalRVO(true);
  EXPECT_EQ(Foo::constructs, 1);  // RVO applies
}
