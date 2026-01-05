#include <gtest/gtest.h>

struct Foo {
  static int copies;
  static int moves;

  Foo() = default;
  Foo(const Foo&) { ++copies; }
  Foo(Foo&&) noexcept { ++moves; }

  static void reset() { copies = moves = 0; }
};

int Foo::copies = 0;
int Foo::moves = 0;

const Foo global_foo;

Foo ReturnGlobal() { return global_foo; }

TEST(NoRVO, GlobalRequiresCopy) {
  Foo::reset();
  Foo f = ReturnGlobal();
  EXPECT_EQ(Foo::copies, 1);
}
