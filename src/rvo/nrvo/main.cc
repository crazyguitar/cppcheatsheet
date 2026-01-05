#include <gtest/gtest.h>

struct Foo {
  static int constructs;
  static int copies;
  static int moves;

  Foo() { ++constructs; }
  Foo(const Foo&) { ++copies; }
  Foo(Foo&&) noexcept { ++moves; }

  static void reset() { constructs = copies = moves = 0; }
};

int Foo::constructs = 0;
int Foo::copies = 0;
int Foo::moves = 0;

Foo MakeNRVO() {
  Foo foo;
  return foo;
}

TEST(NRVO, LikelyElided) {
  Foo::reset();
  Foo f = MakeNRVO();
  EXPECT_EQ(Foo::constructs, 1);
  // NRVO may or may not apply, but no copies
  EXPECT_EQ(Foo::copies, 0);
}
