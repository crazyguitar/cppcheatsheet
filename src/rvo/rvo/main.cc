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

Foo MakeRVO() { return Foo(); }

TEST(RVO, NocopiesOrMoves) {
  Foo::reset();
  Foo f = MakeRVO();
  EXPECT_EQ(Foo::constructs, 1);
  EXPECT_EQ(Foo::copies, 0);
  EXPECT_EQ(Foo::moves, 0);
}
