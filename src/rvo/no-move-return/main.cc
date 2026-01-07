#include <gtest/gtest.h>

#include <utility>

struct Foo {
  static int moves;

  Foo() = default;
  Foo(const Foo&) = default;
  Foo(Foo&&) noexcept { ++moves; }

  static void reset() { moves = 0; }
};

int Foo::moves = 0;

Foo BadMove() {
  Foo foo;
  return std::move(foo);  // Prevents NRVO
}

TEST(NoMoveReturn, StdMovePreventsNRVO) {
  Foo::reset();
  Foo f = BadMove();
  EXPECT_EQ(Foo::moves, 1);  // Move happened, NRVO prevented
}
