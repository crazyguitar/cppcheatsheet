#include <gtest/gtest.h>

struct Foo {
  static int constructs;
  Foo() { ++constructs; }
  static void reset() { constructs = 0; }
};

int Foo::constructs = 0;

void TakeByValue(Foo) {}

TEST(CopyElision, ArgumentElision) {
  Foo::reset();
  TakeByValue(Foo());
  EXPECT_EQ(Foo::constructs, 1);
}
