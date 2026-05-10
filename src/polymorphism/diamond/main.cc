#include <gtest/gtest.h>

namespace non_virtual_diamond {

struct A {
  int x = 0;
};
struct B : A {};
struct C : A {};
struct D : B, C {};

}  // namespace non_virtual_diamond

namespace virtual_diamond {

struct A {
  int x = 0;
};
struct B : virtual A {};
struct C : virtual A {};
struct D : B, C {};

}  // namespace virtual_diamond

TEST(Diamond, NonVirtualDuplicatesBaseSubobject) {
  non_virtual_diamond::D d;
  d.B::x = 1;
  d.C::x = 2;
  // Two independent A subobjects exist.
  EXPECT_EQ(d.B::x, 1);
  EXPECT_EQ(d.C::x, 2);
}

TEST(Diamond, VirtualInheritanceCollapsesToSingleSubobject) {
  virtual_diamond::D d;
  d.x = 7;  // unambiguous
  EXPECT_EQ(d.B::x, 7);
  EXPECT_EQ(d.C::x, 7);
}
