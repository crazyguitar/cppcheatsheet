#include <gtest/gtest.h>

struct Base {
  virtual int f() const { return 1; }
  virtual int g() const { return 2; }
};

struct Derived : Base {
  int f() const override { return 10; }
  int g() const override final { return 20; }
};

struct Sealed final : Derived {};

TEST(OverrideFinal, OverrideKeepsContractWithBase) {
  Derived d;
  const Base& b = d;
  EXPECT_EQ(b.f(), 10);
}

TEST(OverrideFinal, FinalAllowsDevirtualization) {
  Sealed s;
  EXPECT_EQ(s.f(), 10);
  EXPECT_EQ(s.g(), 20);
}

// Negative compile-time examples (uncomment to verify they fail to compile):
//
// struct WrongConst : Base {
//   int f() override { return 0; }   // ERROR: missing const
// };
//
// struct PastFinal : Derived {
//   int g() const override { return 0; }  // ERROR: g is final
// };
//
// struct ExtendsSealed : Sealed {};       // ERROR: Sealed is final
