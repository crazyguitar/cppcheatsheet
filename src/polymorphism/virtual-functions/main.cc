#include <gtest/gtest.h>

#include <string>

struct Animal {
  virtual std::string speak() const { return "..."; }
  std::string name() const { return "Animal"; }
};

struct Dog : Animal {
  std::string speak() const override { return "Woof"; }
  std::string name() const { return "Dog"; }
};

TEST(VirtualFunctions, DynamicDispatchThroughBaseReference) {
  Dog d;
  Animal& a = d;
  EXPECT_EQ(a.speak(), "Woof");
}

TEST(VirtualFunctions, NonVirtualHidesInsteadOfOverrides) {
  Dog d;
  Animal& a = d;
  EXPECT_EQ(a.name(), "Animal");
  EXPECT_EQ(d.name(), "Dog");
}

TEST(VirtualFunctions, DirectCallUsesStaticType) {
  Dog d;
  EXPECT_EQ(d.speak(), "Woof");
}
