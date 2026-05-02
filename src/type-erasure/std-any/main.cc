#include <gtest/gtest.h>

#include <any>
#include <string>
#include <typeinfo>

TEST(StdAny, HoldsAnyCopyableValue) {
  std::any a = 42;
  EXPECT_EQ(a.type(), typeid(int));
  EXPECT_EQ(std::any_cast<int>(a), 42);

  a = std::string("hello");
  EXPECT_EQ(a.type(), typeid(std::string));
  EXPECT_EQ(std::any_cast<std::string>(a), "hello");
}

TEST(StdAny, BadCastThrows) {
  std::any a = 42;
  EXPECT_THROW(std::any_cast<std::string>(a), std::bad_any_cast);
}

TEST(StdAny, PointerCastReturnsNullOnMismatch) {
  std::any a = 42;
  EXPECT_NE(std::any_cast<int>(&a), nullptr);
  EXPECT_EQ(std::any_cast<std::string>(&a), nullptr);
}

TEST(StdAny, EmptyAnyReportsHasValueFalse) {
  std::any a;
  EXPECT_FALSE(a.has_value());
  a = 1;
  EXPECT_TRUE(a.has_value());
  a.reset();
  EXPECT_FALSE(a.has_value());
}

// std::any erases the *type* but exposes no behavior. To call methods you
// must any_cast back to the original type. For behavior-erased dispatch use
// a custom wrapper (see ../any-drawable) or std::function.
