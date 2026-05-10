#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

// Demonstrate what std::move actually does - it's just a cast
TEST(ValueCategories, StdMoveIsJustACast) {
  std::string s1 = "hello";

  // std::move alone does nothing - it's just a cast to rvalue reference
  std::move(s1);
  EXPECT_EQ(s1, "hello");  // s1 unchanged

  // The actual move happens when move constructor/assignment is invoked
  std::string s2 = std::move(s1);
  EXPECT_TRUE(s1.empty() || s1 == "hello");  // Valid but unspecified
  EXPECT_EQ(s2, "hello");
}

// Moving from const copies instead of moving
TEST(ValueCategories, MovingFromConstCopies) {
  const std::string s1 = "hello";
  std::string s2 = std::move(s1);  // Calls copy constructor!

  // s1 is unchanged because copy was performed
  EXPECT_EQ(s1, "hello");
  EXPECT_EQ(s2, "hello");
}

// Demonstrate value category detection
template <typename T>
constexpr bool is_lvalue(T&&) {
  return std::is_lvalue_reference_v<T&&>;
}

template <typename T>
constexpr bool is_rvalue(T&&) {
  return !std::is_lvalue_reference_v<T&&>;
}

TEST(ValueCategories, DetectValueCategory) {
  int x = 42;

  EXPECT_TRUE(is_lvalue(x));             // x is lvalue
  EXPECT_TRUE(is_rvalue(42));            // 42 is prvalue
  EXPECT_TRUE(is_rvalue(std::move(x)));  // std::move(x) is xvalue
}

// Overload resolution based on value category
struct Tracker {
  static int copy_count;
  static int move_count;

  static void reset() { copy_count = move_count = 0; }

  Tracker() = default;
  Tracker(const Tracker&) { ++copy_count; }
  Tracker(Tracker&&) noexcept { ++move_count; }
};

int Tracker::copy_count = 0;
int Tracker::move_count = 0;

TEST(ValueCategories, OverloadResolution) {
  Tracker::reset();

  Tracker t1;
  Tracker t2 = t1;             // Copy: t1 is lvalue
  Tracker t3 = std::move(t1);  // Move: std::move(t1) is xvalue
  Tracker t4 = Tracker();      // Move (or elided): Tracker() is prvalue

  EXPECT_EQ(Tracker::copy_count, 1);
  EXPECT_GE(Tracker::move_count, 1);  // At least 1, possibly 2 without elision
}
