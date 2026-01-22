#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Pitfall 1: std::move doesn't actually move
TEST(MovePitfalls, StdMoveDoesNotMove) {
  std::string s1 = "hello";

  std::move(s1);           // Does nothing! Just a cast, result discarded
  EXPECT_EQ(s1, "hello");  // s1 unchanged

  // Must use the result of std::move
  std::string s2 = std::move(s1);  // Now s1 is moved
  EXPECT_EQ(s2, "hello");
}

// Pitfall 2: Moving from const copies instead
TEST(MovePitfalls, MovingFromConstCopies) {
  const std::string s1 = "hello";
  std::string s2 = std::move(s1);  // Calls copy constructor!

  // Both strings have the same value - copy was performed
  EXPECT_EQ(s1, "hello");
  EXPECT_EQ(s2, "hello");
}

// Pitfall 3: Using moved-from objects
TEST(MovePitfalls, UsingMovedFromObject) {
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = std::move(v1);

  // Don't rely on v1's state - it's unspecified
  // v1.size() could be 0 or something else

  // Safe: assign new value first
  v1 = {4, 5, 6};
  EXPECT_EQ(v1.size(), 3u);
}

// Pitfall 4: Forgetting noexcept
struct NoNoexceptWidget {
  NoNoexceptWidget() = default;
  NoNoexceptWidget(NoNoexceptWidget&&) {}  // Missing noexcept!
  NoNoexceptWidget(const NoNoexceptWidget&) = default;
};

struct NoexceptWidget {
  NoexceptWidget() = default;
  NoexceptWidget(NoexceptWidget&&) noexcept {}
  NoexceptWidget(const NoexceptWidget&) = default;
};

TEST(MovePitfalls, NoexceptMattersForVector) {
  // Vector checks is_nothrow_move_constructible for reallocation strategy
  static_assert(!std::is_nothrow_move_constructible_v<NoNoexceptWidget>);
  static_assert(std::is_nothrow_move_constructible_v<NoexceptWidget>);
}

// Pitfall 5: Not resetting moved-from state (demonstrated conceptually)
class BadResource {
 public:
  explicit BadResource(int* p) : ptr_(p) {}
  ~BadResource() { delete ptr_; }

  // Bug: doesn't reset other.ptr_
  BadResource(BadResource&& other) noexcept : ptr_(other.ptr_) {
    // other.ptr_ still points to same memory!
  }

 private:
  int* ptr_;
};

class GoodResource {
 public:
  explicit GoodResource(int* p) : ptr_(p) {}
  ~GoodResource() { delete ptr_; }

  // Correct: reset other.ptr_ to nullptr
  GoodResource(GoodResource&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}

  int* get() const { return ptr_; }

 private:
  int* ptr_;
};

TEST(MovePitfalls, ProperStateReset) {
  GoodResource r1(new int(42));
  GoodResource r2(std::move(r1));

  EXPECT_EQ(r1.get(), nullptr);  // Properly reset
  EXPECT_NE(r2.get(), nullptr);
}
