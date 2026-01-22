#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// noexcept move: vector will use move during reallocation
struct FastWidget {
  FastWidget() = default;
  FastWidget(FastWidget&&) noexcept = default;
  FastWidget& operator=(FastWidget&&) noexcept = default;
  FastWidget(const FastWidget&) = default;
  FastWidget& operator=(const FastWidget&) = default;

  static int move_count;
  static int copy_count;
};

int FastWidget::move_count = 0;
int FastWidget::copy_count = 0;

// Non-noexcept move: vector will copy for exception safety
struct SlowWidget {
  SlowWidget() = default;
  SlowWidget(SlowWidget&&) { ++move_count; }  // Missing noexcept!
  SlowWidget& operator=(SlowWidget&&) {
    ++move_count;
    return *this;
  }
  SlowWidget(const SlowWidget&) { ++copy_count; }
  SlowWidget& operator=(const SlowWidget&) {
    ++copy_count;
    return *this;
  }

  static int move_count;
  static int copy_count;
};

int SlowWidget::move_count = 0;
int SlowWidget::copy_count = 0;

TEST(ConditionalNoexcept, VectorUsesNoexceptMove) {
  // std::string has noexcept move
  static_assert(std::is_nothrow_move_constructible_v<std::string>);
  static_assert(std::is_nothrow_move_assignable_v<std::string>);
}

TEST(ConditionalNoexcept, VectorFallsBackToCopy) {
  SlowWidget::move_count = 0;
  SlowWidget::copy_count = 0;

  std::vector<SlowWidget> v;
  v.reserve(1);
  v.emplace_back();
  v.emplace_back();  // Triggers reallocation

  // Vector copies instead of moves because move is not noexcept
  EXPECT_GT(SlowWidget::copy_count, 0);
}

// Conditional noexcept based on member type
template <typename T>
class Wrapper {
 public:
  Wrapper() = default;
  explicit Wrapper(T val) : data_(std::move(val)) {}

  // noexcept if T's move is noexcept
  Wrapper(Wrapper&& other) noexcept(std::is_nothrow_move_constructible_v<T>) : data_(std::move(other.data_)) {}

  Wrapper& operator=(Wrapper&& other) noexcept(std::is_nothrow_move_assignable_v<T>) {
    data_ = std::move(other.data_);
    return *this;
  }

  const T& get() const { return data_; }

 private:
  T data_;
};

TEST(ConditionalNoexcept, PropagatesNoexcept) {
  // Wrapper<string> should be nothrow movable because string is
  static_assert(std::is_nothrow_move_constructible_v<Wrapper<std::string>>);

  Wrapper<std::string> w1("hello");
  Wrapper<std::string> w2 = std::move(w1);
  EXPECT_EQ(w2.get(), "hello");
}
