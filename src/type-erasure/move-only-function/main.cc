#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <utility>

// std::move_only_function (C++23) is the move-only counterpart to
// std::function. It can store callables that are themselves move-only —
// most commonly lambdas that capture a unique_ptr by value.
//
// Skipped automatically when the standard library does not yet provide it.

#if defined(__cpp_lib_move_only_function) && __cpp_lib_move_only_function >= 202110L

TEST(MoveOnlyFunction, AcceptsMoveOnlyLambda) {
  std::move_only_function<int()> f = [p = std::make_unique<int>(42)] { return *p; };
  EXPECT_EQ(f(), 42);
}

TEST(MoveOnlyFunction, IsMovableButNotCopyable) {
  static_assert(!std::is_copy_constructible_v<std::move_only_function<int()>>);
  static_assert(std::is_move_constructible_v<std::move_only_function<int()>>);

  std::move_only_function<int()> f = [p = std::make_unique<int>(7)] { return *p; };
  std::move_only_function<int()> g = std::move(f);
  EXPECT_EQ(g(), 7);
}

#else

TEST(MoveOnlyFunction, NotAvailableInThisStdlib) {
  GTEST_SKIP() << "std::move_only_function requires C++23 standard library support";
}

#endif
