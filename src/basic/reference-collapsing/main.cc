#include <gtest/gtest.h>

#include <type_traits>

template <typename T>
struct TypeCapture {
  using type = T;
};

template <typename T>
TypeCapture<T&> f_ref(T&) noexcept {
  return {};
}

template <typename T>
TypeCapture<T&&> f_uref(T&&) noexcept {
  return {};
}

TEST(ReferenceCollapsing, LvalueRef) {
  int x = 123;
  const int cx = x;
  static_assert(std::is_same_v<decltype(f_ref(x))::type, int&>);
  static_assert(std::is_same_v<decltype(f_ref(cx))::type, const int&>);
}

TEST(ReferenceCollapsing, UniversalRef) {
  int x = 123;
  static_assert(std::is_same_v<decltype(f_uref(x))::type, int&>);
  static_assert(std::is_same_v<decltype(f_uref(12))::type, int&&>);
}
