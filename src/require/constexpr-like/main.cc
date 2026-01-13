#include <gtest/gtest.h>

#include <concepts>

// Using if constexpr
template <typename T>
auto get_value_constexpr(T t) {
  if constexpr (std::is_pointer_v<T>) {
    return *t;  // dereference pointer
  } else {
    return t;  // return value directly
  }
}

// Using requires (overload resolution)
template <typename T>
  requires std::is_pointer_v<T>
auto get_value_requires(T t) {
  return *t;  // selected when T is pointer
}

template <typename T>
  requires(!std::is_pointer_v<T>)
auto get_value_requires(T t) {
  return t;  // selected when T is not pointer
}

TEST(RequiresConstexprLike, IfConstexpr) {
  int x = 42;
  int* p = &x;
  EXPECT_EQ(get_value_constexpr(x), 42);  // OK: returns value
  EXPECT_EQ(get_value_constexpr(p), 42);  // OK: dereferences pointer
}

TEST(RequiresConstexprLike, RequiresOverload) {
  int x = 42;
  int* p = &x;
  EXPECT_EQ(get_value_requires(x), 42);  // OK: selects non-pointer overload
  EXPECT_EQ(get_value_requires(p), 42);  // OK: selects pointer overload
}
