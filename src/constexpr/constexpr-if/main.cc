#include <gtest/gtest.h>

#include <type_traits>

template <typename T>
auto get_value(T t) {
  if constexpr (std::is_pointer_v<T>) {
    return *t;
  } else {
    return t;
  }
}

TEST(ConstexprIf, NonPointer) {
  int x = 42;
  EXPECT_EQ(get_value(x), 42);
}

TEST(ConstexprIf, Pointer) {
  int x = 42;
  int* p = &x;
  EXPECT_EQ(get_value(p), 42);
}
