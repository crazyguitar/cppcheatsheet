#include <gtest/gtest.h>

#include <type_traits>

constexpr int compute(int x) {
#if __cplusplus >= 202302L && __cpp_if_consteval >= 202106L
  if consteval {
    return x * x;  // Compile-time path
  } else {
    return x * x;  // Runtime path
  }
#else
  if (std::is_constant_evaluated()) {
    return x * x;  // Compile-time path
  } else {
    return x * x;  // Runtime path
  }
#endif
}

TEST(IfConsteval, CompileTime) {
  constexpr int result = compute(5);
  EXPECT_EQ(result, 25);
}

TEST(IfConsteval, Runtime) {
  int x = 5;
  int result = compute(x);
  EXPECT_EQ(result, 25);
}
