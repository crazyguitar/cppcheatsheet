#include <gtest/gtest.h>

#ifdef __cplusplus
extern "C" {
#endif

int fib(int n) {
  int a = 0, b = 1;
  for (int i = 0; i < n; ++i) {
    int tmp = b;
    b = a + b;
    a = tmp;
  }
  return a;
}

#ifdef __cplusplus
}
#endif

TEST(CLinkage, Fib) { EXPECT_EQ(fib(10), 55); }
