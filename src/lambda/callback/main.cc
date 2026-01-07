#include <gtest/gtest.h>

#include <functional>

template <typename F>
int process(int n, F callback) {
  int sum = 0;
  for (int i = 0; i < n; ++i) sum += callback(i);
  return sum;
}

TEST(Callback, Template) {
  int result = process(5, [](int x) { return x; });
  EXPECT_EQ(result, 10);  // 0+1+2+3+4
}

TEST(Callback, StdFunction) {
  std::function<int(int)> cb = [](int x) { return x * 2; };
  int result = process(5, cb);
  EXPECT_EQ(result, 20);  // 0+2+4+6+8
}
