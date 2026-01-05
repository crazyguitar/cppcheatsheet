#include <gtest/gtest.h>

#include <utility>

struct Data {
  Data(int a, int b) : x(a), y(b), result(0) {}
  int x, y, result;
};

template <typename T, typename Func>
void wrapper(T&& arg, Func fn) {
  fn(std::forward<T>(arg));
}

TEST(PerfectForwarding, ForwardLvalue) {
  Data data{1, 2};
  wrapper(data, [](Data& d) { d.result = d.x + d.y; });
  EXPECT_EQ(data.result, 3);
}

TEST(PerfectForwarding, ForwardRvalue) {
  Data temp{5, 6};
  wrapper(std::move(temp), [](Data&& d) { d.result = d.x * d.y; });
  EXPECT_EQ(temp.result, 30);
}
