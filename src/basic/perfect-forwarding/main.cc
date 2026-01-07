#include <gtest/gtest.h>

#include <chrono>
#include <memory>
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

// Decorator pattern
template <typename Func, typename... Args>
auto timed(Func&& f, Args&&... args) {
  auto start = std::chrono::system_clock::now();
  auto ret = f(std::forward<Args>(args)...);
  std::chrono::duration<double> d = std::chrono::system_clock::now() - start;
  return ret;
}

TEST(PerfectForwarding, Decorator) {
  auto result = timed([](int x) { return x * 2; }, 21);
  EXPECT_EQ(result, 42);
}

// Factory pattern
template <typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

struct Widget {
  int x;
  double y;
};

TEST(PerfectForwarding, Factory) {
  auto w = make<Widget>(42, 3.14);
  EXPECT_EQ(w->x, 42);
}
