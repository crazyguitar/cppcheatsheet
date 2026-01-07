#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>

template <typename C>
decltype(auto) Run(C&& coro) {
  auto fut = Future(std::move(coro));
  IO::Get().Run();
  return std::move(fut).result();
}

Coro<int> square(int x) { co_return x* x; }

Coro<int> sum_of_squares(int a, int b) {
  auto a2 = co_await square(a);
  auto b2 = co_await square(b);
  co_return a2 + b2;
}

TEST(CoroTest, ChainedCoroutines) { EXPECT_EQ(::Run(sum_of_squares(3, 4)), 25); }

TEST(CoroTest, NestedAwait) {
  auto outer = []() -> Coro<int> {
    auto inner = []() -> Coro<int> { co_return 10; };
    int x = co_await inner();
    int y = co_await inner();
    co_return x + y;
  };
  EXPECT_EQ(::Run(outer()), 20);
}
