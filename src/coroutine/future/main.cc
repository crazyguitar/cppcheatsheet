#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>

TEST(FutureTest, Result) {
  Coro<int> compute = []() -> Coro<int> { co_return 42; }();
  auto fut = Future(std::move(compute));
  IO::Get().Run();
  EXPECT_EQ(fut.result(), 42);
}

TEST(FutureTest, Done) {
  auto coro = []() -> Coro<> { co_return; }();
  auto fut = Future(std::move(coro));
  EXPECT_FALSE(fut.done());
  IO::Get().Run();
  EXPECT_TRUE(fut.done());
}
