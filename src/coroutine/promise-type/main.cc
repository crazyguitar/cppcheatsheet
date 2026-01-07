#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>

Coro<int> compute() { co_return 42; }

TEST(PromiseTypeTest, ReturnValue) {
  bool computed = false;
  auto coro = [&]() -> Coro<int> {
    computed = true;
    co_return 42;
  };
  auto fut = Future(coro());
  EXPECT_FALSE(computed);
  IO::Get().Run();
  EXPECT_TRUE(computed);
  EXPECT_EQ(fut.result(), 42);
}
