#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>
#include <io/sleep.h>

#include <chrono>

TEST(SleepTest, DelayedExecution) {
  int step = 0;
  auto coro = [&]() -> Coro<> {
    step = 1;
    co_await Sleep(std::chrono::milliseconds(10));
    step = 2;
  };
  auto fut = Future(coro());
  EXPECT_EQ(step, 0);

  auto start = std::chrono::steady_clock::now();
  IO::Get().Run();
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_EQ(step, 2);
  // Allow 1ms tolerance for timer precision
  EXPECT_GE(elapsed, std::chrono::milliseconds(9));
}
