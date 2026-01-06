#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>

struct ReturnValue {
  int value;
  bool await_ready() const noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) const noexcept {}
  int await_resume() const noexcept { return value; }
};

struct ResumeImmediately {
  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) const noexcept { h.resume(); }
  void await_resume() const noexcept {}
};

TEST(AwaiterTest, CustomAwaiters) {
  int value = 0;
  auto coro = [&]() -> Coro<> {
    value = co_await ReturnValue{42};
    co_await ResumeImmediately{};
    value += 10;
  };
  auto fut = Future(coro());
  IO::Get().Run();
  EXPECT_EQ(value, 52);
}
