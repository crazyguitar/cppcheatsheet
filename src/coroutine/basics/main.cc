#include <gtest/gtest.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>

TEST(BasicsTest, HelloCoroutine) {
  bool called = false;
  auto hello = [&]() -> Coro<> {
    called = true;
    co_return;
  };
  EXPECT_FALSE(called);
  auto fut = Future(hello());
  IO::Get().Run();
  EXPECT_TRUE(called);
}
