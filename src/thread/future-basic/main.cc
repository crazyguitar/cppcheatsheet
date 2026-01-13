#include <gtest/gtest.h>

#include <future>
#include <thread>

int Compute(int x) { return x * x; }

TEST(FutureBasic, GetResult) {
  std::future<int> fut = std::async(std::launch::async, Compute, 5);
  EXPECT_EQ(fut.get(), 25);  // blocks until result ready
}

TEST(FutureBasic, WaitFor) {
  std::future<int> fut = std::async(std::launch::async, [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 42;
  });
  auto status = fut.wait_for(std::chrono::milliseconds(100));
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_EQ(fut.get(), 42);
}
