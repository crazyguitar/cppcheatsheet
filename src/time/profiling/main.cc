#include <gtest/gtest.h>

#include <chrono>
#include <thread>

TEST(Profiling, SteadyClock) {
  auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  EXPECT_GE(elapsed.count(), 10);
}
