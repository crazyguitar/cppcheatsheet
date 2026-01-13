#include <gtest/gtest.h>

#include <future>
#include <numeric>
#include <vector>

TEST(FutureAsync, LaunchAsync) {
  // std::launch::async forces a new thread
  auto fut = std::async(std::launch::async, [] { return 100; });
  EXPECT_EQ(fut.get(), 100);
}

TEST(FutureAsync, LaunchDeferred) {
  // std::launch::deferred runs lazily on get()
  auto fut = std::async(std::launch::deferred, [] { return 200; });
  EXPECT_EQ(fut.get(), 200);  // executes here
}

TEST(FutureAsync, ParallelSum) {
  std::vector<int> v(1000);
  std::iota(v.begin(), v.end(), 1);  // 1 to 1000

  auto mid = v.begin() + v.size() / 2;
  auto fut1 = std::async(std::launch::async, [&] { return std::accumulate(v.begin(), mid, 0); });
  auto fut2 = std::async(std::launch::async, [&] { return std::accumulate(mid, v.end(), 0); });

  int total = fut1.get() + fut2.get();
  EXPECT_EQ(total, 500500);  // sum of 1 to 1000
}
