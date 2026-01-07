#include <gtest/gtest.h>

#include <atomic>
#include <thread>

TEST(Atomics, ConcurrentIncrement) {
  std::atomic<int> counter{0};

  auto increment = [&counter]() {
    for (int i = 0; i < 100000; i++) {
      counter.fetch_add(1);
    }
  };

  std::thread t1(increment);
  std::thread t2(increment);
  t1.join();
  t2.join();

  EXPECT_EQ(counter.load(), 200000);
}
