#include <gtest/gtest.h>

#include <chrono>

TEST(Timestamp, MillisecondsSinceEpoch) {
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
  EXPECT_GT(ms.count(), 0);
}
