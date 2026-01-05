#include <gtest/gtest.h>

#include <chrono>
#include <ctime>

TEST(FromTimestamp, MillisecondsToTimePoint) {
  using namespace std::chrono_literals;
  auto timestamp_ms = 1602207217323ms;
  std::chrono::system_clock::time_point tp(timestamp_ms);
  std::time_t t = std::chrono::system_clock::to_time_t(tp);
  EXPECT_GT(t, 0);
}
