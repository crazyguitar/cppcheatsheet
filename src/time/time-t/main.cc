#include <gtest/gtest.h>

#include <chrono>
#include <ctime>

TEST(TimeT, Conversion) {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  EXPECT_GT(t, 0);
}
