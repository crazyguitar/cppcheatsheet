#include <gtest/gtest.h>

#include <chrono>

TEST(Duration, Arithmetic) {
  using namespace std::chrono_literals;
  auto d = 1h + 30min;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(d);
  EXPECT_EQ(seconds.count(), 5400);
}

TEST(Duration, Literals) {
  using namespace std::chrono_literals;
  auto ms = 1000ms;
  auto s = std::chrono::duration_cast<std::chrono::seconds>(ms);
  EXPECT_EQ(s.count(), 1);
}
