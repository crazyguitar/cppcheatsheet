#include <gtest/gtest.h>

#include <chrono>

TEST(HhMmSs, Breakdown) {
  using namespace std::chrono;
  auto d = 3h + 25min + 45s;
  hh_mm_ss hms{d};
  EXPECT_EQ(hms.hours().count(), 3);
  EXPECT_EQ(hms.minutes().count(), 25);
  EXPECT_EQ(hms.seconds().count(), 45);
}
