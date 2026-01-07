#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

TEST(ISO8601, Format) {
  std::time_t t = 0;  // Unix epoch
  std::ostringstream oss;
  oss << std::put_time(std::gmtime(&t), "%FT%TZ");
  EXPECT_EQ(oss.str(), "1970-01-01T00:00:00Z");
}
