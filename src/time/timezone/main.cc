#include <gtest/gtest.h>

#include <ctime>
#include <iomanip>
#include <sstream>

TEST(Timezone, UTC) {
  std::time_t t = 0;  // Unix epoch
  std::ostringstream oss;
  oss << std::put_time(std::gmtime(&t), "%F %T");
  EXPECT_EQ(oss.str(), "1970-01-01 00:00:00");
}
