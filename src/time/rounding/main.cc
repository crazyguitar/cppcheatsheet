#include <gtest/gtest.h>

#include <chrono>

TEST(Rounding, Floor) {
  using namespace std::chrono;
  auto d = 2700ms;
  auto floored = floor<seconds>(d);
  EXPECT_EQ(floored.count(), 2);
}

TEST(Rounding, Ceil) {
  using namespace std::chrono;
  auto d = 2700ms;
  auto ceiled = ceil<seconds>(d);
  EXPECT_EQ(ceiled.count(), 3);
}

TEST(Rounding, Round) {
  using namespace std::chrono;
  auto d = 2700ms;
  auto rounded = round<seconds>(d);
  EXPECT_EQ(rounded.count(), 3);
}
