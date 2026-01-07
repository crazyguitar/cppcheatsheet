#include <gtest/gtest.h>

#include <chrono>
#include <string>

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  auto elapsed() const {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

TEST(Timer, MeasuresElapsed) {
  Timer t;
  int sum = 0;
  for (int i = 0; i < 1000; ++i) sum += i;
  auto elapsed = t.elapsed();
  EXPECT_GE(elapsed.count(), 0);
  EXPECT_EQ(sum, 499500);  // Prevent optimization
}
