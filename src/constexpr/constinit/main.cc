#include <gtest/gtest.h>

constexpr int compute() { return 42; }

constinit int global_val = compute();

TEST(Constinit, InitialValue) { EXPECT_EQ(global_val, 42); }

TEST(Constinit, Mutable) {
  global_val = 100;
  EXPECT_EQ(global_val, 100);
  global_val = 42;  // Reset for other tests
}
