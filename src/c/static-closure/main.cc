#include <gtest/gtest.h>

namespace {
int call_count = 0;

void counter() {
  static int count = 0;
  count++;
  call_count = count;
}
}  // namespace

TEST(StaticClosure, PersistsAcrossCalls) {
  call_count = 0;
  counter();
  EXPECT_EQ(call_count, 1);
  counter();
  EXPECT_EQ(call_count, 2);
  counter();
  EXPECT_EQ(call_count, 3);
}
