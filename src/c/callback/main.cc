#include <gtest/gtest.h>

namespace {
int callback_result = -999;

void on_complete(int result) { callback_result = result; }

void do_work(int should_fail, void (*cb)(int)) { cb(should_fail ? -1 : 0); }
}  // namespace

TEST(Callback, InvokesCallback) {
  callback_result = -999;
  do_work(0, on_complete);
  EXPECT_EQ(callback_result, 0);

  do_work(1, on_complete);
  EXPECT_EQ(callback_result, -1);
}
