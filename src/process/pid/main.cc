#include <gtest/gtest.h>
#include <unistd.h>

TEST(Pid, GetIds) {
  EXPECT_GT(getpid(), 0);
  EXPECT_GT(getppid(), 0);
  EXPECT_GT(getpgrp(), 0);
  EXPECT_GT(getsid(0), 0);
}
