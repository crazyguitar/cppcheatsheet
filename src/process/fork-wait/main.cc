#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(ForkWait, ChildExitStatus) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    _exit(42);
  }
  int status;
  wait(&status);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 42);
}
