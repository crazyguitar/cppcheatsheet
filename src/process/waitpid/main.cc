#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(Waitpid, NonBlocking) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    _exit(0);
  }
  int status;
  pid_t result;
  do {
    result = waitpid(pid, &status, WNOHANG);
  } while (result == 0);
  EXPECT_EQ(result, pid);
  EXPECT_TRUE(WIFEXITED(status));
}
