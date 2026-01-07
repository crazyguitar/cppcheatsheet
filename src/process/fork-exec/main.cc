#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(ForkExec, RunCommand) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    char* args[] = {(char*)"true", NULL};
    execvp("true", args);
    _exit(1);
  }
  int status;
  waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
