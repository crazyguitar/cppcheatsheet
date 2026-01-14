#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(Pipe, ParentChildCommunication) {
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  pid_t pid = fork();
  if (pid == 0) {
    close(pipefd[1]);
    char buf[100] = {0};
    ssize_t n = read(pipefd[0], buf, sizeof(buf));
    close(pipefd[0]);
    _exit(n == 5 && strncmp(buf, "hello", 5) == 0 ? 0 : 1);
  } else {
    close(pipefd[0]);
    write(pipefd[1], "hello", 5);
    close(pipefd[1]);
    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
}
