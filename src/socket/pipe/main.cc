#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(Pipe, Basic) {
  int fd[2];
  pipe(fd);

  pid_t pid = fork();
  if (pid == 0) {
    close(fd[1]);
    char buf[128];
    read(fd[0], buf, sizeof(buf));
    close(fd[0]);
    _exit(0);
  } else {
    close(fd[0]);
    write(fd[1], "test", 5);
    close(fd[1]);
    wait(NULL);
  }
}
