#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(Socketpair, Bidirectional) {
  int fd[2];
  socketpair(AF_UNIX, SOCK_STREAM, 0, fd);

  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    char buf[128];
    read(fd[1], buf, sizeof(buf));
    write(fd[1], "reply", 6);
    close(fd[1]);
    _exit(0);
  } else {
    close(fd[1]);
    write(fd[0], "hello", 6);
    char buf[128];
    read(fd[0], buf, sizeof(buf));
    close(fd[0]);
    wait(NULL);
  }
}
