#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(Fifo, NamedPipe) {
  const char* path = "/tmp/test_fifo";
  unlink(path);
  mkfifo(path, 0666);

  pid_t pid = fork();
  if (pid == 0) {
    int fd = open(path, O_WRONLY);
    write(fd, "test", 5);
    close(fd);
    _exit(0);
  } else {
    int fd = open(path, O_RDONLY);
    char buf[128];
    read(fd, buf, sizeof(buf));
    close(fd);
    wait(nullptr);
    unlink(path);
  }
}
