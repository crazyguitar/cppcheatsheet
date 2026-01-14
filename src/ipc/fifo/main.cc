#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(FIFO, NamedPipeCommunication) {
  const char* path = "/tmp/test_fifo";
  unlink(path);
  ASSERT_EQ(mkfifo(path, 0666), 0);

  pid_t pid = fork();
  if (pid == 0) {
    int fd = open(path, O_WRONLY);
    write(fd, "fifo_test", 9);
    close(fd);
    _exit(0);
  } else {
    int fd = open(path, O_RDONLY);
    char buf[100];
    ssize_t n = read(fd, buf, sizeof(buf));
    close(fd);
    wait(NULL);
    unlink(path);
    EXPECT_EQ(n, 9);
    EXPECT_EQ(strncmp(buf, "fifo_test", 9), 0);
  }
}
