#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

TEST(LseekSize, GetFileSize) {
  char tmpl[] = "/tmp/testXXXXXX";
  int fd = mkstemp(tmpl);
  write(fd, "hello", 5);

  off_t start = lseek(fd, 0, SEEK_SET);
  off_t end = lseek(fd, 0, SEEK_END);

  close(fd);
  unlink(tmpl);
}
