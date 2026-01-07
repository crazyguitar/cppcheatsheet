#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

TEST(FstatSize, GetFileSize) {
  char tmpl[] = "/tmp/testXXXXXX";
  int fd = mkstemp(tmpl);
  write(fd, "hello", 5);

  struct stat st;
  fstat(fd, &st);

  close(fd);
  unlink(tmpl);
}
