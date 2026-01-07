#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

TEST(CopyFile, Basic) {
  char src[] = "/tmp/srcXXXXXX";
  char dst[] = "/tmp/dstXXXXXX";
  int sfd = mkstemp(src);
  write(sfd, "test data", 9);
  lseek(sfd, 0, SEEK_SET);

  struct stat st;
  fstat(sfd, &st);
  int dfd = mkstemp(dst);

  char buf[1024];
  ssize_t n;
  while ((n = read(sfd, buf, sizeof(buf))) > 0) write(dfd, buf, n);

  close(sfd);
  close(dfd);
  unlink(src);
  unlink(dst);
}
