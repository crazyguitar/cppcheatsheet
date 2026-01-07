#include <gtest/gtest.h>
#include <sys/stat.h>

TEST(FileType, CheckTypes) {
  struct stat st;
  stat("/tmp", &st);
  // just run the type check logic
  switch (st.st_mode & S_IFMT) {
    case S_IFREG:
      break;
    case S_IFDIR:
      break;
    case S_IFLNK:
      break;
    case S_IFBLK:
      break;
    case S_IFCHR:
      break;
    case S_IFIFO:
      break;
    case S_IFSOCK:
      break;
    default:
      break;
  }
}
