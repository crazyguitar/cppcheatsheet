#include <gtest/gtest.h>
#include <sys/wait.h>

#include <cstdio>

TEST(Popen, RunCommand) {
  FILE* fp = popen("echo test", "r");
  char buf[256];
  while (fgets(buf, sizeof(buf), fp)) {
  }
  pclose(fp);
}
