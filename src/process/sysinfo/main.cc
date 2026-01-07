#include <gtest/gtest.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstring>

TEST(Sysinfo, Uname) {
  struct utsname buf;
  ASSERT_EQ(uname(&buf), 0);
  EXPECT_GT(strlen(buf.sysname), 0u);
  EXPECT_GT(strlen(buf.nodename), 0u);
  EXPECT_GT(strlen(buf.release), 0u);
  EXPECT_GT(strlen(buf.machine), 0u);
}

TEST(Sysinfo, Hostname) {
  char hostname[256];
  ASSERT_EQ(gethostname(hostname, sizeof(hostname)), 0);
  EXPECT_GT(strlen(hostname), 0u);
}
