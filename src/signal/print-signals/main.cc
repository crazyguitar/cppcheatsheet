#include <gtest/gtest.h>

#include <csignal>
#include <cstring>

TEST(PrintSignals, Strsignal) {
  // just verify strsignal works
  const char* name = strsignal(SIGINT);
  (void)name;
}
