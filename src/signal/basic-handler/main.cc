#include <gtest/gtest.h>

#include <csignal>

static volatile sig_atomic_t got_signal = 0;
static void handler(int) { got_signal = 1; }

TEST(BasicHandler, Signal) {
  signal(SIGUSR1, handler);
  raise(SIGUSR1);
  signal(SIGUSR1, SIG_DFL);
}
