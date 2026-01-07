#include <gtest/gtest.h>

#include <csignal>

static volatile sig_atomic_t got_signal = 0;
static void handler(int) { got_signal = 1; }

TEST(SigactionHandler, Setup) {
  struct sigaction sa{};
  sa.sa_handler = handler;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGUSR1, &sa, nullptr);
  raise(SIGUSR1);
  sa.sa_handler = SIG_DFL;
  sigaction(SIGUSR1, &sa, nullptr);
}
