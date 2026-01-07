#include <gtest/gtest.h>

#include <csignal>

TEST(SignalMask, BlockUnblock) {
  sigset_t mask, oldmask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGUSR1);
  sigprocmask(SIG_BLOCK, &mask, &oldmask);
  sigprocmask(SIG_UNBLOCK, &mask, nullptr);
}
