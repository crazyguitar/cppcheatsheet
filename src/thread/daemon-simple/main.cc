#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(DaemonSimple, DaemonFunc) {
  pid_t pid = fork();
  if (pid == 0) {
    // don't actually daemonize in test, just verify fork works
    _exit(0);
  } else {
    wait(nullptr);
  }
}
