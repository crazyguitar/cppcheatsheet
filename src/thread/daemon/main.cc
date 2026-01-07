#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(Daemon, ForkSetsid) {
  pid_t pid = fork();
  if (pid == 0) {
    umask(0);
    setsid();
    _exit(0);
  } else {
    wait(nullptr);
  }
}
