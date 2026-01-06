#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>

TEST(UnixSocket, Setup) {
  const char* path = "/tmp/test_echo.sock";
  unlink(path);
  int s = socket(AF_UNIX, SOCK_STREAM, 0);

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  bind(s, (struct sockaddr*)&addr, sizeof(addr));
  listen(s, 10);
  close(s);
  unlink(path);
}
