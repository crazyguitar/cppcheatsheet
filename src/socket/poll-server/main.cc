#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

TEST(PollServer, Setup) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int on = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(15569);
  addr.sin_addr.s_addr = INADDR_ANY;
  bind(s, (struct sockaddr*)&addr, sizeof(addr));
  listen(s, 10);

  struct pollfd fds[1];
  fds[0].fd = s;
  fds[0].events = POLLIN;
  close(s);
}
