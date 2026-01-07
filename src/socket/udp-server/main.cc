#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

TEST(UdpServer, Setup) {
  int s = socket(AF_INET, SOCK_DGRAM, 0);

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(15567);
  addr.sin_addr.s_addr = INADDR_ANY;
  bind(s, (struct sockaddr*)&addr, sizeof(addr));
  close(s);
}
