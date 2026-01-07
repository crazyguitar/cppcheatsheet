#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netdb.h>

TEST(GetAddrInfo, Localhost) {
  struct addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res;

  ASSERT_EQ(getaddrinfo("localhost", nullptr, &hints, &res), 0);
  ASSERT_NE(res, nullptr);

  bool found = false;
  for (auto* p = res; p; p = p->ai_next) {
    char ip[INET6_ADDRSTRLEN];
    void* addr =
        p->ai_family == AF_INET ? (void*)&((struct sockaddr_in*)p->ai_addr)->sin_addr : (void*)&((struct sockaddr_in6*)p->ai_addr)->sin6_addr;
    inet_ntop(p->ai_family, addr, ip, sizeof(ip));
    found = true;
  }
  EXPECT_TRUE(found);
  freeaddrinfo(res);
}
