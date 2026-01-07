#include <arpa/inet.h>
#include <gtest/gtest.h>

TEST(InetConv, IPv4) {
  struct in_addr addr4;
  inet_pton(AF_INET, "192.168.1.1", &addr4);
  char str4[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &addr4, str4, sizeof(str4));
  EXPECT_STREQ(str4, "192.168.1.1");
  EXPECT_EQ(ntohl(addr4.s_addr), 0xc0a80101);
}

TEST(InetConv, IPv6) {
  struct in6_addr addr6;
  inet_pton(AF_INET6, "::1", &addr6);
  char str6[INET6_ADDRSTRLEN];
  inet_ntop(AF_INET6, &addr6, str6, sizeof(str6));
  EXPECT_STREQ(str6, "::1");
}
