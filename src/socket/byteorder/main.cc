#include <arpa/inet.h>
#include <gtest/gtest.h>

TEST(ByteOrder, HostToNetwork) {
  uint16_t port = 8080;
  uint32_t addr = 0x7f000001;  // 127.0.0.1

  EXPECT_EQ(ntohs(htons(port)), port);
  EXPECT_EQ(ntohl(htonl(addr)), addr);
}
