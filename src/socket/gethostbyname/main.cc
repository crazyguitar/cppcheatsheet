#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netdb.h>

TEST(GetHostByName, Localhost) {
  struct hostent* h = gethostbyname("localhost");
  ASSERT_NE(h, nullptr);
  EXPECT_STREQ(h->h_name, "localhost");
  ASSERT_NE(h->h_addr_list[0], nullptr);
}
