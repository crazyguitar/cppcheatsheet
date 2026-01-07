#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <io/client.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/runner.h>
#include <io/server.h>
#include <io/stream.h>

int GetPort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(0);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return 0;
  }
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    close(fd);
    return 0;
  }
  int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

TEST(StreamTest, ReadWrite) {
  int port = GetPort();
  ASSERT_GT(port, 0);
  std::string received;

  ::Run([&]() -> Coro<> {
    auto handler = [](Stream s) -> Coro<> {
      char buf[256];
      size_t n = co_await s.Read(buf, sizeof(buf));
      if (n > 0) co_await s.Write(buf, n);
    };
    auto server_coro = [&]() -> Coro<> {
      auto srv = Server("127.0.0.1", port, handler);
      srv.Start();
      co_await srv.Wait();
    };
    auto client_coro = [&]() -> Coro<> {
      Client c("127.0.0.1", port);
      auto s = co_await c.Connect();
      co_await s.Write("hello", 5);
      char buf[256];
      size_t n = co_await s.Read(buf, sizeof(buf));
      received = std::string(buf, n);
    };
    auto srv = Future(server_coro());
    co_await client_coro();
    srv.Cancel();
  }());
  EXPECT_EQ(received, "hello");
}
