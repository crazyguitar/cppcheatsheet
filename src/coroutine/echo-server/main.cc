#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <io/client.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/runner.h>
#include <io/server.h>
#include <io/stream.h>

#include <string>
#include <vector>

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

TEST(EchoServerTest, LargeTransfer) {
  int port = GetPort();
  ASSERT_GT(port, 0);
  constexpr size_t kSize = 10 * 1024 * 1024, kChunk = 64 * 1024;
  std::vector<char> send_buf(kChunk);
  for (size_t i = 0; i < kChunk; ++i) send_buf[i] = static_cast<char>(i % 256);
  size_t server_recv = 0, client_recv = 0;

  ::Run([&]() -> Coro<> {
    auto handler = [&](Stream s) -> Coro<> {
      std::vector<char> data;
      data.reserve(kSize);
      char buf[kChunk];
      while (data.size() < kSize) {
        size_t n = co_await s.Read(buf, sizeof(buf));
        if (n == 0) break;
        data.insert(data.end(), buf, buf + n);
      }
      server_recv = data.size();
      size_t sent = 0;
      while (sent < data.size()) {
        size_t n = co_await s.Write(data.data() + sent, std::min(kChunk, data.size() - sent));
        if (n == 0) break;
        sent += n;
      }
    };
    auto server_coro = [&]() -> Coro<> {
      auto srv = Server("127.0.0.1", port, handler);
      srv.Start();
      co_await srv.Wait();
    };
    auto client_coro = [&]() -> Coro<> {
      Client c("127.0.0.1", port);
      auto s = co_await c.Connect();
      size_t sent = 0;
      while (sent < kSize) {
        size_t n = co_await s.Write(send_buf.data(), std::min(kChunk, kSize - sent));
        if (n == 0) break;
        sent += n;
      }
      std::vector<char> data;
      data.reserve(kSize);
      char buf[kChunk];
      while (data.size() < kSize) {
        size_t n = co_await s.Read(buf, sizeof(buf));
        if (n == 0) break;
        data.insert(data.end(), buf, buf + n);
      }
      client_recv = data.size();
    };
    auto srv = Future(server_coro());
    co_await client_coro();
    srv.Cancel();
  }());
  EXPECT_EQ(server_recv, kSize);
  EXPECT_EQ(client_recv, kSize);
}

TEST(EchoServerTest, MultiClient) {
  int port = GetPort();
  ASSERT_GT(port, 0);
  constexpr int kNum = 16;
  std::array<std::string, kNum> received;

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
    auto client_coro = [&](int id) -> Coro<> {
      Client c("127.0.0.1", port);
      auto s = co_await c.Connect();
      std::string msg = "client_" + std::to_string(id);
      co_await s.Write(msg.data(), msg.size());
      char buf[256];
      size_t n = co_await s.Read(buf, sizeof(buf));
      received[id] = std::string(buf, n);
    };
    auto srv = Future(server_coro());
    std::vector<Future<Coro<>>> futs;
    for (int i = 0; i < kNum; ++i) futs.emplace_back(client_coro(i));
    for (int i = 0; i < kNum; ++i) co_await std::move(futs[i]);
    srv.Cancel();
  }());
  for (int i = 0; i < kNum; ++i) EXPECT_EQ(received[i], "client_" + std::to_string(i));
}
