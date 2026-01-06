/**
 * @file client.h
 * @brief Asynchronous TCP client with coroutine support
 */
#pragma once

#include <fcntl.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "common.h"
#include "coro.h"
#include "event.h"
#include "io.h"

// SOCK_NONBLOCK is Linux-specific
#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK 0
#endif
#include "stream.h"

/**
 * @brief Asynchronous TCP client for non-blocking connections
 *
 * Provides coroutine-based TCP client functionality with epoll integration.
 * Supports hostname resolution and non-blocking connect operations.
 */
class Client : private NoCopy {
  /**
   * @brief Internal socket wrapper for connection management
   */
  class Socket : private NoCopy {
    /**
     * @brief Coroutine awaiter for asynchronous connection completion
     *
     * Suspends coroutine while connect() is in progress, resumes on completion.
     */
    struct ConnectionAwaiter {
      Socket* socket = nullptr;
      Event event;

      ConnectionAwaiter(Socket* socket, int fd) : socket{socket}, event(fd) {}

      /** @brief Always suspend to wait for connection */
      constexpr bool await_ready() const noexcept { return false; }

      /** @brief Unregister from epoll after connection completes */
      void await_resume() noexcept {
        auto& io = IO::Get();
        io.Quit<Selector>(event);
      }

      /**
       * @brief Register for kEventWrite and suspend
       * @return true to suspend coroutine
       */
      template <typename Promise>
      bool await_suspend(std::coroutine_handle<Promise> coroutine) {
        auto& io = IO::Get();
        event.handle = &coroutine.promise();
        event.flags = kEventWrite;
        io.Join<Selector>(event);
        return true;
      }
    };

   public:
    Socket() = delete;

    /**
     * @brief Construct socket with target address
     * @param ip IP address or hostname to connect to
     * @param port Port number to connect to
     */
    Socket(const std::string& ip, int port) : ip_{ip}, port_{port} {}

    ~Socket() { Close(); }

    Socket(Socket&&) = delete;
    Socket& operator=(Socket&&) = delete;

    friend class Client;

    /**
     * @brief Establish connection to remote host
     * @return Coroutine returning connected Stream
     * @throws std::runtime_error if connection fails
     */
    Coro<Stream> Connect() {
      // Free any previous addrinfo before potentially reallocating
      if (addrinfo_) {
        freeaddrinfo(addrinfo_);
        addrinfo_ = nullptr;
      }

      struct addrinfo hints{.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM};
      auto service = std::to_string(port_);
      int gai_err = getaddrinfo(ip_.data(), service.c_str(), &hints, &addrinfo_);
      if (gai_err != 0) {
        // error
        throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(gai_err)));
      }

      for (auto p = addrinfo_; !!p; p = p->ai_next) {
        if ((fd_ = socket(p->ai_family, p->ai_socktype | SOCK_NONBLOCK, p->ai_protocol)) == -1) continue;
        if (co_await Connect(fd_, p->ai_addr, p->ai_addrlen)) break;
        close(fd_);
        fd_ = -1;
      }

      if (fd_ < 0) {
        freeaddrinfo(addrinfo_);
        addrinfo_ = nullptr;
        // error
        throw std::runtime_error("Failed to connect to " + ip_ + ":" + service);
      }

      // Transfer ownership of fd to Stream - Socket no longer owns it
      int connected_fd = fd_;
      fd_ = -1;
      co_return Stream(connected_fd);
    }

    /**
     * @brief Close socket and free resources
     *
     * Safe to call multiple times.
     */
    void Close() noexcept {
      if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
      }
      if (addrinfo_) {
        freeaddrinfo(addrinfo_);
        addrinfo_ = nullptr;
      }
    }

   private:
    /**
     * @brief Attempt non-blocking connect
     * @param fd Socket file descriptor
     * @param addr Target address
     * @param len Address length
     * @return Coroutine returning true on success
     */
    Coro<bool> Connect(int fd, const sockaddr* addr, socklen_t len) {
      if (!SetBlocking(fd, false)) co_return false;
      int rc = connect(fd, addr, len);
      if (rc == 0) co_return true;

      if (rc < 0 and errno != EINPROGRESS) {
        int conn_errno = errno;
        // error
        co_return false;
      }

      co_await ConnectionAwaiter{this, fd};
      int result{0};
      socklen_t result_len = sizeof(result);
      if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &result, &result_len) < 0) {
        int err = errno;
        // warn
        co_return false;
      }
      co_return result == 0;
    }

    /**
     * @brief Set socket blocking mode
     * @param fd Socket file descriptor
     * @param blocking true for blocking, false for non-blocking
     * @return true on success
     */
    static bool SetBlocking(int fd, bool blocking) noexcept {
      int flags = fcntl(fd, F_GETFL, 0);
      if (flags < 0) {
        // warn
        return false;
      }
      flags = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
      if (fcntl(fd, F_SETFL, flags) < 0) {
        // warn
        return false;
      }
      return true;
    }

   private:
    int fd_ = -1;                          ///< Socket file descriptor
    struct addrinfo* addrinfo_ = nullptr;  ///< Address information from getaddrinfo
    std::string ip_;                       ///< IP address or hostname to connect to
    int port_;                             ///< Port number to connect to
  };

 public:
  Client() = delete;

  /**
   * @brief Construct client with target address
   * @param ip IP address or hostname to connect to
   * @param port Port number to connect to
   */
  Client(const std::string& ip, int port) : ip_{ip}, port_{port}, socket_{ip, port} {}

  Client(Client&&) = delete;
  Client& operator=(Client&&) = delete;

  /**
   * @brief Connect to the configured remote host
   * @return Coroutine returning connected Stream
   * @throws std::runtime_error if connection fails
   */
  Coro<Stream> Connect() { co_return co_await socket_.Connect(); }

 private:
  std::string ip_;  ///< IP address or hostname to connect to
  int port_;        ///< Port number to connect to
  Socket socket_;   ///< Internal socket wrapper
};
