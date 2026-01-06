/**
 * @file server.h
 * @brief Asynchronous TCP server with coroutine support
 */
#pragma once
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <list>
#include <stdexcept>
#include <string>

#include "common.h"
#include "coro.h"
#include "event.h"

// SOCK_NONBLOCK is Linux-specific
#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK 0
#endif
#include "future.h"
#include "io.h"
#include "stream.h"

#ifndef SOMAXCONN
#define SOMAXCONN 1024
#endif

/**
 * @brief Asynchronous TCP server with coroutine-based connection handling
 *
 * Provides non-blocking TCP server functionality with epoll integration.
 * Accepts connections and dispatches them to a user-provided handler function.
 *
 * @tparam F Connection handler type that takes Stream and returns Coro<>
 */
template <typename F>
class Server : private NoCopy {
  /**
   * @brief Non-blocking TCP socket wrapper for server operations
   */
  class Socket : private NoCopy {
   public:
    Socket() = default;
    ~Socket() { Close(); }

    Socket(Socket&&) = delete;
    Socket& operator=(Socket&&) = delete;

    template <typename C>
    friend class Server;

    /**
     * @brief Open and bind a non-blocking listening socket
     * @param addr Address to bind to (hostname or IP)
     * @param port Port number to listen on
     * @throws std::runtime_error if socket creation, bind, or listen fails
     */
    void Open(const std::string& addr, int port) {
      struct addrinfo hints{.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM};
      auto service = std::to_string(port);
      int gai_err = getaddrinfo(addr.data(), service.c_str(), &hints, &addrinfo_);
      if (gai_err != 0) {
        // error
        throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(gai_err)));
      }

      for (auto p = addrinfo_; !!p; p = p->ai_next) {
        if ((fd_ = socket(p->ai_family, p->ai_socktype | SOCK_NONBLOCK, p->ai_protocol)) == -1) continue;
        if (Bind(fd_, p)) break;
        close(fd_);
        fd_ = -1;
      }

      if (fd_ < 0) {
        freeaddrinfo(addrinfo_);
        addrinfo_ = nullptr;
        // error
        throw std::runtime_error("Failed to bind socket to " + addr + ":" + std::to_string(port));
      }

      if (listen(fd_, SOMAXCONN) < 0) {
        int saved_errno = errno;
        // error
        Close();
        throw std::runtime_error("listen failed: " + std::string(strerror(saved_errno)));
      }
    }

    /**
     * @brief Close socket and free address info
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

    /**
     * @brief Check if the socket is open and listening
     * @return true if the socket file descriptor is valid
     */
    [[nodiscard]] bool IsOpen() const noexcept { return fd_ >= 0; }

   private:
    /**
     * @brief Bind socket to address
     * @param fd Socket file descriptor
     * @param p Address info to bind to
     * @return true on success
     */
    static bool Bind(int fd, struct addrinfo* p) noexcept {
      if (!SetSocket(fd)) return false;
      if (bind(fd, p->ai_addr, p->ai_addrlen) < 0) {
        // warn
        return false;
      }
      return true;
    }

    /**
     * @brief Configure socket options
     * @param fd Socket file descriptor
     * @return true on success
     */
    static bool SetSocket(int fd) noexcept { return SetBlocking(fd, false) and SetSockopt(fd); }

    /**
     * @brief Set SO_REUSEADDR socket option
     * @param fd Socket file descriptor
     * @return true on success
     */
    static bool SetSockopt(int fd) noexcept {
      int on = 1;
      if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on))) {
        // warn
        return false;
      }
      return true;
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
  };

 public:
  Server() = delete;

  /**
   * @brief Construct a server bound to the specified address and port
   * @param ip IP address or hostname to bind to
   * @param port Port number to listen on
   * @param fn Connection handler function that takes a Stream and returns Coro<>
   */
  Server(const std::string& ip, int port, F fn) : ip_{ip}, port_{port}, fn_{fn} {}

  /**
   * @brief Destructor - unregisters from epoll if socket is open
   */
  ~Server() {
    if (socket_.fd_ >= 0) {
      Quit(socket_.fd_);
    }
  }

  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;

  /**
   * @brief Open and start listening on the configured address/port
   * @throws std::runtime_error if socket creation, bind, or listen fails
   */
  void Start() { socket_.Open(ip_, port_); }

  /**
   * @brief Accept connections in a loop
   * @return Coroutine that runs until stopped or error
   */
  Coro<> Wait() {
    Stream stream{socket_.fd_};
    while (!stopped) {
      auto client_fd = co_await stream.Accept();
      if (client_fd >= 0) {
        HandleClient(client_fd);
      } else {
        int err = errno;
        if (err == EAGAIN || err == EWOULDBLOCK) {
          continue;
        }
        if (err == EINTR) {
          continue;
        }
        if (err == EMFILE || err == ENFILE || err == ENOMEM || err == ENOBUFS) {
          // warn
          break;
        }
        // warn
        break;
      }
    }
  }

 private:
  /**
   * @brief Handle new client connection
   * @param client_fd Client socket file descriptor
   */
  void HandleClient(int client_fd) {
    if (!Socket::SetBlocking(client_fd, false)) {
      // warn
      close(client_fd);
      return;
    }
    connected_.emplace_back(fn_(Stream{client_fd}));
    this->Clean(connected_);
  }

  /**
   * @brief Unregister socket from epoll
   * @param fd Socket file descriptor
   */
  void Quit(int fd) {
    auto& io = IO::Get();
    Stream stream{fd};
    io.Quit(stream);
  }

 private:
  constexpr static size_t kConn = 128;  ///< Max connections before cleanup

  /**
   * @brief Remove completed connections from list
   * @param connected List of active connections
   */
  void Clean(std::list<Future<Coro<>>>& connected) {
    if (connected.size() < kConn) [[likely]]
      return;
    for (auto it = connected.begin(); it != connected.end();) {
      if (it->done())
        it = connected.erase(it);
      else
        ++it;
    }
  }

 private:
  bool stopped = false;                  ///< Server stop flag
  std::string ip_;                       ///< IP address or hostname to bind to
  int port_;                             ///< Port number to listen on
  F fn_;                                 ///< Connection handler function
  Socket socket_;                        ///< Listening socket
  std::list<Future<Coro<>>> connected_;  ///< Active client connections
};
