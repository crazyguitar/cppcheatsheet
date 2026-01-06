#pragma once

#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>

#include "common.h"
#include "coro.h"
#include "event.h"
#include "handle.h"
#include "selector.h"

// Forward declare IO to break circular dependency
class IO;

/**
 * @brief Non-blocking socket stream with coroutine-based async I/O
 *
 * Provides async read, write, and accept operations using C++20 coroutines.
 * Uses edge-triggered epoll for event notification. Only registers for kEventRead
 * (read events) - write operations return immediately with partial write count.
 *
 * Registration with epoll is deferred until first await to support move semantics.
 */
class Stream : public Event, private NoCopy {
  /**
   * @brief Awaiter for async accept operations
   *
   * Attempts accept immediately. If EAGAIN, registers with epoll and suspends.
   */
  struct AcceptAwaiter {
    Stream* stream = nullptr;
    mutable int client_fd = -1;

    constexpr bool await_ready() const noexcept { return false; }

    int await_resume() const noexcept {
      stream->handle = nullptr;
      if (stream->closed_) [[unlikely]]
        return -1;
      if (client_fd >= 0) [[likely]]
        return client_fd;
      struct sockaddr_storage addr{};
      socklen_t addrlen = sizeof(addr);
      return accept(stream->fd, reinterpret_cast<sockaddr*>(&addr), &addrlen);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) const {
      if (!stream) [[unlikely]]
        throw std::invalid_argument("invalid stream. stream is nullptr.");
      if (stream->fd < 0) [[unlikely]]
        throw std::invalid_argument("invalid fd. fd < 0.");
      if (stream->handle) [[unlikely]]
        throw std::invalid_argument("previous coroutine is suspending.");
      if (stream->closed_) [[unlikely]]
        return false;
      stream->handle = &coroutine.promise();
      struct sockaddr_storage addr{};
      socklen_t addrlen = sizeof(addr);
      client_fd = accept(stream->fd, reinterpret_cast<sockaddr*>(&addr), &addrlen);
      if (client_fd < 0) {
        stream->Join();  // Register with epoll before suspending
        return true;
      }
      return false;
    }
  };

  /**
   * @brief Awaiter for async read operations
   *
   * Attempts read immediately. If data available, returns without suspending.
   * If EAGAIN, registers with epoll and suspends until data ready.
   */
  struct ReadAwaiter {
    Stream* stream = nullptr;
    char* data = nullptr;
    size_t len = 0;
    mutable ssize_t n = -1;               // Use ssize_t to properly handle -1 error case
    mutable bool read_succeeded = false;  // True only if initial read got data

    constexpr bool await_ready() const noexcept { return false; }

    size_t await_resume() const noexcept {
      stream->handle = nullptr;
      if (stream->closed_) [[unlikely]]
        return static_cast<size_t>(-1);
      // If initial read succeeded, return cached result
      if (read_succeeded) return static_cast<size_t>(n);
      // Otherwise try reading again (after epoll woke us up)
      ssize_t result = read(stream->fd, data, len);
      return result >= 0 ? static_cast<size_t>(result) : static_cast<size_t>(-1);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) const {
      if (!stream) [[unlikely]]
        throw std::runtime_error("stream is null");
      if (stream->fd < 0) [[unlikely]]
        throw std::invalid_argument("invalid fd. fd < 0.");
      if (stream->handle) [[unlikely]]
        throw std::runtime_error("previous read not finished.");
      if (!data || len == 0) [[unlikely]]
        throw std::invalid_argument("invalid data or size");
      if (stream->closed_) [[unlikely]]
        return false;
      stream->handle = &coroutine.promise();
      n = read(stream->fd, data, len);
      if (n >= 0) {
        read_succeeded = true;
        return false;  // Don't suspend, data is ready
      }
      // Read would block, register with epoll and suspend
      stream->Join();
      return true;
    }
  };

  /**
   * @brief Awaiter for async write operations
   *
   * On Linux: Attempts to write all data immediately, returns partial count on EAGAIN.
   * On macOS: Suspends on EAGAIN and waits for write readiness via kqueue.
   */
  struct WriteAwaiter {
    Stream* stream = nullptr;
    const char* data = nullptr;
    size_t len = 0;
    mutable size_t written = 0;
    mutable int err = 0;
#if defined(__APPLE__) || defined(__FreeBSD__)
    mutable bool suspended_for_write = false;
#endif

    constexpr bool await_ready() const noexcept { return false; }

    size_t await_resume() const {
      stream->handle = nullptr;
#if defined(__APPLE__) || defined(__FreeBSD__)
      if (suspended_for_write) {
        stream->ReregisterRead();
        while (written < len) {
          ssize_t n = write(stream->fd, data + written, len - written);
          if (n > 0) {
            written += n;
            continue;
          }
          if (n == 0) break;
          int e = errno;
          if (e == EINTR) continue;
          if (e == EAGAIN || e == EWOULDBLOCK) break;
          err = e;
          break;
        }
      }
#endif
      if (stream->closed_) [[unlikely]]
        return -1;
      if (written == 0 && err != 0 && err != EAGAIN && err != EWOULDBLOCK) return -1;
      return written;
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) const {
      if (!stream) [[unlikely]]
        throw std::runtime_error("stream is null");
      if (stream->fd < 0) [[unlikely]]
        throw std::invalid_argument("invalid fd. fd < 0.");
      if (stream->handle) [[unlikely]]
        throw std::runtime_error("previous write not finished.");
      if (!data || len == 0) [[unlikely]]
        throw std::invalid_argument("invalid data or size");
      if (stream->closed_) [[unlikely]]
        return false;
      stream->handle = &coroutine.promise();
      while (written < len) {
        ssize_t n = write(stream->fd, data + written, len - written);
        if (n > 0) {
          written += n;
          continue;
        }
        if (n == 0) break;
        err = errno;
        if (err == EINTR) continue;
#if defined(__APPLE__) || defined(__FreeBSD__)
        if (err == EAGAIN || err == EWOULDBLOCK) {
          stream->JoinWrite();
          suspended_for_write = true;
          return true;
        }
#endif
        break;
      }
      return false;
    }
  };

 public:
  Stream() = delete;

  /**
   * @brief Construct Stream from socket file descriptor
   * @param socket_fd Non-blocking socket file descriptor
   *
   * Does not register with epoll immediately - registration deferred until
   * first await to support move semantics without stale addresses.
   */
  explicit Stream(int socket_fd) noexcept : Event{socket_fd, kEventRead, nullptr}, closed_{false}, registered_{false} {}

  /**
   * @brief Move constructor - transfers ownership of socket and state
   * @param other Source Stream to move from (will be closed)
   */
  Stream(Stream&& other) noexcept
      : Event(std::move(other)), closed_{std::exchange(other.closed_, true)}, registered_{std::exchange(other.registered_, false)} {}

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      fd = std::exchange(other.fd, -1);
      flags = std::exchange(other.flags, 0);
      handle = std::exchange(other.handle, nullptr);
      closed_ = std::exchange(other.closed_, true);
      registered_ = std::exchange(other.registered_, false);
    }
    return *this;
  }

  /**
   * @brief Destructor - closes socket and unregisters from epoll
   */
  ~Stream() noexcept {
    closed_ = true;
    handle = nullptr;
    Quit();
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }

  /**
   * @brief Async read from socket
   * @param buffer Buffer to read into
   * @param len Buffer size
   * @return Coroutine returning bytes read, or -1 on error
   * @throws std::invalid_argument if fd < 0, buffer is null, or len is 0
   */
  [[nodiscard]] Coro<size_t> Read(char* __restrict__ buffer, size_t len) {
    if (fd < 0) throw std::invalid_argument("invalid fd < 0");
    if (!buffer) throw std::invalid_argument("buffer is nullptr");
    if (len == 0) throw std::invalid_argument("invalid buffer size");
    co_return co_await ReadAwaiter{this, buffer, len};
  }

  /**
   * @brief Async write to socket
   * @param data Data to write
   * @param len Data length
   * @return Coroutine returning bytes written, or -1 on error
   * @throws std::invalid_argument if fd < 0, data is null, or len is 0
   *
   * On macOS: Loops until all data written, suspending on buffer full.
   * On Linux: Single write attempt, may return partial count.
   */
  [[nodiscard]] Coro<size_t> Write(const char* __restrict__ data, size_t len) {
    if (fd < 0) throw std::invalid_argument("invalid fd < 0");
    if (!data) throw std::invalid_argument("data is nullptr");
    if (len == 0) throw std::invalid_argument("invalid buffer size");
#if defined(__APPLE__) || defined(__FreeBSD__)
    size_t total = 0;
    while (total < len) {
      size_t n = co_await WriteAwaiter{this, data + total, len - total};
      if (n == static_cast<size_t>(-1)) co_return static_cast<size_t>(-1);
      if (n == 0) break;
      total += n;
    }
    co_return total;
#else
    co_return co_await WriteAwaiter{this, data, len};
#endif
  }

  /**
   * @brief Async accept incoming connection
   * @return Coroutine returning client socket fd, or -1 on error
   * @throws std::invalid_argument if fd < 0
   */
  [[nodiscard]] Coro<int> Accept() {
    if (fd < 0) throw std::invalid_argument("invalid fd < 0");
    co_return co_await AcceptAwaiter{this};
  }

  /**
   * @brief Check if socket is open
   * @return true if fd >= 0, false otherwise
   */
  [[nodiscard]] bool IsOpen() const noexcept { return fd >= 0; }

  /**
   * @brief Register the Stream with selector for read events
   */
  void Join() {
    if (!registered_ && fd >= 0) {
      auto& io = IO::Get();
      io.Join<Selector>(*this);
      registered_ = true;
    }
  }

#if defined(__APPLE__) || defined(__FreeBSD__)
  /**
   * @brief Register the Stream with kqueue for write events
   */
  void JoinWrite() {
    if (fd >= 0) {
      auto& io = IO::Get();
      if (registered_) {
        io.Quit<Selector>(*this);
        registered_ = false;
      }
      auto saved_flags = flags;
      flags = kEventWrite;
      io.Join<Selector>(*this);
      flags = saved_flags;
      registered_ = true;
    }
  }

  /**
   * @brief Re-register for read events after write completes
   */
  void ReregisterRead() {
    if (fd >= 0) {
      auto& io = IO::Get();
      if (registered_) {
        auto saved_flags = flags;
        flags = kEventWrite;
        io.Quit<Selector>(*this);
        flags = saved_flags;
        registered_ = false;
      }
      io.Join<Selector>(*this);
      registered_ = true;
    }
  }
#endif

 private:
  /**
   * @brief Unregister the Stream from selector
   */
  void Quit() noexcept {
    if (registered_ && fd >= 0) {
      auto& io = IO::Get();
      io.Quit<Selector>(*this);
      registered_ = false;
    }
  }

 private:
  bool closed_ = false;
  bool registered_ = false;
};
