/**
 * @file selector.h
 * @brief Cross-platform I/O event multiplexing (epoll/kqueue)
 */
#pragma once

#include <chrono>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "event.h"

#ifdef __linux__
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#else
#error "Unsupported platform"
#endif

#ifdef __linux__

constexpr uint32_t kEventRead = EPOLLIN;
constexpr uint32_t kEventWrite = EPOLLOUT;
constexpr uint32_t kEventError = EPOLLERR | EPOLLHUP;

class EpollSelector : private NoCopy {
 public:
  using ms = std::chrono::milliseconds;

  EpollSelector();
  ~EpollSelector();

  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500});
  [[nodiscard]] bool Stopped() const noexcept { return fds_.empty(); }

  template <typename E>
  void Join(E& e) noexcept {
    struct epoll_event event{};
    event.events = e.flags;
    event.data.ptr = reinterpret_cast<void*>(&e);
    int rc = epoll_ctl(fd_, EPOLL_CTL_ADD, e.fd, &event);
    ASSERT(rc >= 0);
    fds_.emplace(e.fd);
  }

  template <typename E>
  void Quit(E& e) noexcept {
    struct epoll_event event{};
    event.events = e.flags;
    epoll_ctl(fd_, EPOLL_CTL_DEL, e.fd, &event);
    fds_.erase(e.fd);
  }

  static constexpr int kMaxEvents = 64;
  int fd_ = -1;
  std::unordered_set<int> fds_;
  std::vector<struct epoll_event> events_;
};

using Selector = EpollSelector;

#elif defined(__APPLE__) || defined(__FreeBSD__)

constexpr int16_t kEventRead = EVFILT_READ;
constexpr int16_t kEventWrite = EVFILT_WRITE;
constexpr uint32_t kEventError = 4;

class KQueueSelector : private NoCopy {
 public:
  using ms = std::chrono::milliseconds;

  KQueueSelector();
  ~KQueueSelector();

  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500});
  [[nodiscard]] bool Stopped() const noexcept { return fds_.empty(); }

  template <typename E>
  void Join(E& e) noexcept {
    struct kevent kev{};
    EV_SET(&kev, e.fd, e.flags, EV_ADD | EV_ENABLE, 0, 0, &e);
    if (e.fd >= 0) {
      kevent(fd_, &kev, 1, nullptr, 0, nullptr);
      fds_.emplace(e.fd);
    }
  }

  template <typename E>
  void Quit(E& e) noexcept {
    struct kevent kev{};
    EV_SET(&kev, e.fd, e.flags, EV_DELETE | EV_DISABLE, 0, 0, nullptr);
    kevent(fd_, &kev, 1, nullptr, 0, nullptr);
    fds_.erase(e.fd);
  }

  static constexpr int kMaxEvents = 64;
  int fd_ = -1;
  std::unordered_set<int> fds_;
  std::vector<struct kevent> events_;
};

using Selector = KQueueSelector;

#endif
