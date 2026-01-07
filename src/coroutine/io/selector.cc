/**
 * @file selector.cc
 * @brief Cross-platform I/O event multiplexing implementation
 */

#include "selector.h"

#include <unistd.h>

#ifdef __linux__

EpollSelector::EpollSelector() : fd_(epoll_create1(EPOLL_CLOEXEC)), events_(kMaxEvents) { ASSERT(fd_ >= 0); }

EpollSelector::~EpollSelector() {
  if (fd_ >= 0) ::close(fd_);
}

std::vector<Event> EpollSelector::Select(ms duration) {
  std::vector<Event> result;
  if (Stopped()) return result;
  int n = epoll_wait(fd_, events_.data(), kMaxEvents, duration.count());
  for (int i = 0; i < n; ++i) {
    auto* e = reinterpret_cast<Event*>(events_[i].data.ptr);
    if (e && e->handle) {
      uint32_t flags = events_[i].events;
      result.emplace_back(e->fd, flags, e->handle);
    }
  }
  return result;
}

#elif defined(__APPLE__) || defined(__FreeBSD__)

KQueueSelector::KQueueSelector() : fd_(kqueue()), events_(kMaxEvents) { ASSERT(fd_ >= 0); }

KQueueSelector::~KQueueSelector() {
  if (fd_ >= 0) ::close(fd_);
}

std::vector<Event> KQueueSelector::Select(ms duration) {
  std::vector<Event> result;
  if (Stopped()) return result;
  struct timespec ts{};
  ts.tv_sec = duration.count() / 1000;
  ts.tv_nsec = (duration.count() % 1000) * 1000000;
  int n = kevent(fd_, nullptr, 0, events_.data(), kMaxEvents, &ts);
  if (n < 0) return result;
  for (int i = 0; i < n; ++i) {
    auto* e = reinterpret_cast<Event*>(events_[i].udata);
    if (e && e->handle) {
      result.emplace_back(e->fd, events_[i].filter, e->handle);
    }
  }
  return result;
}

#endif
