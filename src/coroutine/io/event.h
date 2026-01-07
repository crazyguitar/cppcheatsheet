/**
 * @file event.h
 * @brief Event descriptor for I/O multiplexing
 */
#pragma once

#include <cstdint>
#include <utility>

#include "handle.h"

#if defined(__APPLE__) || defined(__FreeBSD__)
using Flags_t = int16_t;
#else
using Flags_t = uint32_t;
#endif

struct Event {
  int fd = -1;
  Flags_t flags = 0;
  Handle* handle = nullptr;

  Event() = default;
  Event(int f, Handle* h = nullptr) : fd(f), handle(h) {}
  Event(int f, Flags_t fl, Handle* h) : fd(f), flags(fl), handle(h) {}

  Event(Event&& other) noexcept
      : fd(std::exchange(other.fd, -1)), flags(std::exchange(other.flags, Flags_t{0})), handle(std::exchange(other.handle, nullptr)) {}

  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      fd = std::exchange(other.fd, -1);
      flags = std::exchange(other.flags, Flags_t{0});
      handle = std::exchange(other.handle, nullptr);
    }
    return *this;
  }
};
