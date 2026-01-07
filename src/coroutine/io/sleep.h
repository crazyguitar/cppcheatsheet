/**
 * @file sleep.h
 * @brief Coroutine sleep utilities for delayed execution
 */
#pragma once

#include <chrono>

#include "common.h"
#include "coro.h"
#include "io.h"

namespace detail {

/**
 * @brief Awaiter for coroutine sleep operations
 * @tparam Duration Duration type for sleep delay
 */
template <typename Duration>
class sleep_awaiter : private NoCopy {
 public:
  sleep_awaiter(Duration delay) : delay_{delay} {}
  constexpr bool await_ready() noexcept { return false; }
  constexpr void await_resume() const noexcept {}

  template <typename Promise>
  void await_suspend(std::coroutine_handle<Promise> coroutine) const noexcept {
    IO::Get().Call(delay_, coroutine.promise());
  }

 private:
  Duration delay_;
};

/**
 * @brief Internal sleep implementation
 * @param delay Duration to sleep
 * @return Coroutine that suspends for the specified duration
 */
template <typename Rep, typename Period>
Coro<> Sleep(Oneway, std::chrono::duration<Rep, Period> delay) {
  co_await detail::sleep_awaiter{delay};
}
}  // namespace detail

/**
 * @brief Sleep for specified duration in a coroutine
 * @param delay Duration to sleep
 * @return Coroutine that suspends for the specified duration
 */
template <typename Rep, typename Period>
Coro<> Sleep(std::chrono::duration<Rep, Period> delay) {
  return detail::Sleep(oneway, delay);
}
