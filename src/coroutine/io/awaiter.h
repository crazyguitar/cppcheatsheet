/**
 * @file awaiter.h
 * @brief Coroutine awaiter for yielding execution to other tasks
 */
#pragma once
#include <exception>

#include "coro.h"
#include "event.h"
#include "io.h"

/**
 * @brief Yield awaiter that reschedules the coroutine to run later.
 *
 * This allows other coroutines to run before resuming. Useful when
 * a coroutine is doing busy work (like retrying writes on EAGAIN)
 * and needs to give other coroutines a chance to make progress.
 */
struct YieldAwaiter {
  constexpr bool await_ready() const noexcept { return false; }
  constexpr void await_resume() const noexcept {}

  template <typename Promise>
  void await_suspend(std::coroutine_handle<Promise> coroutine) const noexcept {
    // Reschedule this coroutine to run later
    IO::Get().Call(coroutine.promise());
  }
};
