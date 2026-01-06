/**
 * @file coro.h
 * @brief C++20 coroutine wrapper with async execution and scheduling support
 */
#pragma once
#include <coroutine>
#include <stdexcept>
#include <utility>

#include "common.h"
#include "handle.h"
#include "io.h"
#include "result.h"

/** @brief Tag type for one-way coroutines */
struct Oneway {};
/** @brief Global instance of oneway tag */
inline constexpr Oneway oneway;

/**
 * @brief Coroutine wrapper with async execution support
 * @tparam T Return value type (default: void)
 */
template <typename T = void>
struct [[nodiscard]] Coro : private NoCopy {
  struct promise_type;
  using coro = std::coroutine_handle<promise_type>;

  template <typename C>
  friend class Future;

  explicit Coro(coro h) noexcept : handle_{h} {}
  Coro(Coro&& c) noexcept : handle_(std::exchange(c.handle_, {})) {}
  ~Coro() { Destroy(); }

  /**
   * @brief Base awaiter for coroutine suspension and scheduling
   */
  struct awaiter_base {
    coro h;
    constexpr bool await_ready() const noexcept {
      if (h) [[likely]]
        return h.done();
      return true;
    }

    template <typename Promise>
    void await_suspend(std::coroutine_handle<Promise> coroutine) const noexcept {
      coroutine.promise().SetState(Handle::kSuspend);
      h.promise().next = &coroutine.promise();
      h.promise().schedule();
    }
  };

  auto operator co_await() const& noexcept {
    /**
     * @brief Awaiter for lvalue coroutine references
     * Returns result by reference
     */
    struct awaiter : awaiter_base {
      decltype(auto) await_resume() const {
        if (!awaiter_base::h) [[unlikely]]
          throw std::runtime_error("invalid coro handler");
        return awaiter_base::h.promise().result();
      }
    };
    return awaiter{handle_};
  }

  auto operator co_await() && noexcept {
    /**
     * @brief Awaiter for rvalue coroutine references
     * Takes ownership of the handle so the temporary Coro doesn't destroy it
     * Returns result by move
     */
    struct awaiter : awaiter_base {
      decltype(auto) await_resume() const {
        if (!awaiter_base::h) [[unlikely]]
          throw std::runtime_error("invalid coro handler");
        return std::move(awaiter_base::h.promise()).result();
      }
    };
    // Transfer ownership to the awaiter - the temporary Coro will have null handle
    return awaiter{std::exchange(handle_, {})};
  }

  /**
   * @brief Promise type for C++20 coroutines
   *
   * Implements the coroutine promise interface required by the C++ standard.
   * Inherits from Handle for scheduling and Result<T> for value storage.
   * Manages coroutine lifecycle, suspension points, and continuation chains.
   */
  struct promise_type : Handle, Result<T> {
    promise_type() = default;

    template <typename... Args>
    promise_type(Oneway, Args&&...) : oneway_{true} {}

    auto initial_suspend() noexcept {
      /**
       * @brief Awaiter for coroutine initialization
       * Controls whether coroutine starts immediately or suspends
       */
      struct init_awaiter {
        constexpr bool await_ready() const noexcept { return oneway_; }
        constexpr void await_suspend(std::coroutine_handle<>) const noexcept {}
        constexpr void await_resume() const noexcept {}
        const bool oneway_{false};
      };
      return init_awaiter{oneway_};
    }

    /**
     * @brief Awaiter for coroutine finalization
     * Handles continuation chain when coroutine completes
     */
    struct final_awaiter {
      constexpr bool await_ready() const noexcept { return false; }
      constexpr void await_resume() const noexcept {}

      template <typename Promise>
      constexpr void await_suspend(std::coroutine_handle<Promise> h) const noexcept {
        if (auto next = h.promise().next) [[likely]] {
          IO::Get().Call(*next);
        }
      }
    };

    auto final_suspend() noexcept { return final_awaiter{}; };

    Coro get_return_object() noexcept { return Coro{coro::from_promise(*this)}; }
    /**
     * @brief Execute the coroutine or handle task
     */
    void run() final { coro::from_promise(*this).resume(); }
    void stop() final {
      auto coro = coro::from_promise(*this);
      if (coro) [[likely]]
        coro.destroy();
    }

    const bool oneway_{false};
    Handle* next{nullptr};
  };  // promise_type
      //
  /**
   * @brief Check if coroutine handle is valid
   * @return True if handle is valid, false otherwise
   */
  [[nodiscard]] constexpr bool valid() const noexcept { return handle_ != nullptr; }
  /**
   * @brief Check if coroutine execution is complete
   * @return True if execution finished, false otherwise
   */
  [[nodiscard]] bool done() const noexcept { return handle_.done(); }

  decltype(auto) result() & { return handle_.promise().result(); }

  decltype(auto) result() && { return std::move(handle_.promise()).result(); }

 private:
  /**
   * @brief Clean up and destroy coroutine resources
   *
   * If the coroutine is currently scheduled in the IO loop, we just mark it
   * as cancelled and let the IO loop call stop() which will destroy it.
   * If it's not scheduled, we can safely destroy it immediately.
   */
  void Destroy() noexcept {
    if (auto handle = std::exchange(handle_, nullptr)) [[likely]] {
      auto& promise = handle.promise();
      if (promise.GetState() == Handle::kScheduled) [[unlikely]] {
        // Handle is in IO's queue - just cancel, don't destroy yet
        // The IO loop will call stop() which destroys the coroutine
        promise.cancel();
        // Don't call handle.destroy() - let stop() do it
      } else {
        // Handle is not scheduled - safe to destroy immediately
        handle.destroy();
      }
    }
  }

 private:
  coro handle_;
};  // Coro
