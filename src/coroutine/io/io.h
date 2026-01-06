/**
 * @file io.h
 * @brief Asynchronous I/O event loop with task scheduling and cancellation
 */
#pragma once
#include <chrono>
#include <deque>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "handle.h"
#include "selector.h"

/**
 * @brief Asynchronous I/O event loop with task scheduling
 *
 * The IO class provides an event loop for scheduling and executing asynchronous
 * tasks. It supports:
 * - Immediate task scheduling via Call(handle)
 * - Delayed task scheduling via Call(delay, handle)
 * - Task cancellation via Cancel(handle)
 * - Event-driven I/O via Join/Quit and the internal Selector
 *
 * Cancellation semantics:
 * - Cancel(handle) marks a handle as cancelled and unscheduled
 * - Cancelled handles will not have run() called
 * - Cancelled handles will have stop() called exactly once
 * - Calling Cancel() multiple times is safe (idempotent)
 * - Cancel() on an already-executed handle is a no-op
 */
class IO : private NoCopy {
 public:
  using milliseconds = std::chrono::milliseconds;
  using task_type = std::tuple<milliseconds, uint64_t, Handle*>;
  using priority_queue = std::priority_queue<task_type, std::vector<task_type>, std::greater<task_type> >;

  /// Default timeout for I/O polling when no tasks are ready (500ms)
  static constexpr milliseconds kDefaultPollTimeout{500};
  /// Non-blocking poll timeout when tasks are ready for execution
  static constexpr milliseconds kNonBlockingPollTimeout{0};

  IO() : start_{std::chrono::system_clock::now()} {}

  /**
   * @brief Get singleton IO instance
   * @return Reference to the thread-local IO singleton
   */
  [[nodiscard]] static IO& Get() {
    thread_local IO io;
    return io;
  }

  /**
   * @brief Get current time elapsed since IO instance creation
   * @return Time in milliseconds since start
   */
  [[nodiscard]] milliseconds Time() const noexcept {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<milliseconds>(now - start_);
  }

  /**
   * @brief Cancel a scheduled handle
   * @param handle Handle to cancel
   *
   * The handle will be marked as unscheduled and its ID added to the
   * cancelled set. Any pending execution of this handle will be skipped
   * and stop() will be called instead.
   *
   * This method is idempotent - calling it multiple times on the same
   * handle is safe and will not cause double-stopping.
   *
   * If the handle has already been executed (state is kUnschedule and
   * not in any queue), this is a no-op.
   */
  void Cancel(Handle& handle) noexcept {
    // Only cancel if the handle is currently scheduled
    if (handle.GetState() == Handle::kScheduled) [[likely]] {
      handle.SetState(Handle::kUnschedule);
      cancelled_.insert(handle.GetId());
    }
  }

  /**
   * @brief Check if a handle is pending cancellation
   * @param handle Handle to check
   * @return true if handle is in the cancelled set awaiting stop() processing
   *
   * Note: Returns true only while the handle is in the cancelled set.
   * After stop() is called during cancellation processing, this returns false.
   */
  [[nodiscard]] bool IsCancelled(const Handle& handle) const noexcept { return IsHandleCancelled(handle.GetId()); }

  /**
   * @brief Schedule handle for immediate execution
   * @param handle Handle to execute
   */
  void Call(Handle& handle) noexcept {
    if (handle.GetState() == Handle::kScheduled) return;  // Already scheduled
    handle.SetState(Handle::kScheduled);
    ready_.emplace_back(std::addressof(handle));
  }

  /**
   * @brief Schedule handle for delayed execution
   * @param delay Time delay before execution
   * @param handle Handle to execute
   */
  template <typename Rep, typename Period>
  void Call(std::chrono::duration<Rep, Period> delay, Handle& handle) {
    handle.SetState(Handle::kScheduled);
    const auto when = Time() + std::chrono::duration_cast<milliseconds>(delay);
    schedule_.push(task_type{when, handle.GetId(), std::addressof(handle)});
  }

  /**
   * @brief Run the event loop until stopped
   *
   * Continuously processes scheduled tasks and I/O events until the loop
   * is stopped (no pending tasks or events). Uses adaptive polling:
   * non-blocking poll when tasks are ready, or waits up to 500ms otherwise.
   */
  void Run() {
    while (!Stopped()) {
      Runone();
      Cancel();
      // Always poll for I/O events, but use zero timeout if there's ready work
      // to avoid blocking. This prevents I/O starvation when many coroutines
      // are scheduled while still processing ready work promptly.
      Select(ready_.empty() ? kDefaultPollTimeout : kNonBlockingPollTimeout);
    }
  }

  /**
   * @brief Poll for I/O events and schedule ready handles
   * @param timeout Maximum time to wait for events (default: 500ms)
   *
   * Non-blocking if timeout is zero. Schedules all handles associated
   * with events that occurred during the poll.
   */
  void Select(milliseconds timeout = kDefaultPollTimeout) {
    const auto events = selector_.Select(timeout);
    for (const auto& e : events) {
      if (auto* h = e.handle) [[likely]] {
        Call(*h);
      }
    }
  }

  /**
   * @brief Execute one iteration of scheduled tasks
   *
   * Processes all tasks that are due for execution:
   * 1. Moves due tasks from the schedule queue to the ready queue
   * 2. Executes all ready tasks, skipping cancelled ones
   * 3. Cancelled handles are moved to the stopped queue for cleanup
   */
  void Runone() {
    const auto now = Time();

    // Move due tasks from schedule to ready or stopped queue
    while (!schedule_.empty()) {
      const auto& task = schedule_.top();
      const auto& when = std::get<0>(task);
      const auto id = std::get<1>(task);
      auto* handle = std::get<2>(task);

      if (when > now) break;

      // Skip cancelled handles - move them to stopped queue for cleanup
      if (!IsHandleCancelled(id)) [[likely]] {
        ready_.emplace_back(handle);
      } else {
        stopped_.emplace_back(handle);
        cancelled_.erase(id);
      }

      schedule_.pop();
    }

    // Process ready queue, checking for cancellation
    const size_t ready_count = ready_.size();
    for (size_t i = 0; i < ready_count; ++i) {
      auto* handle = ready_.front();
      ready_.pop_front();

      if (!handle) [[unlikely]]
        continue;

      const auto id = handle->GetId();
      // Check if this handle was cancelled while in the ready queue
      if (IsHandleCancelled(id)) [[unlikely]] {
        stopped_.emplace_back(handle);
        cancelled_.erase(id);
        continue;
      }

      handle->SetState(Handle::kUnschedule);
      handle->run();
    }
  }

  /**
   * @brief Check if event loop should stop
   * @return true if no pending tasks, ready tasks, or I/O events
   */
  [[nodiscard]] bool Stopped() const noexcept { return schedule_.empty() && ready_.empty() && selector_.Stopped(); }

  /**
   * @brief Get the selector
   * @return Reference to the selector
   */
  [[nodiscard]] Selector& GetSelector() noexcept { return selector_; }

  /**
   * @brief Register an event source with the selector
   * @tparam P Event source type
   * @param p Event source to monitor
   */
  template <typename P>
  void Join(P& p) {
    selector_.Join(p);
  }

  /**
   * @brief Unregister an event source from the selector
   * @tparam P Event source type
   * @param p Event source to stop monitoring
   */
  template <typename P>
  void Quit(P& p) {
    selector_.Quit(p);
  }

  /**
   * @brief Process cancelled handles by calling their stop() method
   *
   * Invokes stop() on all handles in the stopped queue. This is called
   * during each event loop iteration to ensure cancelled handles are
   * cleaned up. Each handle's stop() is called exactly once.
   */
  void Cancel() {
    if (stopped_.empty()) [[likely]]
      return;

    const size_t stopped_count = stopped_.size();
    for (size_t i = 0; i < stopped_count; ++i) {
      auto* handle = stopped_.front();
      stopped_.pop_front();

      if (!handle) [[unlikely]]
        continue;

      handle->SetState(Handle::kUnschedule);
      handle->stop();
    }
  }

 private:
  /**
   * @brief Helper to check if a handle ID is in the cancelled set
   * @param id Handle ID to check
   * @return true if the handle is pending cancellation
   */
  [[nodiscard]] bool IsHandleCancelled(uint64_t id) const noexcept { return cancelled_.find(id) != cancelled_.end(); }
  std::chrono::time_point<std::chrono::system_clock> start_;
  Selector selector_;
  priority_queue schedule_;
  std::deque<Handle*> ready_;
  std::deque<Handle*> stopped_;
  std::unordered_set<uint64_t> cancelled_;  ///< Set of cancelled handle IDs
};
