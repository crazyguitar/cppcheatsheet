/**
 * @file handle.h
 * @brief Base class for asynchronous task handles with state management
 */
#pragma once
#include <atomic>
#include <cstdint>

/**
 * @brief Base class for asynchronous task handles with state management
 */
struct Handle {
  /** @brief Handle execution states */
  enum State : uint8_t { kUnschedule, kScheduled, kSuspend };

  Handle() : id_{seq_++} {}
  virtual ~Handle() = default;

  /**
   * @brief Execute the handle's task
   */
  virtual void run() = 0;

  virtual void stop() = 0;

  /**
   * @brief Set handle execution state
   * @param state New state to set
   */
  void SetState(State state) noexcept { state_ = state; }

  /**
   * @brief Get current execution state
   * @return Current state
   */
  [[nodiscard]] State GetState() const noexcept { return state_; }

  /**
   * @brief Get unique handle identifier
   * @return Handle ID
   */
  [[nodiscard]] uint64_t GetId() const noexcept { return id_; }

  /**
   * @brief Schedule handle for execution
   */
  void schedule() noexcept;

  /**
   * @brief Cancel handle execution
   */
  void cancel() noexcept;

 private:
  static inline std::atomic<uint64_t> seq_{0};
  uint64_t id_;
  State state_ = Handle::kUnschedule;
};
