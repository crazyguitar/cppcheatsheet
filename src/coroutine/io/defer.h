/**
 * @file defer.h
 * @brief RAII-based deferred execution utility
 */
#pragma once
#include "common.h"

/**
 * @brief Macro for creating deferred execution blocks
 * @details Usage: defer { cleanup_code(); };
 */
#define defer Defer defer_obj = [&]()

/**
 * @brief RAII wrapper for deferred execution
 * @tparam F Callable type (typically lambda)
 */
template <typename F>
class Defer : private NoCopy {
 public:
  Defer() = delete;
  Defer(Defer&&) = delete;
  Defer& operator=(Defer&&) = delete;
  /**
   * @brief Construct defer object with callable
   * @param fn Function to execute on destruction
   */
  Defer(F fn) : fn_{std::move(fn)}, invoke_{true} {}
  /**
   * @brief Execute deferred function on destruction
   */
  ~Defer() {
    if (invoke_) fn_();
  }

 private:
  F fn_;
  bool invoke_;
};
