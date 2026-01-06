/**
 * @file result.h
 * @brief Result type for coroutine promise values with exception handling
 */
#pragma once

#include <exception>
#include <optional>
#include <stdexcept>
#include <variant>

/**
 * @brief Result type for coroutine promise values with exception handling
 * @tparam T Value type to store
 */
template <typename T>
struct Result {
  /**
   * @brief Check if result has a value
   * @return true if value is set, false otherwise
   */
  constexpr bool has_value() const noexcept { return std::get_if<std::monostate>(&result_) == nullptr; }

  /**
   * @brief Set the result value
   * @param value Value to store
   */
  template <typename R>
  constexpr void set_value(R&& value) noexcept {
    result_.template emplace<T>(std::forward<R>(value));
  }

  /**
   * @brief Set return value for coroutine promise
   * @param value Value to return
   */
  template <typename R>
  constexpr void return_value(R&& value) noexcept {
    return set_value(std::forward<R>(value));
  }

  /**
   * @brief Get the result value (lvalue reference)
   * @return The stored value
   * @throws std::exception_ptr if exception was set
   * @throws std::runtime_error if no value was set
   */
  constexpr T result() & {
    if (auto exception = std::get_if<std::exception_ptr>(&result_)) {
      std::rethrow_exception(*exception);
    }
    if (auto res = std::get_if<T>(&result_)) {
      return *res;
    }
    throw std::runtime_error("result not set");
  }

  /**
   * @brief Get the result value (rvalue reference)
   * @return The stored value
   * @throws std::exception_ptr if exception was set
   * @throws std::runtime_error if no value was set
   */
  constexpr T result() && {
    if (auto exception = std::get_if<std::exception_ptr>(&result_)) {
      std::rethrow_exception(*exception);
    }
    if (auto res = std::get_if<T>(&result_)) {
      return std::move(*res);
    }
    throw std::runtime_error("result not set");
  }

  /**
   * @brief Set exception for the result
   * @param exception Exception pointer to store
   */
  void set_exception(std::exception_ptr exception) noexcept { result_ = exception; }

  /**
   * @brief Handle unhandled exception in coroutine
   */
  void unhandled_exception() noexcept { result_ = std::current_exception(); }

 private:
  std::variant<std::monostate, T, std::exception_ptr> result_;
};

/**
 * @brief Result specialization for void return type
 */
template <>
struct Result<void> {
  /**
   * @brief Check if result has been set
   * @return true if void result was set
   */
  constexpr bool has_value() const noexcept { return result_.has_value(); }

  /**
   * @brief Set void return for coroutine promise
   */
  void return_void() noexcept { result_.emplace(nullptr); }

  /**
   * @brief Get the void result
   * @throws std::exception_ptr if exception was set
   */
  void result() {
    if (result_.has_value() && *result_ != nullptr) {
      std::rethrow_exception(*result_);
    }
  }

  /**
   * @brief Set exception for the result
   * @param exception Exception pointer to store
   */
  void set_exception(std::exception_ptr exception) noexcept { result_ = exception; }

  /**
   * @brief Handle unhandled exception in coroutine
   */
  void unhandled_exception() noexcept { result_ = std::current_exception(); }

 private:
  std::optional<std::exception_ptr> result_;
};
