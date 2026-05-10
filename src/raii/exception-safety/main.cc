#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

// Four exception-safety levels:
//   nothrow: operation will never throw (marked noexcept).
//   strong:  operation either succeeds or leaves state unchanged (rollback).
//   basic:   invariants preserved; state may be partially updated.
//   none:    avoid — broken invariants possible.

class Counter {
 public:
  Counter() = default;
  explicit Counter(int v) : value_(v) {}

  // Nothrow: std::swap on int and vector is nothrow.
  void swap(Counter& other) noexcept {
    std::swap(value_, other.value_);
    std::swap(log_, other.log_);
  }

  // Strong guarantee via copy-and-swap: if the copy throws, *this unchanged;
  // the swap itself is nothrow.
  Counter& operator=(Counter other) noexcept {
    swap(other);
    return *this;
  }

  // Only basic guarantee: log_ may have grown before the throw, even though
  // value_ is unchanged.
  void add(int delta, bool fail = false) {
    log_.push_back(delta);
    if (fail) {
      throw std::runtime_error("simulated failure");
    }
    value_ += delta;
  }

  int value() const noexcept { return value_; }
  std::size_t log_size() const noexcept { return log_.size(); }

 private:
  int value_ = 0;
  std::vector<int> log_;
};

TEST(ExceptionSafety, NothrowSwap) {
  static_assert(noexcept(std::declval<Counter&>().swap(std::declval<Counter&>())));
  Counter a(1), b(2);
  a.swap(b);
  EXPECT_EQ(a.value(), 2);
  EXPECT_EQ(b.value(), 1);
}

TEST(ExceptionSafety, StrongGuaranteeViaCopyAndSwap) {
  Counter c(42);
  try {
    Counter tmp = c;   // copy
    tmp.add(1, true);  // throws: tmp dies unused, c untouched
    c = std::move(tmp);
  } catch (const std::runtime_error&) {
  }
  EXPECT_EQ(c.value(), 42);  // rolled back
}

TEST(ExceptionSafety, BasicGuaranteeKeepsInvariants) {
  Counter c(10);
  EXPECT_THROW(c.add(5, true), std::runtime_error);
  EXPECT_EQ(c.value(), 10);     // value_ unchanged
  EXPECT_EQ(c.log_size(), 1u);  // log_ grew — that's basic, not strong
}

TEST(ExceptionSafety, CopyAndSwapAssignmentReplacesState) {
  Counter c(1);
  c.add(10);
  c.add(20);
  EXPECT_EQ(c.value(), 31);

  Counter replacement(100);
  c = replacement;                      // copy-and-swap via operator=(Counter)
  EXPECT_EQ(c.value(), 100);            // new state committed
  EXPECT_EQ(c.log_size(), 0u);          // old log swapped out
  EXPECT_EQ(replacement.value(), 100);  // source unchanged by copy
}
