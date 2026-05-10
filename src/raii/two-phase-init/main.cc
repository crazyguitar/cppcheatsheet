#include <gtest/gtest.h>

#include <stdexcept>

// Two-phase initialization is the anti-pattern RAII replaces: the object
// exists in a "zombie" state between construction and a separate init()
// call, so every method must either check an `initialized_` flag or risk
// undefined behavior. RAII makes construction all-or-nothing: either the
// constructor returns a fully valid object, or it throws and the object
// never existed.

class TwoPhase {
 public:
  TwoPhase() = default;

  bool init(int v) {
    if (v < 0) return false;
    value_ = v;
    initialized_ = true;
    return true;
  }

  int value() const {
    if (!initialized_) throw std::logic_error("not initialized");
    return value_;
  }

 private:
  int value_ = 0;
  bool initialized_ = false;
};

class Raii {
 public:
  explicit Raii(int v) : value_(v) {
    if (v < 0) throw std::invalid_argument("v must be non-negative");
  }
  int value() const noexcept { return value_; }

 private:
  int value_;
};

TEST(TwoPhaseInit, PartlyConstructedObjectIsObservable) {
  TwoPhase t;  // constructed but unusable
  EXPECT_THROW((void)t.value(), std::logic_error);
  EXPECT_TRUE(t.init(42));
  EXPECT_EQ(t.value(), 42);
}

TEST(TwoPhaseInit, TwoPhaseAllowsInvalidInitInput) {
  TwoPhase t;
  EXPECT_FALSE(t.init(-1));  // silent failure: t remains uninitialized
  EXPECT_THROW((void)t.value(), std::logic_error);
}

TEST(TwoPhaseInit, RaiiConstructionIsAllOrNothing) {
  EXPECT_THROW(Raii(-1), std::invalid_argument);
  Raii r(42);  // fully constructed or exception — never zombie
  EXPECT_EQ(r.value(), 42);
}
