#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

// When a constructor throws, the object's destructor does NOT run, but
// destructors for fully-constructed subobjects (members and bases) DO run.
// Raw pointer members therefore leak if the constructor throws after the
// raw allocation but before handing ownership to the object. Prefer RAII
// members (smart pointers, std::vector, std::string) so cleanup is
// automatic.

// Each test snapshots this counter at the start and asserts a delta, so
// tests are order-independent. The LeakyResource test deliberately leaks
// one Handle for the rest of the process lifetime — subsequent tests
// simply use the new baseline.
static int live_handles = 0;

class Handle {
 public:
  explicit Handle(std::string label) : label_(std::move(label)) { ++live_handles; }
  ~Handle() { --live_handles; }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

 private:
  std::string label_;
};

// Bad: raw pointer member leaks if the constructor throws after new.
class LeakyResource {
 public:
  LeakyResource() {
    first_ = new Handle("first");
    // Simulate a failure acquiring the second resource.
    throw std::runtime_error("second resource failed");
    // Never reached: ~LeakyResource does not run, first_ leaks.
  }
  ~LeakyResource() { delete first_; }

 private:
  Handle* first_ = nullptr;
};

TEST(ConstructorFailure, RawPointerMemberLeaksOnThrow) {
  const int before = live_handles;
  EXPECT_THROW({ LeakyResource r; }, std::runtime_error);
  // live_handles grew by one and stays grown: the Handle leaked.
  EXPECT_EQ(live_handles, before + 1);
}

// Good: unique_ptr member is a fully-constructed subobject whose destructor
// runs automatically when the enclosing constructor throws.
class SafeResource {
 public:
  SafeResource() : first_(std::make_unique<Handle>("first")) {
    // first_ is fully constructed; its destructor runs if we throw now.
    throw std::runtime_error("second resource failed");
  }

 private:
  std::unique_ptr<Handle> first_;
};

TEST(ConstructorFailure, SmartPointerMemberCleansUpOnThrow) {
  const int before = live_handles;
  EXPECT_THROW({ SafeResource r; }, std::runtime_error);
  EXPECT_EQ(live_handles, before);  // No leak.
}

// function-try-block: catch exceptions from member-initializer-list. The
// object is still considered not constructed, so the catch block must
// rethrow or terminate — it cannot swallow the exception.
class Reported {
 public:
  Reported() try : handle_(std::make_unique<Handle>("reported")) { throw std::runtime_error("post-init failure"); } catch (const std::exception&) {
    // Members are already destroyed at this point; we cannot access handle_.
    // The exception is implicitly rethrown at the end of this block.
  }

 private:
  std::unique_ptr<Handle> handle_;
};

TEST(ConstructorFailure, FunctionTryBlockRethrows) {
  const int before = live_handles;
  EXPECT_THROW({ Reported r; }, std::runtime_error);
  EXPECT_EQ(live_handles, before);
}
