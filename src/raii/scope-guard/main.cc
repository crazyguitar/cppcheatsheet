#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>

// A scope guard runs a cleanup callable when it leaves scope — ad-hoc RAII
// without writing a dedicated class. Useful for cleanup whose type does not
// warrant its own RAII wrapper (rollback on early return, reference counts,
// temporary state restoration). C++26 standardises std::scope_exit /
// scope_fail / scope_success; until then a small template works fine.

template <typename F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F f) : f_(std::move(f)) {}
  ~ScopeGuard() {
    if (active_) f_();
  }

  void dismiss() noexcept { active_ = false; }

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

 private:
  F f_;
  bool active_ = true;
};

template <typename F>
ScopeGuard<F> make_scope_guard(F f) {
  return ScopeGuard<F>(std::move(f));
}

TEST(ScopeGuard, RunsCleanupOnScopeExit) {
  int cleanups = 0;
  {
    auto g = make_scope_guard([&] { ++cleanups; });
    EXPECT_EQ(cleanups, 0);
  }
  EXPECT_EQ(cleanups, 1);
}

TEST(ScopeGuard, DismissSkipsCleanup) {
  int cleanups = 0;
  {
    auto g = make_scope_guard([&] { ++cleanups; });
    g.dismiss();  // Commit: skip the rollback.
  }
  EXPECT_EQ(cleanups, 0);
}

TEST(ScopeGuard, UniquePtrWithCustomDeleterAsScopeGuard) {
  // unique_ptr with a custom deleter is a ready-made scope guard: the
  // "pointer" is a dummy and the deleter is the cleanup.
  int cleanups = 0;
  {
    auto deleter = [&](void*) { ++cleanups; };
    std::unique_ptr<void, decltype(deleter)> guard(reinterpret_cast<void*>(uintptr_t{1}), deleter);
  }
  EXPECT_EQ(cleanups, 1);
}
