#include <gtest/gtest.h>

#include <vector>

namespace ctor_dispatch {

struct B {
  std::vector<int>* log;
  explicit B(std::vector<int>* l) : log(l) { f(); }
  virtual ~B() { f(); }
  virtual void f() { log->push_back(0); }
};

struct D : B {
  explicit D(std::vector<int>* l) : B(l) {}
  void f() override { log->push_back(1); }
};

}  // namespace ctor_dispatch

TEST(CtorDtorDispatch, ConstructorCallsBaseVersion) {
  std::vector<int> log;
  { ctor_dispatch::D d{&log}; }
  // ctor logged 0 (B::f), then dtor logged 0 (B::f again).
  // D::f never runs from inside B's ctor or dtor.
  EXPECT_EQ(log, (std::vector<int>{0, 0}));
}

TEST(CtorDtorDispatch, FullyConstructedObjectDispatchesToDerived) {
  std::vector<int> log;
  ctor_dispatch::D d{&log};
  log.clear();
  ctor_dispatch::B& b = d;
  b.f();
  EXPECT_EQ(log, (std::vector<int>{1}));
}
