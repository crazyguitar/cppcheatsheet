#include <gtest/gtest.h>

#include <sstream>

struct Foo {
  static std::ostringstream log;

  Foo() { log << "C"; }
  ~Foo() { log << "D"; }
  Foo(const Foo&) { log << "c"; }
  Foo(Foo&&) noexcept { log << "m"; }

  static void reset() { log.str(""); }
};

std::ostringstream Foo::log;

TEST(Instrumented, LogsOperations) {
  Foo::reset();
  {
    Foo f;
  }
  EXPECT_EQ(Foo::log.str(), "CD");
}
