#include <gtest/gtest.h>

// Backend traits with compile-time feature flags
struct BackendA {
  static constexpr bool kSupportFoo = true;
  static constexpr bool kSupportBar = true;
};

struct BackendB {
  static constexpr bool kSupportFoo = false;
  static constexpr bool kSupportBar = false;
};

template <typename Backend>
class Channel {
 public:
  // Only available when Backend::kSupportFoo is true
  int Foo(int x)
    requires Backend::kSupportFoo
  {
    return x;
  }

  // Only available when Backend::kSupportBar is true
  int Bar()
    requires Backend::kSupportBar
  {
    return 42;
  }

  // Always available
  int Baz(int x) { return x * 2; }
};

TEST(RequiresBoolFlag, BackendA) {
  Channel<BackendA> ch;
  EXPECT_EQ(ch.Foo(10), 10);  // OK: kSupportFoo is true
  EXPECT_EQ(ch.Bar(), 42);    // OK: kSupportBar is true
  EXPECT_EQ(ch.Baz(5), 10);   // OK: always available
}

TEST(RequiresBoolFlag, BackendB) {
  Channel<BackendB> ch;
  // ch.Foo(10);  // Error: kSupportFoo is false
  // ch.Bar();    // Error: kSupportBar is false
  EXPECT_EQ(ch.Baz(5), 10);  // OK: always available
}
