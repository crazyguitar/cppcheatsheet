#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

// Track which overload was called
enum class ValueCategory { Lvalue, Rvalue };

ValueCategory process(int&) { return ValueCategory::Lvalue; }
ValueCategory process(int&&) { return ValueCategory::Rvalue; }

// Without forwarding: always calls lvalue overload
template <typename T>
ValueCategory wrapper_bad(T&& arg) {
  return process(arg);  // arg is always lvalue (it has a name)
}

// With forwarding: preserves original value category
template <typename T>
ValueCategory wrapper_good(T&& arg) {
  return process(std::forward<T>(arg));
}

TEST(PerfectForwarding, WithoutForwardAlwaysLvalue) {
  int x = 42;

  // Both call lvalue overload because 'arg' has a name
  EXPECT_EQ(wrapper_bad(x), ValueCategory::Lvalue);
  EXPECT_EQ(wrapper_bad(42), ValueCategory::Lvalue);  // Lost rvalue-ness!
}

TEST(PerfectForwarding, WithForwardPreservesCategory) {
  int x = 42;

  EXPECT_EQ(wrapper_good(x), ValueCategory::Lvalue);   // x is lvalue
  EXPECT_EQ(wrapper_good(42), ValueCategory::Rvalue);  // 42 is rvalue
}

// Factory pattern with perfect forwarding
struct Widget {
  int x;
  double y;
  std::string name;

  Widget(int x, double y, std::string name) : x(x), y(y), name(std::move(name)) {}
};

template <typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

TEST(PerfectForwarding, FactoryPattern) {
  auto w = make<Widget>(42, 3.14, "test");

  EXPECT_EQ(w->x, 42);
  EXPECT_DOUBLE_EQ(w->y, 3.14);
  EXPECT_EQ(w->name, "test");
}

// Variadic forwarding
template <typename Func, typename... Args>
auto invoke_and_return(Func&& f, Args&&... args) {
  return std::forward<Func>(f)(std::forward<Args>(args)...);
}

TEST(PerfectForwarding, VariadicForwarding) {
  auto add = [](int a, int b) { return a + b; };
  auto result = invoke_and_return(add, 2, 3);
  EXPECT_EQ(result, 5);

  auto concat = [](std::string a, std::string b) { return a + b; };
  auto str = invoke_and_return(concat, std::string("hello"), std::string(" world"));
  EXPECT_EQ(str, "hello world");
}
