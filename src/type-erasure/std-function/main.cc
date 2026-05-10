#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace stdfn {

int free_add(int a, int b) { return a + b; }

struct AddCallable {
  int base;
  int operator()(int a, int b) const { return base + a + b; }
};

}  // namespace stdfn

TEST(StdFunction, ErasesAnyCallableMatchingSignature) {
  std::vector<std::function<int(int, int)>> fs;
  fs.emplace_back(stdfn::free_add);
  fs.emplace_back([](int a, int b) { return a * b; });
  fs.emplace_back(stdfn::AddCallable{100});
  EXPECT_EQ(fs[0](2, 3), 5);
  EXPECT_EQ(fs[1](2, 3), 6);
  EXPECT_EQ(fs[2](2, 3), 105);
}

TEST(StdFunction, EmptyFunctionIsFalsey) {
  std::function<void()> f;
  EXPECT_FALSE(static_cast<bool>(f));
  f = [] {};
  EXPECT_TRUE(static_cast<bool>(f));
}

TEST(StdFunction, CopiesTheStoredCallable) {
  // std::function requires its target to be CopyConstructible. unique_ptr
  // captured by value is not, so the following would fail to compile:
  //
  //   std::function<int()> f = [p = std::make_unique<int>(7)] { return *p; };
  //
  // Workaround: shared_ptr capture, or use std::move_only_function (C++23).
  auto p = std::make_shared<int>(7);
  std::function<int()> f = [p] { return *p; };
  std::function<int()> g = f;  // copy
  EXPECT_EQ(f(), 7);
  EXPECT_EQ(g(), 7);
}
