#include <gtest/gtest.h>

#include <coroutine>
#include <utility>
#include <vector>

template <typename T>
class Generator {
 public:
  struct promise_type {
    T current_value;
    Generator get_return_object() { return Generator{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(T value) {
      current_value = std::move(value);
      return {};
    }
    void return_void() {}
    void unhandled_exception() {}
  };

  explicit Generator(std::coroutine_handle<promise_type> h) : handle_(h) {}
  ~Generator() {
    if (handle_) handle_.destroy();
  }
  Generator(Generator&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}

  struct iterator {
    std::coroutine_handle<promise_type> handle;
    iterator& operator++() {
      handle.resume();
      return *this;
    }
    T& operator*() { return handle.promise().current_value; }
    bool operator==(std::default_sentinel_t) const { return handle.done(); }
  };

  iterator begin() {
    handle_.resume();
    return {handle_};
  }
  std::default_sentinel_t end() { return {}; }

 private:
  std::coroutine_handle<promise_type> handle_;
};

Generator<int> range(int start, int end) {
  for (int i = start; i < end; ++i) co_yield i;
}

Generator<uint64_t> fibonacci(int n) {
  uint64_t a = 0, b = 1;
  for (int i = 0; i < n; ++i) {
    co_yield a;
    auto tmp = a;
    a = b;
    b = tmp + b;
  }
}

TEST(GeneratorTest, Range) {
  std::vector<int> r;
  for (int x : range(1, 5)) r.push_back(x);
  EXPECT_EQ(r, (std::vector<int>{1, 2, 3, 4}));
}

TEST(GeneratorTest, Fibonacci) {
  std::vector<uint64_t> f;
  for (auto x : fibonacci(10)) f.push_back(x);
  EXPECT_EQ(f, (std::vector<uint64_t>{0, 1, 1, 2, 3, 5, 8, 13, 21, 34}));
}
