#include <iostream>
#include <utility>

template <size_t N>
struct Loop {
  template <typename F>
  static void run(F&& f) {
    Loop<N - 1>::run(std::forward<F>(f));
    f(N - 1);
  }
};

template <>
struct Loop<0> {
  template <typename F>
  static void run(F&&) {}
};

int main() {
  size_t sum = 0;
  Loop<5>::run([&](auto i) { sum += i; });
  std::cout << sum << "\n";
}
