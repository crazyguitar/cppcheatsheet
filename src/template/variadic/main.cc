#include <iostream>
#include <utility>
#include <vector>

template <typename T>
T sum_recursive(T x) {
  return x;
}

template <typename T, typename... Args>
T sum_recursive(T x, Args... args) {
  return x + sum_recursive(args...);
}

template <typename... Args>
auto sum_fold(Args... args) {
  return (args + ...);
}

template <typename T>
class Vector {
  std::vector<T> v;

 public:
  template <typename... Args>
  Vector(Args&&... args) {
    (v.emplace_back(std::forward<Args>(args)), ...);
  }
  auto begin() { return v.begin(); }
  auto end() { return v.end(); }
};

int main() {
  std::cout << sum_recursive(1, 2, 3, 4, 5) << "\n";
  std::cout << sum_fold(1, 2, 3, 4, 5) << "\n";

  Vector<int> v{1, 2, 3};
  for (auto x : v) std::cout << x << "\n";
}
