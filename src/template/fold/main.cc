#include <iostream>
#include <vector>

int main() {
  auto sum = [](auto... args) { return (args + ...); };
  std::cout << sum(1, 2, 3, 4, 5) << "\n";

  std::vector<int> v;
  [&v](auto... args) { (v.emplace_back(args), ...); }(1, 2, 3);

  [](auto... args) { (std::cout << ... << args) << "\n"; }(1, 2, 3, 4, 5);

  auto double_sum = [](auto&& f, auto... args) { return (... + f(args)); };
  std::cout << double_sum([](auto x) { return x * 2; }, 1, 2, 3) << "\n";
}
