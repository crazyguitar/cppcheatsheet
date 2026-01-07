#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 2, 3, 2, 4, 2};

  auto n = std::count(v.begin(), v.end(), 2);
  assert(n == 4);

  auto evens = std::count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
  assert(evens == 5);
}
