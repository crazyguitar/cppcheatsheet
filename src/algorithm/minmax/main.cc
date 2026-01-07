#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{3, 1, 4, 1, 5, 9, 2, 6};

  auto min_it = std::min_element(v.begin(), v.end());
  auto max_it = std::max_element(v.begin(), v.end());
  assert(*min_it == 1 && *max_it == 9);

  auto [lo, hi] = std::minmax_element(v.begin(), v.end());
  assert(*lo == 1 && *hi == 9);

  assert(std::clamp(10, 0, 5) == 5);
  assert(std::clamp(-1, 0, 5) == 0);
  assert(std::clamp(3, 0, 5) == 3);
}
