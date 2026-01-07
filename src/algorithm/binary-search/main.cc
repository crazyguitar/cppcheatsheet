#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};

  assert(std::binary_search(v.begin(), v.end(), 3));
  assert(!std::binary_search(v.begin(), v.end(), 99));

  auto lb = std::lower_bound(v.begin(), v.end(), 3);
  assert(*lb == 3);

  auto ub = std::upper_bound(v.begin(), v.end(), 3);
  assert(*ub == 4);

  auto [lo, hi] = std::equal_range(v.begin(), v.end(), 3);
  assert(*lo == 3 && *hi == 4);
}
