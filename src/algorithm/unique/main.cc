#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 1, 2, 2, 2, 3, 1, 1};
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
  assert((v == std::vector{1, 2, 3, 1}));

  std::vector v2{3, 1, 2, 1, 3, 2};
  std::sort(v2.begin(), v2.end());
  v2.erase(std::unique(v2.begin(), v2.end()), v2.end());
  assert((v2 == std::vector{1, 2, 3}));
}
