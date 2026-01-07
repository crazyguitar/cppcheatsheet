#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};
  std::reverse(v.begin(), v.end());
  assert((v == std::vector{5, 4, 3, 2, 1}));

  std::vector v2{1, 2, 3, 4, 5};
  std::rotate(v2.begin(), v2.begin() + 2, v2.end());
  assert((v2 == std::vector{3, 4, 5, 1, 2}));
}
