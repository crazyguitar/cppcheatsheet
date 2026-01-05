#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{3, 1, 4, 1, 5, 9, 2, 6};

  std::sort(v.begin(), v.end());
  assert(v[0] == 1 && v[1] == 1);

  std::sort(v.begin(), v.end(), std::greater<>{});
  assert(v[0] == 9 && v[1] == 6);

  std::sort(v.begin(), v.end(), [](int a, int b) { return a < b; });
  assert(v[0] == 1);
}
