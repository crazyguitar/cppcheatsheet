#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

  std::partial_sort(v.begin(), v.begin() + 3, v.end());
  assert(v[0] == 0 && v[1] == 1 && v[2] == 2);
}
