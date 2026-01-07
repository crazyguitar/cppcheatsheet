#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};

  auto it = std::find(v.begin(), v.end(), 3);
  assert(it != v.end() && *it == 3);

  auto not_found = std::find(v.begin(), v.end(), 99);
  assert(not_found == v.end());
}
