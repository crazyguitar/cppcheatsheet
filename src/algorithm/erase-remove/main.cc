#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 2, 4, 2, 5};
  v.erase(std::remove(v.begin(), v.end(), 2), v.end());
  assert(v.size() == 4);

  std::vector v2{1, 2, 3, 4, 5, 6};
  v2.erase(std::remove_if(v2.begin(), v2.end(), [](int x) { return x % 2 == 0; }), v2.end());
  assert(v2.size() == 3);
}
