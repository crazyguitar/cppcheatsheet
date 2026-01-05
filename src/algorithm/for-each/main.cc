#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3};
  std::for_each(v.begin(), v.end(), [](int& x) { x *= 2; });
  assert(v[0] == 2 && v[1] == 4 && v[2] == 6);
}
