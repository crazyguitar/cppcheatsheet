#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};

  auto even = std::find_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
  assert(*even == 2);

  auto odd = std::find_if_not(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
  assert(*odd == 1);
}
