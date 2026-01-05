#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{2, 4, 6, 8};

  assert(std::all_of(v.begin(), v.end(), [](int x) { return x % 2 == 0; }));
  assert(!std::any_of(v.begin(), v.end(), [](int x) { return x % 2 != 0; }));
  assert(std::none_of(v.begin(), v.end(), [](int x) { return x < 0; }));
}
