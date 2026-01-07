#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};
  std::vector<int> dest(5);

  std::copy(v.begin(), v.end(), dest.begin());
  assert(dest == v);

  std::vector<int> evens;
  std::copy_if(v.begin(), v.end(), std::back_inserter(evens), [](int x) { return x % 2 == 0; });
  assert(evens.size() == 2);
}
