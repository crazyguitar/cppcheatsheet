#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector v{5, 6, 4, 3, 2, 6, 7, 9, 3};

  auto mid = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), mid, v.end());

  // all elements before mid are <= *mid
  for (auto it = v.begin(); it != mid; ++it) {
    assert(*it <= *mid);
  }
}
