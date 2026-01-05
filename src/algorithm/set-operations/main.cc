#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector a{1, 2, 3, 4, 5};
  std::vector b{3, 4, 5, 6, 7};

  std::vector<int> u;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(u));
  assert((u == std::vector{1, 2, 3, 4, 5, 6, 7}));

  std::vector<int> i;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(i));
  assert((i == std::vector{3, 4, 5}));

  std::vector<int> d;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(d));
  assert((d == std::vector{1, 2}));
}
