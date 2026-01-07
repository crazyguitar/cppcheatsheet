#include <cassert>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> v(5);
  std::iota(v.begin(), v.end(), 1);
  assert(v[0] == 1 && v[4] == 5);
}
