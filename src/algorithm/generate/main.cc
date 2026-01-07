#include <algorithm>
#include <cassert>
#include <vector>

int main() {
  std::vector<int> v(5);
  int n = 0;
  std::generate(v.begin(), v.end(), [&n] { return n++; });
  assert(v[0] == 0 && v[4] == 4);
}
