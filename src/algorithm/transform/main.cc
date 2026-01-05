#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

int main() {
  std::string s = "hello";
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  assert(s == "HELLO");

  std::vector<int> v{1, 2, 3};
  std::vector<int> sq(v.size());
  std::transform(v.begin(), v.end(), sq.begin(), [](int x) { return x * x; });
  assert(sq[0] == 1 && sq[1] == 4 && sq[2] == 9);
}
