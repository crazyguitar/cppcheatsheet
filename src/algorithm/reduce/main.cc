#include <cassert>
#include <numeric>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};
  int sum = std::reduce(v.begin(), v.end(), 0);
  assert(sum == 15);
}
