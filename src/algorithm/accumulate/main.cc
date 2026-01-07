#include <cassert>
#include <functional>
#include <numeric>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 4, 5};

  int sum = std::accumulate(v.begin(), v.end(), 0);
  assert(sum == 15);

  int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>{});
  assert(product == 120);
}
