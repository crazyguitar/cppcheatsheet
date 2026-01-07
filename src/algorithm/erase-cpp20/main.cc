#include <cassert>
#include <vector>

int main() {
  std::vector v{1, 2, 3, 2, 4, 2, 5};
  std::erase(v, 2);
  assert(v.size() == 4);

  std::vector v2{1, 2, 3, 4, 5, 6};
  std::erase_if(v2, [](int x) { return x % 2 == 0; });
  assert(v2.size() == 3);
}
