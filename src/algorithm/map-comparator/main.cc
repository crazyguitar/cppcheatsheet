#include <cassert>
#include <map>

int main() {
  std::map<int, int, std::less<>> asc{{3, 3}, {1, 1}, {2, 2}};
  std::map<int, int, std::greater<>> desc{{3, 3}, {1, 1}, {2, 2}};

  assert(asc.begin()->first == 1);
  assert(desc.begin()->first == 3);
}
