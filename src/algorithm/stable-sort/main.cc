#include <algorithm>
#include <cassert>
#include <vector>

struct Item {
  int priority;
  int order;
};

int main() {
  std::vector<Item> items{{1, 1}, {2, 2}, {1, 3}, {2, 4}};

  std::stable_sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.priority < b.priority; });

  // items with priority 1 keep original order: (1,1) before (1,3)
  assert(items[0].order == 1 && items[1].order == 3);
}
