#include <cassert>
#include <map>

struct Point {
  int x, y;
};

int main() {
  auto cmp = [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };

  std::map<Point, int, decltype(cmp)> m(cmp);
  m[{1, 2}] = 1;
  m[{3, 4}] = 2;
  m[{1, 3}] = 3;

  assert(m.size() == 3);
  assert(m.begin()->second == 1);  // (1,2) is smallest
}
