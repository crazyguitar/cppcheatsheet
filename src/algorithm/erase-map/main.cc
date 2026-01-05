#include <cassert>
#include <map>
#include <string>

int main() {
  std::map<int, std::string> m{{1, "a"}, {2, "b"}, {3, "c"}};

  for (auto it = m.begin(); it != m.end();) {
    if (it->first > 1) {
      it = m.erase(it);
    } else {
      ++it;
    }
  }
  assert(m.size() == 1 && m[1] == "a");

  std::map<int, std::string> m2{{1, "a"}, {2, "b"}, {3, "c"}};
  std::erase_if(m2, [](auto& p) { return p.first > 1; });
  assert(m2.size() == 1);
}
