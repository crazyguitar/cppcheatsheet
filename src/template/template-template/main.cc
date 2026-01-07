#include <deque>
#include <iostream>
#include <vector>

template <template <class, class> class V, class T, class A>
void pop(V<T, A>& v) {
  std::cout << v.back() << "\n";
  v.pop_back();
}

int main() {
  std::vector<int> v{1, 2, 3};
  std::deque<int> q{4, 5, 6};
  pop(v);
  pop(q);
}
