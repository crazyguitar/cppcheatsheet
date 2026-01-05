#include <iostream>

template <typename T>
class Area {
 protected:
  T w, h;

 public:
  Area(T a, T b) : w(a), h(b) {}
  T get() { return w * h; }
};

class Rectangle : public Area<int> {
 public:
  Rectangle(int a, int b) : Area<int>(a, b) {}
};

template <typename T>
class GenericRectangle : public Area<T> {
 public:
  GenericRectangle(T a, T b) : Area<T>(a, b) {}
};

int main() {
  Rectangle r(2, 5);
  GenericRectangle<double> g(2.5, 3.0);
  std::cout << r.get() << "\n";
  std::cout << g.get() << "\n";
}
