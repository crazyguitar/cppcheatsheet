#include <iostream>

template <typename T, typename U>
class Base {
  T m_a;
  U m_b;

 public:
  Base(T a, U b) : m_a(a), m_b(b) {}
  T foo() { return m_a; }
  U bar() { return m_b; }
};

template <typename T>
class Base<T, int> {
  T m_a;
  int m_b;

 public:
  Base(T a, int b) : m_a(a), m_b(b) {}
  T foo() { return m_a; }
  int bar() { return m_b; }
};

template <>
class Base<double, double> {
  double d_a;
  double d_b;

 public:
  Base(double a, double b) : d_a(a), d_b(b) {}
  double foo() { return d_a; }
  double bar() { return d_b; }
};

int main() {
  Base<float, int> foo(3.33, 1);
  Base<double, double> bar(55.66, 95.27);
  std::cout << foo.foo() << "\n";
  std::cout << bar.bar() << "\n";
}
