#include <iostream>

template <typename T>
class A {
 protected:
  T p_;

 public:
  A(T p) : p_(p) {}
};

template <typename T>
class B : A<T> {
  using A<T>::p_;

 public:
  B(T p) : A<T>(p) {}
  void print() { std::cout << p_ << "\n"; }
};

template <typename T>
class C : A<T> {
 public:
  C(T p) : A<T>(p) {}
  void print() { std::cout << this->p_ << "\n"; }
};

int main() {
  B<int> b(42);
  C<int> c(99);
  b.print();
  c.print();
}
