#include <iostream>

struct A {};
struct B {};

template <typename T, typename U>
struct Foo {
  Foo(T t, U u) : t_(t), u_(u) {}
  T t_;
  U u_;
};

template <typename F, typename T, typename U>
struct Bar {
  Bar(T t, U u) : f_(t, u) {}
  F f_;
};

template class Foo<A, B>;

int main() { Bar<Foo<A, B>, A, B>(A(), B()); }
