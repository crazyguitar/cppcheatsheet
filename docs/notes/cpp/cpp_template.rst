========
Template
========

.. contents:: Table of Contents
    :backlinks: none

Instantiate a Template
----------------------

.. code-block:: cpp

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

    // instantiate template Foo
    template class Foo<A, B>;

    int main() {
      Bar<Foo<A, B>, A, B>(A(), B());
      return 0;
    }

Template Specialization
-----------------------

.. code-block:: cpp

    #include <iostream>

    template <typename T, typename U>
    class Base
    {
    private:
      T m_a;
      U m_b;

    public:

      Base(T a, U b) : m_a(a), m_b(b) {};

      T foo() { return m_a; }
      U bar() { return m_b; }
    };

    // partial specialization
    template<typename T>
    class Base <T, int>
    {
    private:
      T m_a;
      int m_b;
    public:
      Base(T a, int b) : m_a(a), m_b(b) {}
      T foo() { return m_a; }
      int bar() { return m_b; }
    };

    // full specialization
    template<>
    class Base <double, double>
    {
    private:
      double d_a;
      double d_b;
    public:
      Base(double a, double b) : d_a(a), d_b(b) {}
      double foo() { return d_a; }
      double bar() { return d_b; }
    };


    int main (int argc, char *argv[])
    {
      Base<float, int> foo(3.33, 1);
      Base<double, double> bar(55.66, 95.27);
      std::cout << foo.foo() << std::endl;
      std::cout << foo.bar() << std::endl;
      std::cout << bar.foo() << std::endl;
      std::cout << bar.bar() << std::endl;
      return 0;
    }

Class Template
--------------

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class Area
    {
    protected:
      T w;
      T h;
    public:
      Area(T a, T b) : w(a), h(b) {}
      T get() { return w * h; }
    };

    class Rectangle : public Area<int>
    {
    public:
      Rectangle(int a, int b) : Area<int>(a, b) {}
    };

    template <typename T>
    class GenericRectangle : public Area<T>
    {
    public:
      GenericRectangle(T a, T b) : Area<T>(a, b){}
    };


    int main (int argc, char *argv[])
    {
      Rectangle r(2, 5);
      GenericRectangle<double> g1(2.5, 3.);
      GenericRectangle<int> g2(2, 3);

      std::cout << r.get() << std::endl;
      std::cout << g1.get() << std::endl;
      std::cout << g2.get() << std::endl;
      return 0;
    }

Variadic Template (Parameter Pack)
----------------------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <vector>

    template <typename T>
    class Vector {
    protected:
      std::vector<T> v;
    public:

      template<typename ...Args>
      Vector(Args&&... args) {
        (v.emplace_back(std::forward<Args>(args)), ...);
      }

      using iterator = typename std::vector<T>::iterator;
      iterator begin() noexcept { return v.begin(); }
      iterator end() noexcept { return v.end(); }
    };


    int main(int argc, char *argv[]) {

      Vector<int> v{1,2,3};
      for (const auto &x : v)
      {
        std::cout << x << "\n";
      }
    }

Fold expressions
----------------

.. code-block:: cpp

    // g++ -std=c++17 -Wall -Werror -O3 a.cc

    #include <iostream>
    #include <utility>

    template <typename ...Args>
    decltype(auto) f(Args&& ...args) {
      auto l = [](auto &&x) { return x * 2; };
      return (l(std::forward<Args>(args)) + ...);
    }

    int main(int argc, char *argv[]) {
      std::cout << f(1, 2, 3, 4, 5) << std::endl;
    }

Limit a Template Types
----------------------

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <type_traits>

    template<typename S,
      typename = typename std::enable_if<
        std::is_same<
          std::string,
          typename std::decay<S>::type
        >::value
      >::type
    >
    void Foo(S s) {
      std::cout << s << "\n";
    }


    int main(int argc, char *argv[]) {
      std::string s1 = "Foo";
      const std::string s2 = "Bar";
      Foo(s1);
      Foo(s2);

      // Foo(123);    compile error
      // Foo("Baz");  compile error
    }

Specialize Types
----------------

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <type_traits>

    template<typename S>
    void Foo(S s) {
      if (std::is_integral<S>::value) {
        std::cout << "do a task for integer..." << "\n";
        return;
      }
      if (std::is_same<std::string, typename std::decay<s>::type>::value)
      {
        std::cout << "do a task for string..." << "\n";
        return;
      }
    }

    int main(int argc, char *argv[]) {
      std::string s1 = "Foo";
      Foo(s1);
      Foo(123);
    }

Template Specialization approach

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <type_traits>

    template<typename S>
    void Foo(S s) {}

    template <>
    void Foo<int>(int s) {
      std::cout << "do a task for integer..." << "\n";
    }
    template<>
    void Foo<std::string>(std::string s) {
      std::cout << "do a task for string..." << "\n";
    }


    int main(int argc, char *argv[]) {
      std::string s1 = "Foo";
      Foo(s1);
      Foo(123);
    }


Curiously recurring template pattern
------------------------------------

.. code-block:: cpp

    #include <iostream>

    // Curiously Recurring Template Pattern (CRTP)

    template <typename D>
    class Base
    {
    public:
      void interface() {
        static_cast<D *>(this)->implement();
      }

      static void static_interface() {
        D::static_interface();
      }

      void implement() {
        std::cout << "Base" << std::endl;
      }
    };

    class DerivedFoo : public Base<DerivedFoo>
    {
    public:
      void implement() {
        std::cout << "Foo" << std::endl;
      }
      static void static_interface() {
        std::cout << "Static Foo" << std::endl;
      }
    };

    class DerivedBar : public Base<DerivedBar> {};

    int main (int argc, char *argv[])
    {
      DerivedFoo foo;
      DerivedBar bar;

      foo.interface();
      foo.static_interface();
      bar.interface();

      return 0;
    }

Parametric Expressions
----------------------

.. code-block:: cpp

    #include <iostream>

    // g++ -std=c++17 -fconcepts -g -O3 a.cpp

    decltype(auto) min(auto&& lhs, auto&& rhs) {
      return lhs < rhs ? lhs : rhs;
    }

    int main(int argc, char *argv[]) {
      std::cout << min(1, 2) << "\n";
      std::cout << min(3.14, 2.718) << "\n";
    }

.. code-block:: cpp

    #include <iostream>

    template<typename T>
    decltype(auto) min(T&& lhs,T&& rhs) {
      return lhs < rhs ? lhs : rhs;
    }

    int main(int argc, char *argv[]) {
      std::cout << min(1, 2) << "\n";
      std::cout << min(3.14, 2.718) << "\n";
    }

.. code-block:: cpp

    #include <iostream>

    auto min = [](auto&& lhs, auto&& rhs) {
      return lhs < rhs ? lhs : rhs;
    };

    int main(int argc, char *argv[]) {
      std::cout << min(1, 2) << "\n";
      std::cout << min(3.14, 2.718) << "\n";
    }

Reference

_ `Parametric Expressions`_

.. _Parametric Expressions: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1221r0.html

Template Template Parameters
----------------------------

.. code-block:: cpp

    #include <vector>
    #include <deque>

    template <template<class, class> class V, class T, class A>
    void f(V<T, A> &v) {
      v.pop_back();
    }

    int main(int argc, char *argv[]) {
      std::vector<int> v{0};
      std::deque<int> q{1};
      f<std::vector, int>(v);
      f<std::deque, int>(q);
    }

Access Protected Membors in Sub-Template
----------------------------------------

Accessing protected members by pulling the names into the current scope via ``using``.

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class A {
     public:
      A(T p) : p_{p} {}
      decltype(auto) f() { std::cout << p_ << "\n"; }
     protected:
      T p_;
    };

    template <typename T>
    class B : A<T> {
      using A<T>::p_;
     public:
      B(T p) : A<T>(p) {}
      decltype(auto) g() { std::cout << p_ << "\n"; }
    };

    int main(int argc, char *argv[]) {
      A<int> a(0);
      B<int> b(0);
      a.f();
      b.g();
    }

Another option is qualifying name via the ``this`` pointer.

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class A {
     public:
      A(T p) : p_{p} {}
      decltype(auto) f() { std::cout << p_ << "\n"; }
     protected:
      T p_;
    };

    template <typename T>
    class B : A<T> {
     public:
      B(T p) : A<T>{p} {}
      decltype(auto) g() { std::cout << this->p_ << "\n"; }
    };

    int main(int argc, char *argv[]) {
      A<int> a(0);
      B<int> b(0);
      a.f();
      b.g();
    }
