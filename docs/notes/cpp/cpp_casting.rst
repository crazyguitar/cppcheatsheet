=======
Casting
=======

.. contents:: Table of Contents
    :backlinks: none

C Style Casting
---------------

.. code-block:: cpp

    #include <iostream>
    #include <cmath>

    int main(int argc, char *argv[]) {
      double x = M_PI;
      int xx = (int) x;
      std::cout << xx << "\n";

      long z = LONG_MAX;
      int zz = (int) z;  // dangerous. overflow
      std::cout << zz << "\n";
    }

Const Casting
-------------

.. code-block:: cpp

    #include <iostream>

    void f(const int &x) {
      const_cast<int &>(x) = 0;
    }

    int main(int argc, char *argv[]) {
      int x = 123;
      f(x);
      std::cout << x << "\n";
    }

Reinterpret Casting
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <iomanip>

    struct A { int x; int y; };

    int main(int argc, char *argv[]) {
      A a{1, 2};

      // convert a struct to a byte array
      char *buf = reinterpret_cast<char *>(&a);
      for (int i = 0; i < sizeof(A); ++i) {
        std::cout << static_cast<int>(buf[i]) << " ";
      }
      // output: 1 0 0 0 2 0 0 0
    }

Static Casting
--------------

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct A {
      virtual void f() { std::cout << __func__ << "\n"; }
      virtual ~A() = default;
    };

    struct B : public A {;
      B(int *x) : x_{x} {}
      int *g() { return x_; }
      int *x_;
    };

    int main(int argc, char *argv[]) {
      auto a = std::make_unique<A>();
      // downcasting may be dangerous
      auto b = static_cast<B *>(a.get());
      auto x = b->g();
      std::cout << *x << "\n";
    }

Dynamic Casting
---------------

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct A {
      virtual void f() { std::cout << __func__ << "\n"; }
    };

    struct B : public A {
       void f() override { std::cout << __PRETTY_FUNCTION__ << "\n"; }
    };

    int main(int argc, char *argv[]) {
      auto a = std::make_unique<A>();
      auto b = std::make_unique<B>();

      // downcast
      auto bb = dynamic_cast<B *>(a.get());
      std::cout << "Is dynamic_cast(*a) to *b success? " << !!bb << "\n";
      // output: Is dynamic_cast(*a) to *b success? 0

      // upcast
      auto aa = dynamic_cast<A *>(b.get());
      std::cout << "Is dynamic_cast(*b) to *a success? " << !!aa << "\n";
      // output: Is dynamic_cast(*a) to *b success? 1
    }

