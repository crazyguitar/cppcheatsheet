================
Basic cheatsheet
================

.. contents:: Table of Contents
    :backlinks: none

C Linkage
---------

.. code-block:: cpp

    #include <iostream>

    #ifdef __cplusplus
    extern "C" {
    #endif

    int fib(int n) {
      int a = 0, b = 1;
      for (int i = 0; i < n; ++i) {
        auto x = b;
        b = a + b;
        a = x;
      }
      return a;
    }

    #ifdef __cplusplus
    }
    #endif

    int main(int argc, char *argv[]) {
      std::cout << fib(10) << "\n";
    }
    // $ g++ -std=c++17 -Wall -Werror -O3 a.cc
    // $ nm -g a.out | grep fib
    // 0000000100003a58 T _fib

.. code-block:: cpp

    #include <iostream>

    int fib(int n) {
      int a = 0, b = 1;
      for (int i = 0; i < n; ++i) {
        auto x = b;
        b = a + b;
        a = x;
      }
      return a;
    }

    int main(int argc, char *argv[]) {
      std::cout << fib(10) << "\n";
    }
    // $ g++ -std=c++17 -Wall -Werror -O3 a.cc
    // nm -g a.out | grep fib
    // 0000000100003a58 T __Z3fibi

.. code-block:: cpp

    #include <iostream>
    #include <sys/cdefs.h>

    __BEGIN_DECLS

    int fib(int n) {
      int a = 0, b = 1;
      for (int i = 0; i < n; ++i) {
        auto x = b;
        b = a + b;
        a = x;
      }
      return a;
    }

    __END_DECLS

    int main(int argc, char *argv[]) {
      std::cout << fib(10) << "\n";
    }
    // $ g++ -std=c++17 -Wall -Werror -O3 a.cc
    // $ nm -g a.out | grep fib
    // 0000000100003a58 T _fib

Uniform Initialization
----------------------

*Uniform Initialization* is also called braced initialization, which unifies
constructing an object using a brace. However, there are some pitfalls in using
syntax. For example, the compiler prefers to call ``std::initializer_list`` to
initialize an object even with a matched constructor. The following snippet shows
that ``x{10, 5.0}`` will call ``Foo(std::initializer_list<long double>)`` to
construct an object event though ``Foo(int a, double b)`` is the more suitable one.

.. code-block:: cpp

   #include <iostream>
   #include <initializer_list>

   class Foo {
   public:
     Foo(int a, double b) {
       std::cout << "without initializer_list\n";
     }

     Foo(std::initializer_list<long double> il) {
       std::cout << "with initializer_list\n";
     }
   };

   int main(int argc, char *argv[]) {
     Foo x{10, 5.0};
      // output: with initializer_list
   }


Moreover, *uniform initialization* does not support narrowing conversion.
Therefore, the following snippet will compile errors because ``int`` and
``double`` need to do narrowing conversion ``bool``.

.. code-block:: cpp

    #include <iostream>
    #include <initializer_list>

    class Foo {
    public:
      Foo(int a, double b) {
        std::cout << "without initializer_list\n";
      }

      // compile error
      Foo(std::initializer_list<bool> il) {
        std::cout << "with initializer_list\n";
      }
    };

    int main(int argc, char *argv[]) {
      Foo x{10, 5.0};
    }

Note that when types cannot convert, the compiler does not use ``std::initializer_list``
to initialize an object. For example, ``int`` and ``double`` cannot convert to
``std::string``, so the compiler will call ``Foo(int, double)`` to create an object.

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <initializer_list>

    class Foo {
    public:
      Foo(int a, double b) {
        std::cout << "without initializer_list\n";
      }

      Foo(std::initializer_list<std::string> il) {
        std::cout << "with initializer_list\n";
      }
    };

    int main(int argc, char *argv[]) {
      Foo x{10, 5.0};
      // output: without initializer_list
    }


Negative Array index
--------------------

.. code-block:: cpp

    #include <iostream>

    int main(int argc, char *argv[]) {
        // note: arr[i] = *(a + i)
        int arr[] = {1, 2, 3};
        int *ptr = &arr[1];

        std::cout << ptr[-1] << "\n";
        std::cout << ptr[0] << "\n";
        std::cout << ptr[1] << "\n";
    }


Reference
---------

.. code-block:: cpp

    #include <iostream>

    template<typename T>
    void f(T& param) noexcept {}
    // param is a reference

    int main(int argc, char *argv[])
    {
        int x = 123;
        const int cx = x;
        const int &rx = x;

        f(x);   // type(param) = int&
        f(cx);  // type(param) = const int&
        f(rx);  // type(param) = const int&

        return 0;
    }


.. code-block:: cpp

    #include <iostream>

    template<typename T>
    void f(T&& param) noexcept {}
    // param is a universal reference

    int main(int argc, char *argv[])
    {
        int x = 123;
        const int cx = x;
        const int &rx = x;

        f(x);   // x is a lvalue, type(param) = int&
        f(cx);  // cx is a lvalue, type(param) = const int&
        f(rx);  // rx is a lvalue, type(param) = const int&
        f(12);  // 12 is a rvalue, type(param) = int&&

        return 0;
    }

.. code-block:: cpp

    #include <iostream>

    template<typename T>
    void f(T param) noexcept {}
    // param is neither a pointer nor a reference.

    int main(int argc, char *argv[])
    {
        int x = 123;
        const int cx = x;
        const int &rx = x;

        f(x);   // type(param) = int
        f(cx);  // type(param) = int
        f(rx);  // type(param) = int
        f(12);  // type(param) = int

        return 0;
    }

auto
----

.. code-block:: cpp

    auto x = 123;        // type(x) = int
    const auto cx = x;   // type(cx) = const int
    const auto &rx = x;  // type(rx) = const int&

    auto &&urx = x;      // type(urx) = int&
    auto &&urcx = cx;    // type(urcx) = const int&
    auto &&urrx = rx;    // type(urrx) = const int&
    auto &&urrv = 12;    // type(urrv) = int&&

decltype(auto)
--------------

The ``decltype(auto)`` is similar to auto, which decudes type via compiler.
However, ``decltype(auto)`` preserves types reference and cv-qualifiers, while
auto does not.

.. code-block:: cpp

    #include <type_traits>

    int main(int argc, char *argv[]) {
      int x;
      const int cx = x;
      const int &crx = x;
      int &&z = 0;

      // decltype(auto) preserve cv-qualifiers
      decltype(auto) y1 = crx;
      static_assert(std::is_same<const int &, decltype(y1)>::value == 1);
      // auto does not preserve cv-qualifiers
      auto y2 = crx;
      static_assert(std::is_same<int, decltype(y2)>::value == 1);
      // decltype(auto) preserve rvalue reference
      decltype(auto) z1 = std::move(z);
      static_assert(std::is_same<int &&, decltype(z1)>::value == 1);
    }

``decltype(auto)`` is especially useful for writing a generic function's return.

.. code-block:: cpp

    #include <type_traits>

    auto foo(const int &x) {
      return x;
    }

    decltype(auto) bar(const int &x) {
      return x;
    }

    int main(int argc, char *argv[]) {
      static_assert(std::is_same<int, decltype(foo(1))>::value == 1);
      static_assert(std::is_same<const int &, decltype(bar(1))>::value == 1);
    }

Reference Collapsing
--------------------

.. code-block:: cpp

    // T& & -> T&
    // T& && -> T&
    // T&& & -> T&
    // T&& && -> T&&
    // note & always wins. that is T& && == T&& & == T& & == T&
    // only T&& && == T&&

Perfect Forwarding
------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <type_traits>

    template <typename T>
    T&& forward(typename std::remove_reference<T>::type& t) noexcept {
      std::cout << std::is_lvalue_reference<decltype(t)>::value << std::endl;
      return static_cast<T&&>(t);
    }

    template <typename T>
    T&& forward(typename std::remove_reference<T>::type&& t) noexcept {
      static_assert(
        !std::is_lvalue_reference<T>::value,
        "Can not forward an rvalue as an lvalue."
      );
      std::cout << std::is_lvalue_reference<decltype(t)>::value << std::endl;
      return static_cast<T&&>(t);
    }

    int main (int argc, char *argv[])
    {
      int a = 0;
      forward<int>(a);     // forward lvalues to rvalues
      forward<int>(9527);  // forward rvalues to rvalues
      return 0;
    }

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <type_traits>

    template <typename T, typename Func>
    void wrapper(T &&a, Func fn) {
      fn(std::forward<T>(a)); // forward lvalue to lvalues or rvalues
    }

    struct Foo {
      Foo(int a1, int a2) : a(a1), b(a2), ret(0) {}
      int a, b, ret;
    };

    int main (int argc, char *argv[])
    {
      Foo foo{1, 2};
      Foo &bar = foo;
      Foo &&baz = Foo(5, 6);

      wrapper(foo, [](Foo foo) {
        foo.ret =  foo.a + foo.b;
        return foo.ret;
      });
      std::cout << foo.ret << std::endl;

      wrapper(bar, [](Foo &foo) {
        foo.ret = foo.a - foo.b;
        return foo.ret;
      });
      std::cout << bar.ret << std::endl;

      // move an rvalue to lvalue
      wrapper(std::move(baz), [](Foo &&foo) {
        foo.ret = foo.a * foo.b;
        return foo.ret;
      });
      std::cout << baz.ret << std::endl;
      return 0;
    }

Bit Manipulation
----------------

.. code-block:: cpp

	#include <iostream>
	#include <bitset>

	int main(int argc, char *argv[]) {
		std::bitset<4> b{8};

		// show number of bits set
		std::cout << b.count() << "\n";
		// compare with int
		std::cout << (b == 8) << "\n";
	}

Using ``std::addressof``
------------------------

Because C++ allows the overloading of ``operator &``, accessing the address of
an reference will result in infinite recusion. Therefore, when it is necessary
to access the address of reference, it would be safer by using ``std::addressof``.

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct A {
      int x;
    };

    const A *operator &(const A& a) {
      // return &a; <- infinite recursion
      return std::addressof(a);
    }

    int main(int argc, char *argv[]) {
      A a;
      std::cout << &a << "\n";
    }
