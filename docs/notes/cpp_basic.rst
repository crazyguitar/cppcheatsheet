====================
C++ Basic cheatsheet
====================

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
