========
Variadic
========

.. contents:: Table of Contents
    :backlinks: none

Variadic Function
-----------------

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    int sum(T x)
    {
      return x;
    }

    template <typename T, typename ...Args>
    int sum(T x, Args ...args)
    {
      return x + sum(args...);
    }

    int main(int argc, char *argv[])
    {
      std::cout << sum(1, 2, 3, 4, 5) << std::endl;
    }

By using C++17 or above, Fold expression can simplify the previous snippet.

.. code-block:: cpp

    #include <iostream>

    template <typename ...Args>
    int sum(Args ...args)
    {
      return (args + ...);
    }

    int main(int argc, char *argv[])
    {
      std::cout << sum(1, 2, 3, 4, 5) << std::endl;
    }

Generic lambda expressions
--------------------------

C++14 allows lambda function using ``auto`` type-specifier in the arguments,
which is similar to a template function.

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    auto sum(T x)
    {
      return x;
    }

    template <typename T, typename ...Args>
    auto sum(T x, Args ...args)
    {
      return x + sum(args...);
    }

    int main(int argc, char *argv[])
    {
      auto s = [](auto ...args) { return sum(args...); };
      std::cout << s(1, 2, 3, 4, 5) << std::endl;
    }

By using C++17 or above, a programmer can simplify the previous code as following
snippet.

.. code-block:: cpp

    // g++ -std=c++17 -Wall -Werror -O3 a.cc

    #include <iostream>

    int main(int argc, char *argv[])
    {
      auto sum = [](auto ...args) { return (args + ...); };
      std::cout << sum(1, 2, 3, 4, 5) << std::endl;
    }

Variadic constructor
--------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    class Foo {
     public:

      template <typename ...Args>
      Foo(Args ...args)
      {
        Sum(args...);
      }

      template <typename T>
      void Sum(T t)
      {
        sum += t;
      }

      template <typename T, typename ...Args>
      void Sum(T t, Args ...args)
      {
        sum += t;
        Sum(args...);
      }

      void Print()
      {
        std::cout << sum << std::endl;
      }

     private:
       int sum = 0;
    };

    int main(int argc, char *argv[])
    {
      auto f = Foo(1, 2, 3, 4, 5);
      f.Print();
    }

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    class Foo {
     public:

      template <typename T>
      Foo(T t)
      {
        sum += t;
      }

      template <typename T, typename ...Args>
      Foo(T t, Args ...args) : Foo(args...)
      {
        sum += t;
      }

      void Print()
      {
        std::cout << sum << std::endl;
      }

     private:
       int sum = 0;
    };

    int main(int argc, char *argv[])
    {
      auto f = Foo(1, 2, 3, 4, 5);
      f.Print();
    }


.. warning::

    Please don't invoke a template constructor in a contructor because a new object
    will be created instead of updating the current object's status.

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    class Foo {
     public:
      template <typename T>
      Foo(T t)
      {
        sum += t;
      }

      template <typename T, typename ...Args>
      Foo(T t, Args ...args)
      {
        sum += t;
        Foo(args...);
      }

      void Print()
      {
        std::cout << sum << std::endl;
      }

     private:
       int sum = 0;
    };

    int main(int argc, char *argv[])
    {
      auto f = Foo(1, 2, 3, 4, 5);
      f.Print();
    }

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    class Foo {
     public:
      template <typename ...Args>
      Foo(Args ...args)
      {
        sum = (args + ...);
      }

      void Print()
      {
        std::cout << sum << std::endl;
      }

     private:
       int sum = 0;
    };

    int main(int argc, char *argv[])
    {
      auto f = Foo(1, 2, 3, 4, 5);
      f.Print();
    }

Static Loop unrolling
---------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    template <size_t N>
    struct Loop {
      template <typename F, typename ...Args>
      static void run(F &&f, Args&& ...args)
      {
        Loop<N-1>::run(std::forward<F>(f),std::forward<Args>(args)...);
        f(args..., N-1);
      }
    };

    template <>
    struct Loop<0> {
      template <typename F, typename ...Args>
      static void run(F &&f, Args&& ...args) {}
    };

    int main(int argc, char *argv[])
    {
      size_t counter = 0;
      // for (int i = 0; i < 5; ++i) { counter += i; }
      Loop<5>::run([&](auto i) { counter += i; });
      std::cout << counter << std::endl;
    }

Fold expression
---------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[])
    {
      [](auto ...args) {
        return (args + ...);
      }(1, 2, 3 ,4 ,5);

      std::vector<int> v;
      [](auto &&v, auto ...args) {
        (v.emplace_back(args), ...);
      }(v);

      [](auto ...args) {
        (std::cout << ... << args) << "\n";
      }(1, 2, 3, 4, 5);

      [](auto &&f, auto ...args) {
        return (... + f(args));
      }([](auto x) { return x * 2; }, 1, 2, 3, 4, 5);
    }

