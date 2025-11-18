======
Lambda
======

.. contents:: Table of Contents
    :backlinks: none

Callable Objects
----------------

.. code-block:: cpp

    #include <iostream>

    class Fib {
    public:
        long operator() (const long n) {
            return (n <= 2) ? 1 : operator()(n-1) + operator()(n-2);
        }
    };

    int main() {
        Fib fib;
        std::cout << fib(10) << "\n";
        return 0;
    }

Lambda version

.. code-block:: cpp

    #include <iostream>
    #include <functional>

    int main() {
        std::function<long(long)> fib = [&](long n) {
            return (n <= 2) ? 1 : fib(n-1) + fib(n-2);
        };
        std::cout << fib(10) << "\n";
        return 0;
    }

Default Arguments
-----------------

.. code-block:: cpp

    #include <iostream>

    int main(int argc, char *argv[]) {
        auto fib = [](long n=0) {
            long a = 0, b = 1;
            for (long i = 0; i < n; ++i) {
                long tmp = b;
                b = a + b;
                a = tmp;
            }
            return a;
        };
        std::cout << fib() << "\n";
        std::cout << fib(10) << "\n";
        return 0;
    }

Captureless
-----------

.. code-block:: cpp

    #include <iostream>

    int main() {
        long (*fib)(long) = [](long n) {
            long a = 0, b = 1;
            for (long i = 0; i < n; ++i) {
                long tmp = b;
                b = a + b;
                a = tmp;
            }
            return a;
        };
        std::cout << fib(10) << "\n";
        return 0;
    }

Lambda capture initializers
---------------------------

.. code-block:: cpp

    // g++ -std=c++17 -Wall -Werror -O3 a.cc

    #include <iostream>
    #include <utility>
    #include <memory>

    int main(int argc, char *argv[])
    {
      std::unique_ptr<int> p = std::make_unique<int>(5566);
      auto f = [x = std::move(p)]() { std::cout << *x << std::endl; };
      f();
    }

Capture by ``std::move``
------------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    struct Foo {
        Foo() { std::cout << "Constructor" << "\n"; }
        ~Foo() { std::cout << "Destructor" << "\n"; }
        Foo(const Foo&) { std::cout << "Copy Constructor" << "\n"; }
        Foo(Foo &&) { std::cout << "Move Constructor" << "\n";}

        Foo& operator=(const Foo&) {
            std::cout << "Copy Assignment" << "\n";
            return *this;
        }
        Foo& operator=(Foo &&){
            std::cout << "Move Assignment" << "\n";
            return *this;
        }
    };

    int main(int argc, char *argv[]) {
        Foo foo;
        [f=std::move(foo)] { /* do some tasks here...*/ }();
    }


Copy a Global into a Capture
----------------------------

.. code-block:: cpp

    #include <iostream>

    int g = 1;

    // copy a global to a capture
    auto bar = [g=g]() { return g + 1; };

    int main(int argc, char *argv[]) {
        int g = 10;
        std::cout << bar() << "\n";
    }

constexpr by Default
--------------------

.. code-block:: cpp

    #include <iostream>

    int main() {
        auto fib = [](long n) {
            long a = 0, b = 1;
            for (long i = 0; i < n; ++i) {
                long tmp = b;
                b = a + b;
                a = tmp;
            }
            return a;
        };

        // constexpr by default is new in c++17
        static_assert(fib(10) == 55);
        return 0;
    }

output:

.. code-block:: bash

    $ g++ -std=c++17 -g -O3 a.cpp

Generic Lambda
--------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    // g++ -std=c++17 -g -O3 a.cpp

    class Sum {
    public:
        template <typename ...Args>
        constexpr auto operator()(Args&& ...args) {
            // Fold expression (since c++17)
            return (std::forward<Args>(args) + ...);
        }
    };

    int main() {
        Sum sum;
        constexpr int ret = sum(1,2,3,4,5);
        std::cout << ret << std::endl;
        return 0;
    }

The snippet is equal to the following example

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    int main() {
        auto sum = [](auto&& ...args) {
            return (std::forward<decltype(args)>(args) + ...);
        };
        constexpr int ret = sum(1,2,3,4,5);
        std::cout << ret << std::endl;
        return 0;
    }

In c+20, lambda supports explicit template paramter list allowing a programmer
to utilize parameters' type instead of using `decltype`.

.. code-block:: cpp

    #include <iostream>

    // g++ -std=c++2a -g -O3 a.cpp

    int main(int argc, char *argv[])
    {
        auto sum = []<typename ...Args>(Args&&... args) {
            return (std::forward<Args>(args) + ...);
        };
        constexpr int ret = sum(1,2,3,4,5);
        std::cout << ret << std::endl;
        return 0;
    }

Comparison Function
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <map>

    struct Cmp {
        template<typename T>
        bool operator() (const T &lhs, const T &rhs) const {
            return lhs < rhs;
        }
    };

    int main(int argc, char *argv[]) {

        // sort by keys
        std::map<int, std::string, Cmp> m;

        m[3] = "Foo";
        m[2] = "Bar";
        m[1] = "Baz";

        for (auto it : m) {
            std::cout << it.first << ", " << it.second << "\n";
        }
        return 0;
    }

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <map>

    bool cmp(const int &lhs, const int &rhs) {
        return lhs < rhs;
    }

    int main(int argc, char *argv[]) {

        // sort by keys
        std::map<int, std::string, decltype(&cmp)> m(cmp);

        m[3] = "Foo";
        m[2] = "Bar";
        m[1] = "Baz";

        for (auto it : m) {
            std::cout << it.first << ", " << it.second << "\n";
        }
        return 0;
    }

.. code-block:: cpp

    #include <iostream>
    #include <functional>
    #include <string>
    #include <map>

    template<typename T>
    using Cmp = std::function<bool(const T &, const T &)>;

    template<typename T>
    bool cmp(const T &lhs, const T &rhs) {
        return lhs < rhs;
    }

    int main(int argc, char *argv[]) {

        // sort by keys
        std::map<int, std::string, Cmp<int>> m(cmp<int>);

        m[3] = "Foo";
        m[2] = "Bar";
        m[1] = "Baz";

        for (auto it : m) {
            std::cout << it.first << ", " << it.second << "\n";
        }
        return 0;
    }


.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <map>

    int main(int argc, char *argv[]) {

        auto cmp = [](auto &lhs, auto &rhs) {
            return lhs < rhs;
        };

        // sort by keys
        std::map<int, std::string, decltype(cmp)> m(cmp);

        m[3] = "Foo";
        m[2] = "Bar";
        m[1] = "Baz";

        for (auto it : m) {
            std::cout << it.first << ", " << it.second << "\n";
        }
        return 0;
    }


Break Loops
-----------

.. code-block:: cpp

    #include <iostream>

    int main(int argc, char *argv[]) {
        bool is_stoped = false;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::cout << i + j << " ";
                if (i + j == 5) {
                    is_stoped = true;
                    break;
                }
            }
            if (is_stoped) {
                break;
            }
        }
        std::cout << std::endl;
        return 0;
    }

The previous example shows a common way to break multiple loops via a flag.
However, the drawback is a programmer requires to maintain flags if code
includes nested loops. By using a lambda function, it is convenient for
developers to break nested loops through the return.

.. code-block:: cpp

    #include <iostream>

    int main(int argc, char *argv[]) {
        [&] {
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    std::cout << i + j << " ";
                    if (i + j == 5) {
                        return;
                    }
                }
            }
        }();
        std::cout << std::endl;
        return 0;
    }

Callback
--------

.. code-block:: cpp

    #include <iostream>

    template<typename F>
    long fib(long n, F f) {
        long a = 0, b = 1;
        for (long i = 0; i < n; ++i) {
            long tmp = b;
            b = a + b;
            a = tmp;
            f(a);
        }
        return a;
    }

    int main(int argc, char *argv[]) {
        fib(10, [](long res) {
            std::cout << res << " ";
        });
        std::cout << "\n";
        return 0;
    }

.. code-block:: cpp

    #include <iostream>
    #include <functional>

    using fibcb = std::function<void(long x)>;

    long fib(long n, fibcb f) {
        long a = 0, b = 1;
        for (long i = 0; i < n; ++i) {
            long tmp = b;
            b = a + b;
            a = tmp;
            f(a);
        }
        return a;
    }

    int main(int argc, char *argv[]) {
        fib(10, [](long res) {
            std::cout << res << " ";
        });
        std::cout << "\n";
        return 0;
    }

Programmers can also use function pointers to define a functino's callback
parameter. However, function pointers are only suitable for captureless lambda
functions.

.. code-block:: cpp

    #include <iostream>
    #include <functional>

    using fibcb = void(*)(long n);

    long fib(long n, fibcb f) {
        long a = 0, b = 1;
        for (long i = 0; i < n; ++i) {
            long tmp = b;
            b = a + b;
            a = tmp;
            f(a);
        }
        return a;
    }

    int main(int argc, char *argv[]) {
        fib(10, [](long res) {
            std::cout << res << " ";
        });
        std::cout << "\n";
        return 0;
    }

Recursion
---------

There are two ways to run a lambda function recursively. The first one is using
``std::function``. However, this solution is slower than normal recursive function.

.. code-block:: cpp

    #include <iostream>
    #include <functional>
    #include <chrono>

    int main(int argc, char *argv[]) {
      using namespace std::chrono;
      std::function<int(int)> fib;
      fib = [&](auto n) { return n <= 1 ? 1 : (fib(n - 1) + fib(n - 2)); };
      auto s = system_clock::now();
      fib(30);
      auto e = system_clock::now();
      std::cout << duration_cast<milliseconds>(e - s).count() << "\n";
    }

Another way is making lamdbda as an input arguemnt. The performance approaches
the normal function.

.. code-block:: cpp

    #include <iostream>
    #include <chrono>


    int main(int argc, char *argv[]) {
      using namespace std::chrono;
      auto fib = [&](auto &&n, auto &&fib) -> int {
        return n <= 1 ? 1 : (fib(n - 1, fib) + fib(n - 2, fib));
      };
      auto s = system_clock::now();
      fib(30, fib);
      auto e = system_clock::now();
      std::cout << duration_cast<milliseconds>(e - s).count() << "\n";
    }

Reference
---------

1. `Back to Basics: Lambdas from Scratch`_
2. `Demystifying C++ lambdas`_

.. _Back to Basics\: Lambdas from Scratch: https://youtu.be/3jCOwajNch0
.. _Demystifying C++ lambdas: https://blog.feabhas.com/2014/03/demystifying-c-lambdas/
