=========
constexpr
=========

.. contents:: Table of Contents
    :backlinks: none

constexpr Function
------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <chrono>

    class Timer {
    public:
        inline void start() {
            start_ = std::chrono::system_clock::now();
        }

        inline void end() {
            end_ = std::chrono::system_clock::now();
        }

        inline void out() {
            std::chrono::duration<double> d = end_ - start_;
            std::cout << "Time cost: " << d.count() << "\n";
        }
    private:
        std::chrono::time_point<std::chrono::system_clock> start_;
        std::chrono::time_point<std::chrono::system_clock> end_;
    };

    constexpr long fib(long n) {
        return (n < 2) ? 1 : fib(n-1) + fib(n-2);
    }

    int main() {
        Timer timer;
        long n = 40;

        timer.start();
        int r1 = fib(n);
        timer.end();
        timer.out();

        timer.start();
        constexpr long r2 = fib(40);
        timer.end();
        timer.out();

        return 0;
    }

output:

.. code-block:: bash

    $ g++ -std=c++14 -g -O3 a.cpp
    $ ./a.out
    Time cost: 0.268229
    Time cost: 8e-06

Compare to Metaprogramming
--------------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include "timer.h"

    template <long N>
    struct Fib {
        static long const v = Fib<N-1>::v + Fib<N-2>::v;
    };

    template <>
    struct Fib<0> {
        static long const v = 1;
    };

    template <>
    struct Fib<1> {
        static long const v = 1;
    };

    constexpr long fib(long n)
    {
        return (n < 2) ? 1 : fib(n-1) + fib(n-2);
    }

    int main() {

        Timer timer;

        timer.start();
        constexpr long r1 = Fib<40>::v;
        timer.end();
        timer.out();

        timer.start();
        constexpr long r2 = fib(40);
        timer.end();
        timer.out();

        return 0;
    }

output:

.. code-block:: bash

    g++ -std=c++14 -g -O3 a.cpp
    $ ./a.out
    Time cost: 9.7e-06
    Time cost: 9.2e-06

After C++14, constexpr functions can

- invoke other constexpr functions.
- have variables with a constant expression.
- include conditional expressions or loops.
- be implicit inline.
- not have static or thread_local data.
