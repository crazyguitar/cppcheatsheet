=========
Constexpr
=========

.. meta::
   :description: C++ constexpr tutorial for compile-time computation, constexpr functions, variables, and if constexpr with examples.
   :keywords: C++, constexpr, compile-time, constant expression, consteval, if constexpr, C++11, C++14, C++17, C++20

.. contents:: Table of Contents
    :backlinks: none

The ``constexpr`` keyword, introduced in C++11, enables compile-time evaluation
of expressions and functions. By shifting computation from runtime to compile
time, ``constexpr`` improves performance, enables stronger type checking, and
allows values to be used in contexts requiring compile-time constants (such as
array sizes and template arguments). C++14, C++17, and C++20 progressively
relaxed restrictions, making ``constexpr`` increasingly powerful. Related
keywords ``consteval`` (C++20) and ``constinit`` (C++20) provide finer control
over when evaluation and initialization must occur.

constexpr Functions
-------------------

:Source: `src/constexpr/constexpr-function <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/constexpr-function>`_

The ``constexpr`` specifier declares that a function can be evaluated at
compile time when called with constant expressions. This enables the compiler
to compute results during compilation, eliminating runtime overhead entirely.
When called with non-constant arguments, the function executes at runtime
like a normal function. This dual behavior makes ``constexpr`` functions
versatile for both compile-time and runtime use.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    constexpr long fib(long n) {
      return (n < 2) ? n : fib(n - 1) + fib(n - 2);
    }

    int main() {
      // Runtime evaluation
      auto start = std::chrono::system_clock::now();
      long r1 = fib(40);
      std::chrono::duration<double> d1 = std::chrono::system_clock::now() - start;
      std::cout << "Runtime: " << d1.count() << "s\n";

      // Compile-time evaluation
      start = std::chrono::system_clock::now();
      constexpr long r2 = fib(40);
      std::chrono::duration<double> d2 = std::chrono::system_clock::now() - start;
      std::cout << "Compile-time: " << d2.count() << "s\n";
    }

.. code-block:: bash

    $ g++ -std=c++17 -O3 a.cpp && ./a.out
    Runtime: 0.268229s
    Compile-time: 8e-06s

constexpr vs Template Metaprogramming
-------------------------------------

:Source: `src/constexpr/constexpr-vs-tmp <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/constexpr-vs-tmp>`_

Before C++11, compile-time computation required template metaprogramming (TMP),
which uses recursive template instantiation. While powerful, TMP is verbose
and difficult to read. The ``constexpr`` keyword provides a cleaner alternative
that looks like regular code. Both approaches achieve compile-time evaluation,
but ``constexpr`` is more maintainable and easier to debug.

.. code-block:: cpp

    #include <iostream>

    // Template metaprogramming approach (pre-C++11 style)
    template <long N>
    struct Fib {
      static constexpr long value = Fib<N - 1>::value + Fib<N - 2>::value;
    };

    template <>
    struct Fib<0> {
      static constexpr long value = 0;
    };

    template <>
    struct Fib<1> {
      static constexpr long value = 1;
    };

    // constexpr approach (modern C++)
    constexpr long fib(long n) {
      return (n < 2) ? n : fib(n - 1) + fib(n - 2);
    }

    int main() {
      constexpr long r1 = Fib<40>::value;  // TMP
      constexpr long r2 = fib(40);          // constexpr
      std::cout << r1 << " " << r2 << "\n"; // Both: 102334155
    }

constexpr Variables
-------------------

:Source: `src/constexpr/constexpr-variable <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/constexpr-variable>`_

A ``constexpr`` variable must be initialized with a constant expression and
its value is fixed at compile time. Unlike ``const``, which only promises
the variable won't be modified after initialization, ``constexpr`` guarantees
the value is known at compile time. This makes ``constexpr`` variables usable
in contexts requiring compile-time constants, such as array sizes and template
arguments.

.. code-block:: cpp

    #include <array>
    #include <iostream>

    constexpr int square(int x) { return x * x; }

    int main() {
      constexpr int size = square(4);     // Computed at compile time: 16
      std::array<int, size> arr{};        // Array size must be compile-time constant
      std::cout << arr.size() << "\n";    // Output: 16

      const int runtime_val = size;       // const: won't change, but could be runtime
      constexpr int compile_val = size;   // constexpr: guaranteed compile-time
    }

constexpr if (C++17)
--------------------

:Source: `src/constexpr/constexpr-if <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/constexpr-if>`_

C++17 introduced ``if constexpr``, which evaluates conditions at compile time
and discards the untaken branch entirely. This is essential for template
metaprogramming because it prevents compilation errors in branches that would
be invalid for certain template instantiations. Unlike regular ``if``, both
branches of a normal ``if`` must be valid code even if one is never executed.

.. code-block:: cpp

    #include <iostream>
    #include <type_traits>

    template <typename T>
    auto get_value(T t) {
      if constexpr (std::is_pointer_v<T>) {
        return *t;  // Only compiled when T is a pointer
      } else {
        return t;   // Only compiled when T is not a pointer
      }
    }

    int main() {
      int x = 42;
      int *p = &x;

      std::cout << get_value(x) << "\n";  // Output: 42
      std::cout << get_value(p) << "\n";  // Output: 42
    }

if consteval: Detecting Constant Evaluation (C++23)
---------------------------------------------------

:Source: `src/constexpr/if-consteval <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/if-consteval>`_

C++23 introduced ``if consteval`` to detect whether code is executing during
constant evaluation. This allows a function to choose different implementations
for compile-time versus runtime execution. Unlike ``if constexpr``, which
evaluates a compile-time boolean condition, ``if consteval`` checks the
evaluation context itself. This is useful when compile-time evaluation requires
a different algorithm (e.g., a safer but slower approach) than runtime execution.

.. code-block:: cpp

    #include <iostream>

    constexpr int compute(int x) {
      if consteval {
        // Compile-time path: safe but potentially slower
        return x * x;
      } else {
        // Runtime path: can use optimized or platform-specific code
        return x * x;  // In practice, might call intrinsics here
      }
    }

    int main() {
      constexpr int a = compute(5);  // Uses compile-time path
      int b = compute(5);            // Uses runtime path
      std::cout << a << " " << b << "\n";  // Output: 25 25
    }

**Negated form (if !consteval):**

.. code-block:: cpp

    constexpr int optimized(int x) {
      if !consteval {
        // Runtime-only optimizations
        return x * x;
      }
      // Compile-time fallback
      return x * x;
    }

consteval: Immediate Functions (C++20)
--------------------------------------

:Source: `src/constexpr/consteval <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/consteval>`_

C++20 introduced ``consteval`` to declare *immediate functions* that must be
evaluated at compile time. Unlike ``constexpr``, which allows runtime evaluation
when arguments are not constant, ``consteval`` functions produce a compilation
error if called with non-constant arguments. This guarantees that the function
never generates runtime code, making it useful for compile-time-only utilities
like string hashing or lookup table generation.

.. code-block:: cpp

    #include <iostream>

    consteval int square(int x) { return x * x; }

    int main() {
      constexpr int a = square(4);  // OK: compile-time evaluation
      std::cout << a << "\n";       // Output: 16

      // int b = 5;
      // int c = square(b);  // Error: b is not a constant expression
    }

**Comparison of constexpr vs consteval:**

.. code-block:: cpp

    constexpr int ce_square(int x) { return x * x; }
    consteval int cv_square(int x) { return x * x; }

    int main() {
      int runtime_val = 5;

      int a = ce_square(runtime_val);  // OK: runtime evaluation
      // int b = cv_square(runtime_val);  // Error: must be compile-time

      constexpr int c = ce_square(5);  // OK: compile-time
      constexpr int d = cv_square(5);  // OK: compile-time
    }

constinit: Constant Initialization (C++20)
------------------------------------------

:Source: `src/constexpr/constinit <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/constexpr/constinit>`_

C++20 also introduced ``constinit`` to ensure a variable is initialized at
compile time, avoiding the *static initialization order fiasco*. Unlike
``constexpr``, a ``constinit`` variable is not const and can be modified at
runtime. It only guarantees that initialization happens at compile time,
preventing issues where static variables in different translation units
depend on each other's initialization order.

.. code-block:: cpp

    #include <iostream>

    constexpr int compute() { return 42; }

    constinit int global = compute();  // Initialized at compile time

    int main() {
      std::cout << global << "\n";  // Output: 42
      global = 100;                 // OK: constinit doesn't mean const
      std::cout << global << "\n";  // Output: 100
    }

constexpr Evolution by Standard
-------------------------------

**C++11:**

- Introduced ``constexpr`` for simple functions (single return statement)
- ``constexpr`` variables must be initialized with constant expressions

**C++14:**

- ``constexpr`` functions can have multiple statements, loops, and local variables
- ``constexpr`` member functions are no longer implicitly ``const``

**C++17:**

- ``if constexpr`` for compile-time conditional compilation
- ``constexpr`` lambdas

**C++20:**

- ``consteval`` for immediate functions (must evaluate at compile time)
- ``constinit`` for compile-time initialization of non-const variables
- ``constexpr`` virtual functions
- ``constexpr`` dynamic allocation (``new``/``delete``) within constant expressions
- ``constexpr`` ``std::vector`` and ``std::string``

**C++23:**

- ``if consteval`` for detecting constant evaluation context
- Relaxed ``constexpr`` restrictions on non-literal types
