======
Lambda
======

.. contents:: Table of Contents
    :backlinks: none

Lambda expressions, introduced in C++11, provide a concise way to create
anonymous function objects inline. They are essential for modern C++ programming,
enabling functional programming patterns, simplifying callbacks, and working
seamlessly with STL algorithms. Lambdas eliminate the boilerplate of writing
separate functor classes while offering the same performance characteristics
through compiler-generated closure types.

Lambda Syntax Overview
----------------------

:Source: `src/lambda/syntax <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/syntax>`_

A lambda expression consists of a capture clause, parameter list, optional
specifiers, and a body. The capture clause ``[]`` determines which variables
from the enclosing scope are accessible inside the lambda. The compiler
generates a unique closure type for each lambda, with ``operator()`` containing
the lambda body.

.. code-block:: cpp

    // [capture](parameters) specifiers -> return_type { body }

    #include <iostream>

    int main() {
      int x = 10;

      auto by_value = [x]() { return x * 2; };        // Capture x by value
      auto by_ref = [&x]() { x *= 2; };               // Capture x by reference
      auto all_val = [=]() { return x; };             // Capture all by value
      auto all_ref = [&]() { x = 0; };                // Capture all by reference

      std::cout << by_value() << "\n";  // Output: 20
    }

Callable Objects vs Lambdas
---------------------------

:Source: `src/lambda/callable <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/callable>`_

Before lambdas, callable objects (functors) required defining a class with
``operator()``. Lambdas provide the same functionality with less boilerplate.
The compiler transforms each lambda into an anonymous class where captures
become member variables initialized via constructor, and the lambda body
becomes ``operator()``.

.. code-block:: cpp

    int x = 10;
    auto f = [x](int y) { return x + y; };

The compiler generates something equivalent to:

.. code-block:: cpp

    class __lambda_1 {
      int x;  // Captured variable as member
     public:
      __lambda_1(int x) : x(x) {}  // Constructor initializes capture
      int operator()(int y) const { return x + y; }  // Lambda body
    };

    __lambda_1 f(x);  // Instantiate with captured value

Capture by reference (``[&x]``) stores a reference member instead. This is why
lambdas with captures cannot convert to function pointers—they require storage
for those members.

**Functor vs Lambda comparison:**

.. code-block:: cpp

    #include <functional>
    #include <iostream>

    // Functor approach (pre-C++11)
    class Fib {
     public:
      long operator()(long n) const {
        return (n < 2) ? n : operator()(n - 1) + operator()(n - 2);
      }
    };

    int main() {
      Fib fib;
      std::cout << fib(10) << "\n";  // Output: 55

      // Lambda equivalent (requires std::function for recursion)
      std::function<long(long)> fib_lambda = [&](long n) {
        return (n < 2) ? n : fib_lambda(n - 1) + fib_lambda(n - 2);
      };
      std::cout << fib_lambda(10) << "\n";  // Output: 55
    }

Captureless Lambdas and Function Pointers
-----------------------------------------

:Source: `src/lambda/captureless <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/captureless>`_

Lambdas without captures can implicitly convert to function pointers, making
them compatible with C APIs and legacy code expecting function pointers. This
conversion is only possible when the lambda captures nothing, as captured
variables require storage that function pointers cannot provide.

.. code-block:: cpp

    #include <iostream>

    int main() {
      // Captureless lambda converts to function pointer
      long (*fib)(long) = [](long n) {
        long a = 0, b = 1;
        for (long i = 0; i < n; ++i) {
          long tmp = b;
          b = a + b;
          a = tmp;
        }
        return a;
      };

      std::cout << fib(10) << "\n";  // Output: 55
    }

Mutable Lambdas
---------------

:Source: `src/lambda/mutable <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/mutable>`_

By default, lambdas capturing by value have a const ``operator()``, preventing
modification of captured copies. The ``mutable`` keyword removes this const
qualification, allowing the lambda to modify its captured state across calls.

.. code-block:: cpp

    #include <iostream>

    int main() {
      int counter = 0;

      auto increment = [counter]() mutable {
        return ++counter;  // Modifies the lambda's copy
      };

      std::cout << increment() << "\n";  // Output: 1
      std::cout << increment() << "\n";  // Output: 2
      std::cout << counter << "\n";      // Output: 0 (original unchanged)
    }

Immediately Invoked Lambda Expression (IILE)
--------------------------------------------

:Source: `src/lambda/iile <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/iile>`_

Lambdas can be invoked immediately after definition, useful for complex
initialization of const variables or breaking out of nested loops. This
pattern is called IILE (Immediately Invoked Lambda Expression).

.. code-block:: cpp

    #include <iostream>

    int main() {
      // Complex const initialization
      const int value = []() {
        int result = 0;
        for (int i = 1; i <= 10; ++i) result += i;
        return result;
      }();

      std::cout << value << "\n";  // Output: 55

      // Breaking nested loops
      [&]() {
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 5; ++j) {
            if (i + j == 5) return;  // Breaks both loops
            std::cout << i + j << " ";
          }
        }
      }();
      std::cout << "\n";
    }

Lambda as Callback
------------------

:Source: `src/lambda/callback <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/callback>`_

Lambdas are ideal for callbacks in algorithms and asynchronous operations.
Use templates for zero-overhead callbacks, ``std::function`` when type erasure
is needed, or function pointers for C API compatibility (captureless only).

.. code-block:: cpp

    #include <functional>
    #include <iostream>

    // Template: zero overhead, accepts any callable
    template <typename F>
    void process(int n, F callback) {
      for (int i = 0; i < n; ++i) callback(i);
    }

    // std::function: type erasure, slight overhead
    void process2(int n, std::function<void(int)> callback) {
      for (int i = 0; i < n; ++i) callback(i);
    }

    int main() {
      process(5, [](int x) { std::cout << x << " "; });
      std::cout << "\n";

      int sum = 0;
      process2(5, [&sum](int x) { sum += x; });
      std::cout << sum << "\n";  // Output: 10
    }

Recursive Lambdas
-----------------

:Source: `src/lambda/recursive <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/recursive>`_

Lambdas cannot directly reference themselves. Two common solutions exist:
using ``std::function`` (simpler but slower due to type erasure) or passing
the lambda to itself (faster, approaching normal function performance).

.. code-block:: cpp

    #include <functional>
    #include <iostream>

    int main() {
      // Method 1: std::function (slower)
      std::function<long(long)> fib1 = [&](long n) {
        return n < 2 ? n : fib1(n - 1) + fib1(n - 2);
      };

      // Method 2: Self-passing (faster)
      auto fib2 = [](auto&& self, long n) -> long {
        return n < 2 ? n : self(self, n - 1) + self(self, n - 2);
      };

      std::cout << fib1(20) << "\n";        // Output: 6765
      std::cout << fib2(fib2, 20) << "\n";  // Output: 6765
    }

Init Capture (C++14)
--------------------

:Source: `src/lambda/init-capture <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/init-capture>`_

C++14 introduced *init capture* (also called generalized lambda capture),
allowing you to create new variables in the capture clause with arbitrary
initializers. This enables move-capturing, renaming variables, and capturing
expressions. Init capture is essential for capturing move-only types like
``std::unique_ptr``.

.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <utility>

    int main() {
      auto ptr = std::make_unique<int>(42);

      // Move-capture: transfers ownership into lambda
      auto f = [p = std::move(ptr)]() { std::cout << *p << "\n"; };
      f();  // Output: 42

      // ptr is now nullptr

      // Capture with expression
      int x = 10;
      auto g = [y = x * 2]() { return y; };
      std::cout << g() << "\n";  // Output: 20
    }

Generic Lambdas (C++14)
-----------------------

:Source: `src/lambda/generic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/generic>`_

C++14 introduced generic lambdas with ``auto`` parameters, creating a template
``operator()`` under the hood. This allows a single lambda to work with multiple
types without explicit template syntax. Combined with fold expressions (C++17),
generic lambdas enable powerful variadic operations.

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    int main() {
      // Generic lambda with auto parameters
      auto sum = [](auto&&... args) {
        return (std::forward<decltype(args)>(args) + ...);  // Fold expression
      };

      std::cout << sum(1, 2, 3, 4, 5) << "\n";      // Output: 15
      std::cout << sum(1.5, 2.5, 3.0) << "\n";      // Output: 7.0
    }

constexpr Lambdas (C++17)
-------------------------

:Source: `src/lambda/constexpr-lambda <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/constexpr-lambda>`_

Since C++17, lambdas are implicitly ``constexpr`` when possible, allowing
compile-time evaluation. You can also explicitly mark a lambda ``constexpr``
or ``consteval`` (C++20) to enforce compile-time evaluation.

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

      static_assert(fib(10) == 55);  // Compile-time evaluation
      std::cout << fib(10) << "\n";
    }

Template Lambdas (C++20)
------------------------

:Source: `src/lambda/template-lambda <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/template-lambda>`_

C++20 allows explicit template parameter lists in lambdas, providing more
control than generic lambdas with ``auto``. This enables template constraints,
accessing type information directly, and cleaner syntax when forwarding
arguments. Template lambdas eliminate the need for ``decltype`` workarounds.

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <concepts>

    int main() {
      // Template lambda with explicit parameters
      auto sum = []<typename... Args>(Args&&... args) {
        return (std::forward<Args>(args) + ...);
      };

      std::cout << sum(1, 2, 3, 4, 5) << "\n";  // Output: 15

      // With concepts constraint
      auto add = []<typename T>(T a, T b) requires std::integral<T> {
        return a + b;
      };

      std::cout << add(10, 20) << "\n";  // Output: 30
    }

Default-Constructible Lambdas (C++20)
-------------------------------------

:Source: `src/lambda/default-constructible <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/default-constructible>`_

In C++20, stateless (captureless) lambdas are default-constructible and
assignable. This allows using lambda types directly as template arguments
without passing an instance, simplifying code for containers and algorithms
that require comparators or hash functions.

.. code-block:: cpp

    #include <iostream>
    #include <map>
    #include <set>

    int main() {
      // C++20: Lambda type as template argument, no instance needed
      auto cmp = [](int a, int b) { return a > b; };

      std::set<int, decltype(cmp)> s1;  // C++20: default constructs comparator
      s1.insert({3, 1, 4, 1, 5});

      for (int x : s1) std::cout << x << " ";  // Output: 5 4 3 1
      std::cout << "\n";
    }

Lambdas in Unevaluated Contexts (C++20)
---------------------------------------

:Source: `src/lambda/unevaluated <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/unevaluated>`_

C++20 permits lambdas in unevaluated contexts like ``sizeof``, ``decltype``,
and template arguments. Combined with default construction, this enables
declaring containers with lambda comparators without creating lambda instances.

.. code-block:: cpp

    #include <iostream>
    #include <map>
    #include <string>

    int main() {
      // Lambda in unevaluated context (decltype)
      std::map<int, std::string, decltype([](int a, int b) { return a > b; })> m;

      m[3] = "three";
      m[1] = "one";
      m[2] = "two";

      for (const auto& [k, v] : m) {
        std::cout << k << ": " << v << "\n";
      }
      // Output: 3: three, 2: two, 1: one (descending order)
    }

Pack Capture (C++20)
--------------------

:Source: `src/lambda/pack-capture <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/pack-capture>`_

C++20 allows capturing parameter packs with init-capture, enabling perfect
forwarding of variadic arguments into lambdas for deferred execution.

.. code-block:: cpp

    #include <iostream>
    #include <tuple>
    #include <utility>

    template <typename... Args>
    auto make_delayed_call(Args&&... args) {
      return [...args = std::forward<Args>(args)]() {
        return (args + ...);
      };
    }

    int main() {
      auto delayed = make_delayed_call(1, 2, 3, 4, 5);
      std::cout << delayed() << "\n";  // Output: 15
    }

Deducing this (C++23)
---------------------

:Source: `src/lambda/deducing-this <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/lambda/deducing-this>`_

C++23 introduces *deducing this*, allowing lambdas to take an explicit object
parameter. This enables recursive lambdas without ``std::function`` or
self-passing, and provides access to the lambda's own type for advanced patterns.

.. code-block:: cpp

    #include <iostream>

    int main() {
      // C++23: Recursive lambda with deducing this
      auto fib = [](this auto&& self, long n) -> long {
        return n < 2 ? n : self(n - 1) + self(n - 2);
      };

      std::cout << fib(20) << "\n";  // Output: 6765
    }

Lambda Attributes (C++23)
-------------------------

C++23 allows attributes on the lambda's ``operator()``, enabling optimizations
and compiler hints like ``[[nodiscard]]``, ``[[likely]]``, and ``[[deprecated]]``.

.. code-block:: cpp

    #include <iostream>

    int main() {
      auto compute = []() [[nodiscard]] { return 42; };

      // compute();  // Warning: ignoring return value with [[nodiscard]]
      int x = compute();  // OK
      std::cout << x << "\n";
    }

Lambda Evolution by Standard
----------------------------

**C++11:**

- Basic lambda syntax with capture, parameters, and body
- Capture by value ``[=]`` and by reference ``[&]``

**C++14:**

- Generic lambdas with ``auto`` parameters
- Init capture (generalized lambda capture)
- Return type deduction for all lambdas

**C++17:**

- ``constexpr`` lambdas (implicit when possible)
- Capture of ``*this`` by value

**C++20:**

- Template parameter lists ``[]<typename T>(T x) {}``
- Default-constructible and assignable stateless lambdas
- Lambdas in unevaluated contexts
- Pack capture with init-capture
- Lambdas with concepts constraints

**C++23:**

- Deducing this for explicit object parameters
- Attributes on lambda ``operator()``
- Simplified syntax improvements
