========
Requires
========

.. meta::
   :description: C++20 requires clauses and requires expressions for constraining templates with concepts. Learn how to use requires with boolean flags, nested expressions, conjunction, disjunction, and compile-time branching as an alternative to if constexpr and SFINAE.
   :keywords: C++20, requires, concepts, constraints, templates, SFINAE, if constexpr, compile-time, type traits, generic programming, template metaprogramming

.. contents:: Table of Contents
    :backlinks: none

C++20 introduced the ``requires`` keyword as part of the concepts feature, providing
a cleaner and more expressive way to constrain template parameters. Unlike traditional
SFINAE techniques that rely on template substitution failures, ``requires`` clauses
produce readable compiler error messages and make template constraints explicit in
the function signature. This chapter covers basic requires clauses, nested requires
expressions, logical combinations with conjunction and disjunction, and practical
patterns for compile-time interface selection.

Basic Requires Clause
---------------------

:Source: `src/require/basic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/basic>`_

The ``requires`` clause appears after the template parameter list and before the
function return type. It accepts any constant expression that evaluates to a boolean,
including standard library concepts like ``std::integral``, ``std::floating_point``,
or ``std::copyable``. When the constraint evaluates to false, the template is excluded
from overload resolution entirely, resulting in clear "no matching function" errors
rather than cryptic substitution failure messages.

.. code-block:: cpp

    #include <concepts>

    template <typename T>
      requires std::integral<T>
    T add(T a, T b) {
      return a + b;
    }

    int main() {
      add(1, 2);      // OK: int is integral
      add(10L, 20L);  // OK: long is integral
      // add(1.0, 2.0);  // Error: double is not integral
    }

Nested Requires Expression
--------------------------

:Source: `src/require/nested <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/nested>`_

A *requires expression* allows defining ad-hoc constraints inline without declaring
a separate named concept. The syntax ``requires requires(T x) { ... }`` may look
redundant, but the first ``requires`` introduces the clause while the second begins
the expression body. Inside the body, you can check that expressions compile, verify
return types using ``-> concept`` syntax, and test for noexcept guarantees. This
approach is useful for one-off constraints that don't warrant a reusable concept
definition.

.. code-block:: cpp

    #include <concepts>

    // Ad-hoc constraint: T must support + and * returning T
    template <typename T>
      requires requires(T x) {
        { x + x } -> std::convertible_to<T>;
        { x * x } -> std::convertible_to<T>;
      }
    T square_sum(T a, T b) {
      return (a + b) * (a + b);
    }

    int main() {
      square_sum(2, 3);      // OK: int supports + and *
      square_sum(1.5, 2.5);  // OK: double supports + and *
      // square_sum(std::string("a"), std::string("b"));  // Error: string has no *
    }

Conjunction (&&)
----------------

:Source: `src/require/conjunction <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/conjunction>`_

Multiple constraints can be combined using the logical AND operator ``&&`` to require
that all conditions are satisfied simultaneously. The compiler evaluates constraints
from left to right and short-circuits on the first failure, which can improve compile
times when earlier constraints are cheaper to check. Conjunction is commonly used to
layer additional requirements on top of broad concepts, such as requiring both
``std::integral`` and ``std::signed_integral`` to exclude unsigned types.

.. code-block:: cpp

    #include <concepts>

    // Both constraints must be satisfied
    template <typename T>
      requires std::integral<T> && std::signed_integral<T>
    T abs_val(T x) {
      return x < 0 ? -x : x;
    }

    int main() {
      abs_val(-5);    // OK: int is signed integral
      abs_val(-100L); // OK: long is signed integral
      // abs_val(5u);  // Error: unsigned int is not signed
    }

Disjunction (||)
----------------

:Source: `src/require/disjunction <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/disjunction>`_

Constraints can be combined using the logical OR operator ``||`` to accept types
that satisfy at least one of the specified conditions. This enables writing generic
functions that work across multiple unrelated type categories without code duplication.
For example, a numeric utility function might accept both integral and floating-point
types by combining ``std::integral<T> || std::floating_point<T>``, covering all
built-in arithmetic types with a single template.

.. code-block:: cpp

    #include <concepts>

    // Either constraint can be satisfied
    template <typename T>
      requires std::integral<T> || std::floating_point<T>
    T twice(T x) {
      return x + x;
    }

    int main() {
      twice(5);    // OK: int is integral
      twice(2.5);  // OK: double is floating_point
      // twice(std::string("x"));  // Error: string is neither
    }

Requires vs if constexpr
------------------------

:Source: `src/require/constexpr-like <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/constexpr-like>`_

Both ``if constexpr`` (C++17) and ``requires`` (C++20) enable compile-time branching
based on type properties, but they use different mechanisms. ``if constexpr`` keeps
all code paths in a single function body and discards branches that don't apply,
while ``requires`` uses overload resolution to select between entirely separate
function definitions. The ``requires`` approach can be cleaner when the implementations
differ significantly, and it allows each overload to have different return types or
exception specifications.

.. code-block:: cpp

    #include <type_traits>

    // Using if constexpr
    template <typename T>
    auto get_value_constexpr(T t) {
      if constexpr (std::is_pointer_v<T>) {
        return *t;  // dereference pointer
      } else {
        return t;   // return value directly
      }
    }

    // Using requires (overload resolution)
    template <typename T>
      requires std::is_pointer_v<T>
    auto get_value_requires(T t) {
      return *t;  // selected when T is pointer
    }

    template <typename T>
      requires (!std::is_pointer_v<T>)
    auto get_value_requires(T t) {
      return t;   // selected when T is not pointer
    }

    int main() {
      int x = 42;
      int* p = &x;
      get_value_constexpr(x);  // returns value
      get_value_constexpr(p);  // dereferences pointer
      get_value_requires(x);   // selects non-pointer overload
      get_value_requires(p);   // selects pointer overload
    }

Requires with Boolean Flags
---------------------------

:Source: `src/require/bool-flag <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/require/bool-flag>`_

A powerful design pattern combines ``requires`` with compile-time boolean constants
defined in traits or policy classes. By checking flags like ``Backend::kSupportFoo``,
you can conditionally enable or disable class methods based on backend capabilities.
This technique is widely used in high-performance libraries where different backends
(CPU, GPU, network transports) support different operations. Methods guarded by
``requires`` simply don't exist when the flag is false, preventing accidental use
and enabling clean API design without runtime overhead.

.. code-block:: cpp

    // Backend traits with compile-time feature flags
    struct BackendA {
      static constexpr bool kSupportFoo = true;
      static constexpr bool kSupportBar = true;
    };

    struct BackendB {
      static constexpr bool kSupportFoo = false;
      static constexpr bool kSupportBar = false;
    };

    template <typename Backend>
    class Channel {
     public:
      // Only available when Backend::kSupportFoo is true
      int Foo(int x)
        requires Backend::kSupportFoo
      {
        return x;
      }

      // Only available when Backend::kSupportBar is true
      int Bar()
        requires Backend::kSupportBar
      {
        return 42;
      }

      // Always available
      int Baz(int x) { return x * 2; }
    };

    int main() {
      Channel<BackendA> a;
      a.Foo(10);  // OK: kSupportFoo is true
      a.Bar();    // OK: kSupportBar is true
      a.Baz(5);   // OK: always available

      Channel<BackendB> b;
      // b.Foo(10);  // Error: kSupportFoo is false
      // b.Bar();    // Error: kSupportBar is false
      b.Baz(5);     // OK: always available
    }
