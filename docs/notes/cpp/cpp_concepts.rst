========
Concepts
========

.. meta::
   :description: C++20 concepts tutorial with working code examples. Learn to define custom concepts, use standard library concepts like std::integral and std::invocable, write requires expressions, and understand concept subsumption for cleaner template constraints.
   :keywords: C++20, concepts, constraints, templates, generic programming, type constraints, std::integral, std::same_as, std::convertible_to, requires expression, SFINAE replacement, concept definition, template constraints

.. contents:: Table of Contents
    :backlinks: none

Concepts, introduced in C++20, provide a way to specify constraints on template
parameters directly in the language. They replace verbose SFINAE techniques and
``std::enable_if`` with readable, composable constraints that produce clear compiler
error messages. A concept is essentially a named compile-time boolean predicate that
documents and enforces requirements on template arguments. This guide covers defining
custom concepts, using standard library concepts, writing requires expressions, and
understanding how concept subsumption affects overload resolution.

Defining Concepts
-----------------

:Source: `src/concepts/defining <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/concepts/defining>`_

A concept is defined using the ``concept`` keyword followed by a requires expression
or any compile-time boolean expression. You can build concepts from type traits like
``std::is_arithmetic_v``, from requires expressions that check for valid operations,
or by combining existing concepts with logical operators (``&&``, ``||``). Once defined,
a concept can be used to constrain template parameters, providing clear documentation
of requirements and producing readable error messages when constraints are not satisfied.

.. code-block:: cpp

    #include <concepts>
    #include <type_traits>

    // Simple concept using type traits
    template <typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    // Concept using requires expression
    template <typename T>
    concept Addable = requires(T a, T b) {
      { a + b } -> std::convertible_to<T>;
    };

    // Compound concept
    template <typename T>
    concept Number = Numeric<T> && Addable<T>;

    template <Number T>
    T add(T a, T b) {
      return a + b;
    }

Standard Library Concepts
-------------------------

:Source: `src/concepts/stdlib <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/concepts/stdlib>`_

The ``<concepts>`` header provides a rich set of predefined concepts organized into
categories: core language concepts (``same_as``, ``derived_from``, ``convertible_to``),
comparison concepts (``equality_comparable``, ``totally_ordered``), object concepts
(``movable``, ``copyable``, ``regular``), and callable concepts (``invocable``,
``predicate``). These standard concepts cover common constraints and serve as building
blocks for defining more specific custom concepts.

.. code-block:: cpp

    #include <concepts>

    // Core language concepts
    static_assert(std::same_as<int, int>);
    static_assert(std::derived_from<std::string, std::string>);
    static_assert(std::convertible_to<int, double>);

    // Comparison concepts
    static_assert(std::equality_comparable<int>);
    static_assert(std::totally_ordered<double>);

    // Object concepts
    static_assert(std::movable<std::string>);
    static_assert(std::copyable<int>);
    static_assert(std::regular<int>);

    // Callable concepts
    template <std::invocable<int> F>
    void call_with_42(F&& f) {
      f(42);
    }

Common Standard Concepts
~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------------------------------------+
| Concept                   | Description                              |
+===========================+==========================================+
| ``std::same_as<T, U>``    | T and U are the same type                |
+---------------------------+------------------------------------------+
| ``std::derived_from``     | Derived inherits from Base               |
+---------------------------+------------------------------------------+
| ``std::convertible_to``   | T is implicitly convertible to U         |
+---------------------------+------------------------------------------+
| ``std::integral``         | T is an integral type                    |
+---------------------------+------------------------------------------+
| ``std::floating_point``   | T is a floating-point type               |
+---------------------------+------------------------------------------+
| ``std::invocable<F,Args>``| F can be called with Args                |
+---------------------------+------------------------------------------+
| ``std::predicate<F,Args>``| F returns bool when called with Args     |
+---------------------------+------------------------------------------+

Requires Expressions
--------------------

:Source: `src/concepts/requires-expr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/concepts/requires-expr>`_

Requires expressions provide a powerful way to check compile-time properties of types.
They support four kinds of requirements: simple requirements check that an expression
is valid, type requirements verify that a nested type exists, compound requirements
constrain both validity and return type of an expression, and nested requirements
allow embedding additional boolean constraints. This makes requires expressions ideal
for defining concepts that check interface conformance, like verifying a type has
specific member functions with expected signatures.

.. code-block:: cpp

    template <typename T>
    concept Container = requires(T c) {
      // Simple requirement - expression must be valid
      c.begin();
      c.end();
      c.size();

      // Type requirement - nested type must exist
      typename T::value_type;
      typename T::iterator;

      // Compound requirement - expression with return type constraint
      { c.size() } -> std::convertible_to<std::size_t>;
      { *c.begin() } -> std::same_as<typename T::value_type&>;

      // Nested requirement
      requires std::same_as<decltype(c.begin()), typename T::iterator>;
    };

Concept Syntax Variations
-------------------------

:Source: `src/concepts/syntax <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/concepts/syntax>`_

C++20 provides four equivalent ways to apply concept constraints to templates. The
requires clause after the template parameter list is the most explicit form. The
trailing requires clause places constraints after the function signature. Constrained
template parameters embed the concept directly in the parameter declaration. Finally,
abbreviated function templates use ``auto`` with a concept for the most concise syntax.
All four forms are semantically equivalent, so choose based on readability and team
conventions.

.. code-block:: cpp

    #include <concepts>

    // 1. Requires clause after template
    template <typename T>
      requires std::integral<T>
    T square1(T x) {
      return x * x;
    }

    // 2. Trailing requires clause
    template <typename T>
    T square2(T x) requires std::integral<T> {
      return x * x;
    }

    // 3. Constrained template parameter
    template <std::integral T>
    T square3(T x) {
      return x * x;
    }

    // 4. Abbreviated function template (auto)
    auto square4(std::integral auto x) {
      return x * x;
    }

Concept Subsumption
-------------------

:Source: `src/concepts/subsumption <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/concepts/subsumption>`_

Concept subsumption determines overload resolution when multiple constrained functions
match a call. A concept A subsumes concept B if A's constraints logically imply B's
constraints. When both overloads are viable, the compiler selects the more constrained
one. In the example below, ``Dog`` subsumes ``Animal`` because ``Dog`` requires everything
``Animal`` requires plus additional constraints. This enables writing generic fallback
functions while providing specialized implementations for more specific types.

.. code-block:: cpp

    #include <concepts>
    #include <iostream>

    template <typename T>
    concept Animal = requires(T t) { t.speak(); };

    template <typename T>
    concept Dog = Animal<T> && requires(T t) { t.bark(); };

    void greet(Animal auto& a) {
      std::cout << "Hello, animal\n";
    }

    void greet(Dog auto& d) {
      std::cout << "Hello, dog\n";
      d.bark();
    }

    struct Cat {
      void speak() { std::cout << "Meow\n"; }
    };

    struct Beagle {
      void speak() { std::cout << "Woof\n"; }
      void bark() { std::cout << "Bark!\n"; }
    };

    int main() {
      Cat c;
      Beagle b;
      greet(c);  // "Hello, animal"
      greet(b);  // "Hello, dog" - Dog subsumes Animal
    }

Practical Example: Serializable
-------------------------------

This example demonstrates a practical ``Serializable`` concept that constrains types
to those supporting stream insertion and extraction operators. The concept verifies
that a type can be written to an ``ostream`` and read from an ``istream``, enabling
generic serialization functions that work with any conforming type including built-in
types, ``std::string``, and user-defined types with overloaded stream operators.

.. code-block:: cpp

    #include <concepts>
    #include <sstream>
    #include <string>

    template <typename T>
    concept Serializable = requires(T t, std::ostream& os, std::istream& is) {
      { os << t } -> std::same_as<std::ostream&>;
      { is >> t } -> std::same_as<std::istream&>;
    };

    template <Serializable T>
    std::string to_string(const T& value) {
      std::ostringstream oss;
      oss << value;
      return oss.str();
    }

    template <Serializable T>
    T from_string(const std::string& s) {
      std::istringstream iss(s);
      T value;
      iss >> value;
      return value;
    }

See Also
--------

- :doc:`cpp_requires` - Detailed coverage of requires clauses and expressions
- :doc:`cpp_template` - Template fundamentals
