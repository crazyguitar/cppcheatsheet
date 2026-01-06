===================
Template
===================

.. meta::
   :description: C++ template programming guide covering function templates, class templates, variadic templates, SFINAE, and concepts.
   :keywords: C++, template, generic programming, variadic template, SFINAE, concepts, template specialization, metaprogramming

.. contents:: Table of Contents
    :backlinks: none

C++ templates enable generic programming by allowing functions and classes to
operate with different data types without rewriting code. Templates are a
cornerstone of modern C++ development, powering the Standard Template Library
(STL) and enabling type-safe, high-performance abstractions. Understanding
templates is essential for writing reusable, efficient C++ code.

Explicit Template Instantiation
-------------------------------

:Source: `src/template/instantiate <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/instantiate>`_

Explicit template instantiation forces the compiler to generate code for a
specific template specialization at a particular point in the source. This
technique is particularly valuable in large codebases where the same template
is used across multiple translation units. By explicitly instantiating templates
in a single source file, you can significantly reduce compile times and control
where template code is generated. This approach is commonly used in library
development to reduce binary size and improve build performance.

**Key benefits:** Reduced compilation time, controlled symbol generation,
smaller binary size when templates are used across many files.

.. code-block:: cpp

    #include <iostream>

    struct A {};
    struct B {};

    template <typename T, typename U>
    struct Foo {
      Foo(T t, U u) : t_(t), u_(u) {}
      T t_;
      U u_;
    };

    template <typename F, typename T, typename U>
    struct Bar {
      Bar(T t, U u) : f_(t, u) {}
      F f_;
    };

    // explicit instantiation
    template class Foo<A, B>;

    int main() {
      Bar<Foo<A, B>, A, B>(A(), B());
    }

Template Specialization
-----------------------

:Source: `src/template/specialization <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/specialization>`_

Template specialization allows customizing template behavior for specific types,
enabling optimized or alternative implementations for particular type combinations.
This is one of the most powerful features of C++ templates, allowing generic
code to have type-specific optimizations without sacrificing the generic interface.

**Partial specialization** fixes some template parameters while leaving others
generic. This is useful when you want specialized behavior for a category of
types (e.g., all pointers, all containers with a specific allocator).

**Full specialization** provides a complete implementation for specific types,
replacing the primary template entirely for those types. This enables highly
optimized implementations for known types.

.. code-block:: cpp

    #include <iostream>

    template <typename T, typename U>
    class Base {
      T m_a;
      U m_b;
    public:
      Base(T a, U b) : m_a(a), m_b(b) {}
      T foo() { return m_a; }
      U bar() { return m_b; }
    };

    // partial specialization - second parameter fixed to int
    template<typename T>
    class Base<T, int> {
      T m_a;
      int m_b;
    public:
      Base(T a, int b) : m_a(a), m_b(b) {}
      T foo() { return m_a; }
      int bar() { return m_b; }
    };

    // full specialization - both parameters fixed
    template<>
    class Base<double, double> {
      double d_a;
      double d_b;
    public:
      Base(double a, double b) : d_a(a), d_b(b) {}
      double foo() { return d_a; }
      double bar() { return d_b; }
    };

    int main() {
      Base<float, int> foo(3.33, 1);           // uses partial specialization
      Base<double, double> bar(55.66, 95.27);  // uses full specialization
      std::cout << foo.foo() << "\n";
      std::cout << bar.bar() << "\n";
    }

Class Template Inheritance
--------------------------

:Source: `src/template/class-template <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/class-template>`_

Class templates can serve as base classes for both regular classes and other
templates, enabling powerful code reuse patterns. This flexibility allows you
to build inheritance hierarchies that combine generic and concrete types.

A non-template class can inherit from a specific template instantiation, which
is useful when you need a concrete class with a fixed type but want to reuse
template-based functionality. A template class can also inherit from a template
base, preserving generic behavior through the entire inheritance hierarchy.

**Important:** When inheriting from a template base class, dependent names
(names that depend on template parameters) require special handling due to
two-phase name lookup rules.

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class Area {
    protected:
      T w, h;
    public:
      Area(T a, T b) : w(a), h(b) {}
      T get() { return w * h; }
    };

    // non-template inherits from specific instantiation
    class Rectangle : public Area<int> {
    public:
      Rectangle(int a, int b) : Area<int>(a, b) {}
    };

    // template inherits from template base
    template <typename T>
    class GenericRectangle : public Area<T> {
    public:
      GenericRectangle(T a, T b) : Area<T>(a, b) {}
    };

    int main() {
      Rectangle r(2, 5);
      GenericRectangle<double> g(2.5, 3.0);
      std::cout << r.get() << "\n";
      std::cout << g.get() << "\n";
    }

Variadic Templates and Parameter Packs
--------------------------------------

:Source: `src/template/variadic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/variadic>`_

Variadic templates accept an arbitrary number of template arguments using
parameter packs (``...``). This feature, introduced in C++11, revolutionized
generic programming by enabling functions and classes that work with any
number of arguments of any types. Variadic templates are the foundation of
many modern C++ idioms including ``std::tuple``, ``std::make_unique``, and
perfect forwarding utilities.

**Parameter pack expansion** is the mechanism by which the compiler generates
code for each element in the pack. Before C++17, this required recursive
function calls. C++17 introduced fold expressions for more concise and often
more efficient pack expansion.

**Recursive expansion (C++11):**

The classic approach uses function overloading with a base case (single argument)
and a recursive case that peels off one argument at a time:

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    T sum(T x) { return x; }

    template <typename T, typename ...Args>
    T sum(T x, Args ...args) { return x + sum(args...); }

    int main() {
      std::cout << sum(1, 2, 3, 4, 5) << "\n";  // 15
    }

**Fold expression (C++17):**

Fold expressions provide a more elegant and often more efficient way to expand
parameter packs using binary operators:

.. code-block:: cpp

    #include <iostream>

    template <typename ...Args>
    auto sum(Args ...args) { return (args + ...); }

    int main() {
      std::cout << sum(1, 2, 3, 4, 5) << "\n";  // 15
    }

**Variadic class constructor:**

Variadic templates combined with perfect forwarding enable constructors that
accept any number of arguments and forward them efficiently. This pattern is
used extensively in the standard library for in-place construction:

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <vector>

    template <typename T>
    class Vector {
      std::vector<T> v;
    public:
      template<typename ...Args>
      Vector(Args&&... args) {
        (v.emplace_back(std::forward<Args>(args)), ...);
      }

      auto begin() { return v.begin(); }
      auto end() { return v.end(); }
    };

    int main() {
      Vector<int> v{1, 2, 3};
      for (auto x : v) std::cout << x << "\n";
    }

Fold Expressions (C++17)
------------------------

:Source: `src/template/fold <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/fold>`_

Fold expressions reduce parameter packs using binary operators, providing a
concise syntax for operations that would otherwise require recursive templates.
They significantly simplify variadic template code and can generate more
efficient code by avoiding recursive function calls.

**Syntax forms:**

- ``(args op ...)`` - unary right fold: ``a op (b op (c op d))``
- ``(... op args)`` - unary left fold: ``((a op b) op c) op d``
- ``(init op ... op args)`` - binary right fold with initial value
- ``(args op ... op init)`` - binary left fold with initial value

Fold expressions work with any binary operator including the comma operator
for side effects and ``<<`` for stream output. This makes them extremely
versatile for a wide range of use cases.

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      // unary right fold: (args op ...)
      auto sum = [](auto ...args) { return (args + ...); };
      std::cout << sum(1, 2, 3, 4, 5) << "\n";  // 15

      // fold with comma operator for side effects
      std::vector<int> v;
      [&v](auto ...args) { (v.emplace_back(args), ...); }(1, 2, 3);

      // fold with stream operator
      [](auto ...args) {
        (std::cout << ... << args) << "\n";
      }(1, 2, 3, 4, 5);  // 12345

      // fold with transformation
      auto double_sum = [](auto &&f, auto ...args) {
        return (... + f(args));
      };
      std::cout << double_sum([](auto x) { return x * 2; }, 1, 2, 3) << "\n";  // 12
    }

SFINAE and Type Constraints
---------------------------

:Source: `src/template/sfinae <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/sfinae>`_

SFINAE (Substitution Failure Is Not An Error) is a fundamental C++ template
mechanism that enables conditional template instantiation based on type
properties. When the compiler attempts to substitute template arguments and
the substitution results in an invalid type, instead of producing a compilation
error, the compiler simply removes that template from the overload set.

**This technique is fundamental for writing type-safe generic code** that only
accepts types meeting certain requirements. Combined with type traits from
``<type_traits>``, SFINAE enables powerful compile-time type checking and
conditional compilation.

Using ``std::enable_if`` with type traits restricts which types can instantiate
a template. This is the classic SFINAE approach, though C++20 concepts provide
a more readable alternative for new code.

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <type_traits>

    template<typename S,
      typename = std::enable_if_t<
        std::is_same_v<std::string, std::decay_t<S>>
      >
    >
    void Foo(S s) {
      std::cout << s << "\n";
    }

    int main() {
      std::string s1 = "Foo";
      const std::string s2 = "Bar";
      Foo(s1);
      Foo(s2);
      // Foo(123);    // compile error
      // Foo("Baz");  // compile error
    }

**Template specialization approach:**

An alternative to SFINAE is providing explicit specializations for supported
types, leaving the primary template empty or deleted. This approach is simpler
to understand but less flexible than SFINAE for complex type constraints:

.. code-block:: cpp

    #include <iostream>
    #include <string>

    template<typename S>
    void Foo(S s) {}

    template <>
    void Foo<int>(int s) {
      std::cout << "integer: " << s << "\n";
    }

    template<>
    void Foo<std::string>(std::string s) {
      std::cout << "string: " << s << "\n";
    }

    int main() {
      Foo(123);
      Foo(std::string("hello"));
    }

CRTP (Curiously Recurring Template Pattern)
-------------------------------------------

:Source: `src/template/crtp <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/crtp>`_

CRTP is a powerful C++ idiom where a class derives from a template instantiated
with itself as the argument. This enables static polymorphism, allowing base
class methods to call derived class implementations without the runtime overhead
of virtual functions. CRTP is widely used for mixin classes, compile-time
polymorphism, and implementing the "template method" design pattern with
zero runtime cost.

**Understanding the trade-offs between virtual functions and CRTP is crucial**
for making informed design decisions in performance-sensitive code.

**Virtual function (dynamic polymorphism):**

Traditional runtime polymorphism uses virtual functions and vtable lookup.
Each object with virtual functions contains a hidden pointer to a vtable,
and each virtual call requires an indirect function call through this table.
This approach supports runtime type decisions and is essential when the
concrete type is not known until runtime, but it incurs overhead from
indirect function calls and prevents the compiler from inlining:

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    class Base {
    public:
      virtual void impl() { std::cout << "Base\n"; }
      virtual ~Base() = default;
    };

    class Derived : public Base {
    public:
      void impl() override { std::cout << "Derived\n"; }
    };

    int main() {
      std::unique_ptr<Base> p = std::make_unique<Derived>();
      p->impl();  // runtime dispatch via vtable
    }

**CRTP (static polymorphism):**

CRTP resolves the derived type at compile time through template instantiation.
The base class uses ``static_cast`` to access the derived class, which is
safe because the template parameter guarantees the relationship. No vtable
is needed, enabling the compiler to inline the derived implementation and
eliminating all virtual call overhead:

.. code-block:: cpp

    #include <iostream>

    template <typename D>
    class Base {
    public:
      void interface() { static_cast<D*>(this)->impl(); }
      void impl() { std::cout << "Base\n"; }
    };

    class Derived : public Base<Derived> {
    public:
      void impl() { std::cout << "Derived\n"; }
    };

    int main() {
      Derived d;
      d.interface();  // compile-time dispatch, no vtable
    }

**Comparison:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Virtual Functions
     - CRTP
   * - Dispatch
     - Runtime (vtable)
     - Compile-time (static)
   * - Performance
     - Indirect call overhead
     - Inlinable, zero overhead
   * - Flexibility
     - Runtime type switching
     - Types fixed at compile time
   * - Memory
     - vtable pointer per object
     - No extra memory
   * - Binary size
     - Smaller (shared vtable)
     - Larger (template instantiation)

**When to use each:** Use virtual functions when you need runtime polymorphism
(e.g., plugin systems, heterogeneous containers). Use CRTP when types are
known at compile time and performance is critical (e.g., numeric libraries,
game engines, embedded systems).

Template Template Parameters
----------------------------

:Source: `src/template/template-template <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/template-template>`_

Template template parameters allow templates to accept other templates as
arguments, rather than just types or values. This enables writing highly
generic functions and classes that work with any container type or any
template following a specific signature, providing maximum flexibility
for generic algorithms.

**This feature is essential** for writing container-agnostic algorithms
and for implementing policy-based design patterns where behaviors are
injected via template parameters.

.. code-block:: cpp

    #include <vector>
    #include <deque>
    #include <iostream>

    template <template<class, class> class V, class T, class A>
    void pop(V<T, A> &v) {
      std::cout << v.back() << "\n";
      v.pop_back();
    }

    int main() {
      std::vector<int> v{1, 2, 3};
      std::deque<int> q{4, 5, 6};
      pop(v);  // 3
      pop(q);  // 6
    }

Accessing Protected Members in Derived Templates
------------------------------------------------

:Source: `src/template/protected-access <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/protected-access>`_

When a template class inherits from another template, protected members of
the base class are not directly visible in the derived class. This behavior
is due to C++'s two-phase name lookup rules: names that depend on template
parameters are not looked up until the template is instantiated.

**This is a common source of confusion** for developers new to template
programming. The compiler cannot know at parse time whether ``p_`` refers
to a base class member or something else, so it requires explicit qualification.

Two solutions exist: using a ``using`` declaration to bring names into the
current scope, or qualifying access through the ``this`` pointer to make
the name dependent:

**Using declaration:**

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class A {
    protected:
      T p_;
    public:
      A(T p) : p_(p) {}
    };

    template <typename T>
    class B : A<T> {
      using A<T>::p_;  // bring into scope
    public:
      B(T p) : A<T>(p) {}
      void print() { std::cout << p_ << "\n"; }
    };

    int main() {
      B<int> b(42);
      b.print();
    }

**Via this pointer:**

.. code-block:: cpp

    template <typename T>
    class B : A<T> {
    public:
      B(T p) : A<T>(p) {}
      void print() { std::cout << this->p_ << "\n"; }
    };

Compile-Time Loop Unrolling
---------------------------

:Source: `src/template/loop-unroll <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/loop-unroll>`_

Template metaprogramming enables compile-time loop unrolling through recursive
template instantiation. The compiler generates fully unrolled code at compile
time, completely eliminating loop overhead including the loop counter, condition
check, and branch instruction. This technique is particularly valuable for
performance-critical code with known iteration counts, such as matrix operations,
SIMD processing, and embedded systems programming.

**The trade-off is increased binary size** due to code duplication, but for
small, hot loops, the performance benefit often outweighs this cost. Modern
compilers may unroll loops automatically, but template-based unrolling gives
you explicit control.

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    template <size_t N>
    struct Loop {
      template <typename F>
      static void run(F &&f) {
        Loop<N-1>::run(std::forward<F>(f));
        f(N-1);
      }
    };

    template <>
    struct Loop<0> {
      template <typename F>
      static void run(F&&) {}
    };

    int main() {
      size_t sum = 0;
      Loop<5>::run([&](auto i) { sum += i; });
      std::cout << sum << "\n";  // 10 (0+1+2+3+4)
    }
