============
C++ Basics
============

.. meta::
   :description: C++ fundamentals covering syntax, types, operators, control flow, functions, classes, and memory management basics.
   :keywords: C++, basics, fundamentals, syntax, types, operators, functions, classes, pointers, references

.. contents:: Table of Contents
    :backlinks: none

std::print (C++23)
------------------

:Source: `src/basic/print <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/print>`_

C++23 introduces ``std::print`` and ``std::println`` from the ``<print>`` header,
bringing Python-style formatted output to C++. This is the modern way to print
in C++, replacing the traditional ``std::cout`` approach.

.. code-block:: cpp

    #include <print>

    int main() {
      std::print("Hello, World!");    // No newline
      std::println("Hello, World!");  // With newline
    }

The old way using ``std::cout``:

.. code-block:: cpp

    #include <iostream>

    int main() {
      std::cout << "Hello, World!" << std::endl;
    }

C Linkage and Name Mangling
---------------------------

:Source: `src/basic/c-linkage <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/c-linkage>`_

In C++, the compiler applies *name mangling* to function symbols to support
function overloading. Name mangling encodes function signatures (parameter types,
namespaces, and template arguments) into the symbol name, allowing multiple
functions with the same name but different signatures to coexist. When interfacing
with C libraries or exposing functions to C code, ``extern "C"`` disables name
mangling to ensure binary compatibility across different compilers and languages.

**With extern "C" (C linkage):**

Using ``nm`` to inspect the symbol table, we can observe that the ``fib``
function retains its original name without mangling. The symbol ``_fib``
follows C naming conventions, making it accessible from C code or other
languages that use C-compatible calling conventions:

.. code-block:: cpp

    #include <iostream>

    #ifdef __cplusplus
    extern "C" {
    #endif

    int fib(int n) {
      int a = 0, b = 1;
      for (int i = 0; i < n; ++i) {
        int tmp = b;
        b = a + b;
        a = tmp;
      }
      return a;
    }

    #ifdef __cplusplus
    }
    #endif

    int main(int argc, char *argv[]) {
      std::cout << fib(10) << "\n";  // Output: 55
    }

.. code-block:: bash

    $ g++ -std=c++17 -Wall -Werror -O3 a.cc
    $ nm -g a.out | grep fib
    0000000100003a58 T _fib    # C-style symbol (no mangling)

**Without extern "C" (C++ linkage):**

Without the ``extern "C"`` wrapper, the compiler mangles the function name
to encode type information for overload resolution. The mangled symbol
``__Z3fibi`` follows the Itanium C++ ABI naming scheme: ``_Z`` indicates a
mangled name, ``3fib`` represents the function name with its length prefix,
and ``i`` denotes an ``int`` parameter. This encoding enables the linker to
distinguish between overloaded functions:

.. code-block:: cpp

    #include <iostream>

    int fib(int n) {
      int a = 0, b = 1;
      for (int i = 0; i < n; ++i) {
        int tmp = b;
        b = a + b;
        a = tmp;
      }
      return a;
    }

    int main(int argc, char *argv[]) {
      std::cout << fib(10) << "\n";
    }

.. code-block:: bash

    $ nm -g a.out | grep fib
    0000000100003a58 T __Z3fibi    # Mangled symbol: _Z3fibi

**Platform-specific macros:**

On BSD and macOS systems, ``<sys/cdefs.h>`` provides ``__BEGIN_DECLS`` and
``__END_DECLS`` as portable alternatives to ``extern "C"``. These macros
expand to the appropriate linkage specification when compiled as C++ and
to nothing when compiled as C, simplifying header files shared between
both languages:

.. code-block:: cpp

    #include <iostream>
    #include <sys/cdefs.h>

    __BEGIN_DECLS
    int fib(int n) { /* ... */ }
    __END_DECLS

Uniform Initialization (Brace Initialization)
---------------------------------------------

:Source: `src/basic/uniform-initialization <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/uniform-initialization>`_

Introduced in C++11, *uniform initialization* (also known as *brace initialization*)
provides a consistent syntax for initializing objects of any type. This syntax
works for primitives, aggregates, containers, and user-defined types. However,
the compiler prioritizes ``std::initializer_list`` constructors when applicable,
which can lead to surprising behavior if not understood properly.

**Initializer list takes precedence:**

When both a direct constructor and an ``std::initializer_list`` constructor
are viable, the compiler strongly prefers the initializer list version. In this
example, ``Widget{10, 5.0}`` invokes the initializer list constructor even though
``Widget(int, double)`` appears to be a better match. The values ``10`` and ``5.0``
are implicitly converted to ``long double`` and passed as a two-element initializer
list. This behavior is mandated by the C++ standard to ensure consistent semantics
for brace initialization:

.. code-block:: cpp

    #include <iostream>
    #include <initializer_list>

    class Widget {
     public:
      Widget(int a, double b) { std::cout << "Direct constructor\n"; }

      Widget(std::initializer_list<long double> il) { std::cout << "Initializer list constructor\n"; }
    };

    int main(int argc, char *argv[]) {
      Widget w{10, 5.0};  // Output: Initializer list constructor
    }

**Narrowing conversions are prohibited:**

Brace initialization prevents implicit narrowing conversions, providing an
additional layer of type safety compared to parentheses initialization. The
following code produces a compilation error because ``int`` and ``double``
cannot implicitly narrow to ``bool`` without potential data loss. This
protection helps catch bugs at compile time that might otherwise cause
subtle runtime errors:

.. code-block:: cpp

    #include <initializer_list>

    class Widget {
     public:
      Widget(int a, double b) {}
      Widget(std::initializer_list<bool> il) {}
    };

    int main(int argc, char *argv[]) {
      Widget w{10, 5.0};  // Compilation error: narrowing conversion
    }

**Fallback to direct constructor:**

When no valid conversion exists for the initializer list element type,
the compiler falls back to the matching direct constructor. Here, ``int``
and ``double`` cannot convert to ``std::string`` (there is no implicit
conversion path), so the compiler bypasses the initializer list constructor
entirely and selects the direct constructor instead:

.. code-block:: cpp

    #include <iostream>
    #include <initializer_list>
    #include <string>

    class Widget {
     public:
      Widget(int a, double b) { std::cout << "Direct constructor\n"; }

      Widget(std::initializer_list<std::string> il) { std::cout << "Initializer list constructor\n"; }
    };

    int main(int argc, char *argv[]) {
      Widget w{10, 5.0};  // Output: Direct constructor
    }

Pointer Arithmetic and Negative Indices
---------------------------------------

:Source: `src/basic/negative-array-index <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/negative-array-index>`_

In C++, the subscript operator ``[]`` is defined as pointer arithmetic:
``arr[i]`` is equivalent to ``*(arr + i)``. This equivalence is fundamental
to how arrays work in C and C++, and it allows negative indices when the
pointer references an element beyond the array's beginning. While this
flexibility is powerful, it requires careful bounds management to avoid
undefined behavior.

In this example, ``ptr`` points to ``arr[1]``, the second element of the array.
Using ``ptr[-1]`` computes ``*(ptr - 1)``, which accesses ``arr[0]``. This
technique is commonly used in algorithms that need to look backward from a
current position, such as insertion sort or string parsing:

.. code-block:: cpp

    #include <iostream>

    int main(int argc, char *argv[]) {
      int arr[] = {1, 2, 3};
      int *ptr = &arr[1];  // Points to second element

      std::cout << ptr[-1] << "\n";  // Output: 1 (arr[0])
      std::cout << ptr[0] << "\n";   // Output: 2 (arr[1])
      std::cout << ptr[1] << "\n";   // Output: 3 (arr[2])
    }

Template Type Deduction
-----------------------

:Source: `src/basic/reference-collapsing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/reference-collapsing>`_

Understanding how template parameters deduce types is essential for writing
generic C++ code. The deduction rules differ based on the parameter declaration,
and mastering these rules is crucial for effective use of templates, ``auto``,
and perfect forwarding. The following three examples demonstrate how the same
arguments produce different deduced types depending on whether the parameter
is an lvalue reference, universal reference, or value.

**Lvalue reference parameters (T&):**

When the parameter is an lvalue reference, the deduced type ``T`` preserves
const-qualifiers from the argument, but the reference itself is not part of ``T``.
This means passing a ``const int`` deduces ``T`` as ``const int``, and the
parameter type becomes ``const int&``. This behavior ensures that const-correctness
is maintained through template instantiation:

.. code-block:: cpp

    template <typename T>
    void f(T& param) noexcept {}

    int main(int argc, char *argv[]) {
      int x = 123;
      const int cx = x;
      const int& rx = x;

      f(x);   // T = int,       param: int&
      f(cx);  // T = const int, param: const int&
      f(rx);  // T = const int, param: const int&
    }

**Universal reference parameters (T&&):**

Universal references (also called forwarding references) behave differently
for lvalues and rvalues, making them the foundation of perfect forwarding.
When passed an lvalue, ``T`` deduces to an lvalue reference type (e.g., ``int&``),
and reference collapsing produces an lvalue reference parameter. When passed
an rvalue, ``T`` deduces to a non-reference type, and the parameter becomes
an rvalue reference. This dual behavior allows a single function template to
accept both lvalues and rvalues:

.. code-block:: cpp

    template <typename T>
    void f(T&& param) noexcept {}

    int main(int argc, char *argv[]) {
      int x = 123;
      const int cx = x;
      const int& rx = x;

      f(x);   // x is lvalue:  T = int&,       param: int&
      f(cx);  // cx is lvalue: T = const int&, param: const int&
      f(rx);  // rx is lvalue: T = const int&, param: const int&
      f(12);  // 12 is rvalue: T = int,        param: int&&
    }

**Value parameters (T):**

When the parameter is passed by value, the argument is copied, and both
references and top-level const-qualifiers are stripped from the deduced type.
This occurs because the function receives an independent copy that can be
modified without affecting the original. The decay behavior mirrors what
happens when you assign to a new variable of type ``auto``:

.. code-block:: cpp

    template <typename T>
    void f(T param) noexcept {}

    int main(int argc, char *argv[]) {
      int x = 123;
      const int cx = x;
      const int& rx = x;

      f(x);   // T = int, param: int (copy)
      f(cx);  // T = int, param: int (const dropped)
      f(rx);  // T = int, param: int (reference and const dropped)
      f(12);  // T = int, param: int
    }

Auto Type Deduction
-------------------

The ``auto`` keyword follows the same deduction rules as template value
parameters, with one notable exception: braced initializers. This example
demonstrates how ``auto`` deduces types for different initializers, including
the special behavior of ``auto&&`` as a universal reference. Understanding
these rules is essential for writing modern C++ code that leverages type
inference effectively:

.. code-block:: cpp

    int main() {
      auto x = 123;        // int
      const auto cx = x;   // const int
      const auto& rx = x;  // const int&

      auto&& urx = x;   // int& (x is lvalue)
      auto&& urcx = cx; // const int& (cx is lvalue)
      auto&& urrx = rx; // const int& (rx is lvalue)
      auto&& urrv = 12; // int&& (12 is rvalue)
    }

decltype(auto) Type Deduction
-----------------------------

Unlike ``auto``, ``decltype(auto)`` preserves the exact type including
references and cv-qualifiers. This distinction is critical when the original
type information must be retained, such as when forwarding return values
from wrapped functions. While ``auto`` applies template argument deduction
rules (which strip references), ``decltype(auto)`` applies ``decltype``
semantics to the initializer expression.

The first example contrasts ``decltype(auto)`` with ``auto``. Note how
``auto`` strips the reference and const from ``crx``, producing a plain ``int``,
while ``decltype(auto)`` preserves the full ``const int&`` type. Similarly,
``decltype(auto)`` preserves rvalue references, which ``auto`` would decay
to a value type:

.. code-block:: cpp

    #include <type_traits>

    int main(int argc, char *argv[]) {
      int x = 0;
      const int cx = x;
      const int& crx = x;
      int&& z = 0;

      // decltype(auto) preserves cv-qualifiers and references
      decltype(auto) y1 = crx;
      static_assert(std::is_same_v<const int&, decltype(y1)>);

      // auto strips cv-qualifiers and references
      auto y2 = crx;
      static_assert(std::is_same_v<int, decltype(y2)>);

      // decltype(auto) preserves rvalue references
      decltype(auto) z1 = std::move(z);
      static_assert(std::is_same_v<int&&, decltype(z1)>);
    }

**Application in return type deduction:**

This behavior is particularly useful for generic functions that must preserve
the exact return type of an expression. In this example, ``foo`` uses ``auto``
return type deduction, which strips the reference and returns by value. In
contrast, ``bar`` uses ``decltype(auto)``, preserving the ``const int&`` return
type. This distinction matters for performance (avoiding copies) and semantics
(maintaining reference semantics):

.. code-block:: cpp

    #include <type_traits>

    auto foo(const int& x) {
      return x;  // Returns int (reference stripped)
    }

    decltype(auto) bar(const int& x) {
      return x;  // Returns const int& (reference preserved)
    }

    int main(int argc, char *argv[]) {
      static_assert(std::is_same_v<int, decltype(foo(1))>);
      static_assert(std::is_same_v<const int&, decltype(bar(1))>);
    }

Reference Collapsing Rules
--------------------------

When references to references occur during template instantiation or type
aliasing, the compiler applies *reference collapsing* rules. These rules
are fundamental to understanding how universal references and ``std::forward``
work. In C++, you cannot directly declare a reference to a reference, but
such types can arise indirectly through template instantiation or type aliases:

.. code-block:: text

    T& &   -> T&
    T& &&  -> T&
    T&& &  -> T&
    T&& && -> T&&

The rule can be summarized as: **lvalue reference always wins**. Only when
both references are rvalue references does the result remain an rvalue reference.
This mechanism enables perfect forwarding by allowing a single function template
to handle both lvalues and rvalues correctly. When an lvalue is passed to a
universal reference parameter, ``T`` deduces to an lvalue reference, and
reference collapsing ensures the parameter type is also an lvalue reference.

Perfect Forwarding with std::forward
------------------------------------

:Source: `src/basic/perfect-forwarding <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/perfect-forwarding>`_

*Perfect forwarding* preserves the value category (lvalue/rvalue) of function
arguments when passing them to other functions. This technique is essential
for writing wrapper functions, factory functions, and generic code that must
not alter the semantics of the forwarded arguments. Perfect forwarding combines
universal references with ``std::forward`` and relies on the reference collapsing
rules described above.

**Implementation of std::forward:**

The standard library's ``std::forward`` uses ``static_cast`` combined with
reference collapsing to conditionally cast to an rvalue reference. The first
overload handles lvalue arguments: when ``T`` is an lvalue reference type,
``T&&`` collapses to an lvalue reference. The second overload handles rvalue
arguments and includes a static assertion to prevent forwarding an rvalue
as an lvalue:

.. code-block:: cpp

    #include <type_traits>

    template <typename T>
    T&& forward(std::remove_reference_t<T>& t) noexcept {
      return static_cast<T&&>(t);
    }

    template <typename T>
    T&& forward(std::remove_reference_t<T>&& t) noexcept {
      static_assert(!std::is_lvalue_reference_v<T>);
      return static_cast<T&&>(t);
    }

**Wrapper function:**

This example demonstrates a wrapper function that forwards arguments to
a callable while preserving their value category. When ``d`` (an lvalue)
is passed, ``T`` deduces to ``Data&``, and ``std::forward`` returns an lvalue
reference. When ``std::move(t)`` is passed, ``T`` deduces to ``Data``, and
``std::forward`` returns an rvalue reference:

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    template <typename T, typename Func>
    void wrapper(T&& arg, Func fn) {
      fn(std::forward<T>(arg));
    }

    struct Data { int x, y, result; };

    int main() {
      Data d{1, 2, 0};
      wrapper(d, [](Data& d) { d.result = d.x + d.y; });
      std::cout << d.result << "\n";  // 3

      Data t{5, 6, 0};
      wrapper(std::move(t), [](Data&& d) { d.result = d.x * d.y; });
      std::cout << t.result << "\n";  // 30
    }

**Decorator pattern:**

A timing decorator that wraps any callable and measures execution time while
perfectly forwarding all arguments to the wrapped function:

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <chrono>

    template <typename Func, typename ...Args>
    auto timed(Func &&f, Args&&... args) {
      auto start = std::chrono::system_clock::now();
      auto ret = f(std::forward<Args>(args)...);
      std::chrono::duration<double> d = std::chrono::system_clock::now() - start;
      std::cout << "Time: " << d.count() << "s\n";
      return ret;
    }

    long fib(long n) { return n < 2 ? n : fib(n-1) + fib(n-2); }

    int main() { timed(fib, 35); }

**Factory pattern:**

A generic factory function that constructs objects by forwarding arguments
to constructors, preserving value categories for efficient object creation:

.. code-block:: cpp

    #include <memory>
    #include <utility>

    template <typename T, typename ...Args>
    std::unique_ptr<T> make(Args&&... args) {
      return std::make_unique<T>(std::forward<Args>(args)...);
    }

    struct Widget { int x; double y; };

    int main() { auto w = make<Widget>(42, 3.14); }

Bit Manipulation with std::bitset
---------------------------------

:Source: `src/basic/bit-manipulation <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/bit-manipulation>`_

The ``std::bitset`` class template provides a fixed-size sequence of bits
with convenient member functions for bit manipulation. Unlike raw integer
bit operations, ``std::bitset`` offers type safety, clear semantics, and
bounds checking in debug builds. The template parameter specifies the number
of bits, which is fixed at compile time.

This example creates a 4-bit bitset initialized to 8 (binary ``1000``).
The ``count()`` member function returns the number of set bits (population
count), which is useful for algorithms that need to count flags or compute
Hamming weights. The equality operator allows direct comparison with integer
values, automatically handling the conversion:

.. code-block:: cpp

    #include <bitset>
    #include <iostream>

    int main(int argc, char *argv[]) {
      std::bitset<4> bits{0b1000};  // Binary: 1000

      std::cout << bits.count() << "\n";  // Output: 1 (popcount)
      std::cout << (bits == 8) << "\n";   // Output: 1 (true)
    }

Safe Address-of with std::addressof
-----------------------------------

:Source: `src/basic/addressof <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/basic/addressof>`_

C++ permits overloading ``operator&``, which can interfere with obtaining
an object's actual memory address. While overloading this operator is rare
and generally discouraged, it does occur in some legacy codebases and smart
pointer implementations. The ``std::addressof`` function template bypasses
any overloaded ``operator&`` to return the true address, making it essential
for generic code that must work with arbitrary types.

In this example, the overloaded ``operator&`` could potentially return
an arbitrary pointer (perhaps for proxy object patterns or debugging).
Using ``std::addressof`` guarantees we obtain the object's actual memory
location regardless of any operator overloading. This is particularly
important in allocator implementations, container internals, and other
low-level generic code:

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct Widget {
      int value;
    };

    const Widget* operator&(const Widget& w) {
      // Custom behavior - could return anything
      return std::addressof(w);  // Safe: returns actual address
    }

    int main(int argc, char *argv[]) {
      Widget w;
      std::cout << &w << "\n";                // Uses overloaded operator&
      std::cout << std::addressof(w) << "\n"; // Always returns true address
    }

.. note::

    Always use ``std::addressof`` in generic code where the type may have
    an overloaded ``operator&``. The standard library containers and algorithms
    use ``std::addressof`` internally for this reason.
