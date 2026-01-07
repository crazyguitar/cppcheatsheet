===============================
Return Value Optimization (RVO)
===============================

.. meta::
   :description: C++ RVO and NRVO explained with examples showing copy elision, move semantics, and compiler optimizations.
   :keywords: C++, RVO, NRVO, return value optimization, copy elision, move semantics, optimization

.. contents:: Table of Contents
    :backlinks: none

Return Value Optimization (RVO) and Named Return Value Optimization (NRVO) are
compiler optimizations that eliminate unnecessary copy or move operations when
returning objects from functions. Instead of constructing an object locally and
then copying it to the caller's storage, the compiler constructs the object
directly in the caller's memory location. This optimization can significantly
improve performance for types with expensive copy operations. C++17 made RVO
mandatory (guaranteed copy elision) for certain cases, meaning the optimization
is no longer optional but required by the standard. NRVO remains an optional
optimization that most compilers implement but is not guaranteed.

Instrumented Class for Testing
------------------------------

:Source: `src/rvo/instrumented <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/instrumented>`_

To observe copy elision behavior, use a class that logs all special member
function calls. This technique helps verify when copies and moves are elided
by the compiler. By examining the output, you can determine whether RVO or
NRVO was applied, or if a copy/move operation occurred instead.

.. code-block:: cpp

    #include <iostream>

    struct Foo {
      Foo() { std::cout << "Constructor\n"; }
      ~Foo() { std::cout << "Destructor\n"; }
      Foo(const Foo&) { std::cout << "Copy Constructor\n"; }
      Foo(Foo&&) { std::cout << "Move Constructor\n"; }
      Foo& operator=(const Foo&) { std::cout << "Copy Assignment\n"; return *this; }
      Foo& operator=(Foo&&) { std::cout << "Move Assignment\n"; return *this; }
    };

Return Value Optimization (RVO)
-------------------------------

:Source: `src/rvo/rvo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/rvo>`_

RVO applies when returning a temporary (prvalue) directly from a function.
The compiler constructs the object directly in the caller's storage location,
completely eliminating any copy or move operation. Since C++17, this optimization
is guaranteed by the standard (mandatory copy elision), meaning the copy or move
constructor doesn't even need to be accessible or defined. This is particularly
useful for factory functions and builder patterns where objects are created
and returned frequently.

.. code-block:: cpp

    Foo MakeRVO() {
      return Foo();  // Guaranteed RVO since C++17
    }

    int main() {
      Foo f = MakeRVO();  // Only one Constructor call, no copy/move
    }

.. code-block:: text

    Output:
    Constructor
    Destructor

Named Return Value Optimization (NRVO)
--------------------------------------

:Source: `src/rvo/nrvo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/nrvo>`_

NRVO applies when returning a named local variable from a function. The compiler
may construct the local variable directly in the caller's return slot, avoiding
any copy or move when the function returns. Unlike RVO, NRVO is not guaranteed
by the C++ standard and depends on compiler optimization settings. However,
most modern compilers (GCC, Clang, MSVC) implement NRVO reliably when optimization
is enabled. When NRVO cannot be applied, the compiler will use move semantics
if available, falling back to copy only when necessary.

.. code-block:: cpp

    Foo MakeNRVO() {
      Foo foo;      // Named local variable
      return foo;   // NRVO may apply
    }

    int main() {
      Foo f = MakeNRVO();  // Typically one Constructor, no copy/move
    }

.. code-block:: text

    Output (with NRVO):
    Constructor
    Destructor

Copy Elision in Function Arguments
----------------------------------

:Source: `src/rvo/copy-elision <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/copy-elision>`_

When passing a temporary to a function by value, the compiler can construct
the argument directly in the parameter's storage location, eliding the copy
entirely. This optimization applies when the argument is a prvalue (temporary)
and the parameter is passed by value. Combined with move semantics, this makes
passing objects by value efficient in many cases, especially for sink parameters
that will be stored or consumed by the function.

.. code-block:: cpp

    void TakeByValue(Foo foo) {}

    int main() {
      TakeByValue(Foo());  // Copy elision: no copy/move
    }

.. code-block:: text

    Output:
    Constructor
    Destructor

Cases Where RVO Does NOT Apply
------------------------------

:Source: `src/rvo/no-rvo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/no-rvo>`_

RVO and NRVO cannot apply in certain situations where the compiler cannot
determine a single return location at compile time, or when the returned
object already exists elsewhere. Understanding these cases helps you write
more efficient code and avoid unexpected performance costs.

**Returning a global or static variable:**

Global and static variables have fixed storage locations that cannot be
changed. When returning such a variable, the compiler must copy it to the
caller's storage because the original must remain intact.

.. code-block:: cpp

    const Foo global_foo;

    Foo ReturnGlobal() {
      return global_foo;  // Must copy, cannot elide
    }

**Returning a function parameter:**

Function parameters are allocated in the caller's stack frame before the
function is called. The return value location is separate, so the parameter
must be moved or copied to the return location.

.. code-block:: cpp

    Foo ReturnParam(Foo foo) {
      return foo;  // Must move (or copy), cannot elide
    }

**Returning a member of another object:**

When returning a member of an object, the member's storage is part of the
containing object. The compiler cannot construct it directly in the return
location, so a move or copy is required.

.. code-block:: cpp

    struct Bar { Foo foo; };

    Foo ReturnMember() {
      return Bar().foo;  // Must move, cannot elide
    }

**Multiple return paths with different named variables:**

When a function has multiple return statements returning different named
variables, the compiler cannot determine which variable's storage to use
as the return location. NRVO requires a single, unambiguous return variable.

.. code-block:: cpp

    Foo NoNRVO(bool flag) {
      Foo x, y;
      return flag ? x : y;  // NRVO cannot apply
    }

Don't Use std::move on Return
-----------------------------

:Source: `src/rvo/no-move-return <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/no-move-return>`_

Using ``std::move`` on a return statement is a common mistake that actually
pessimizes performance. It prevents RVO and NRVO because ``std::move`` converts
the expression to an xvalue, which disables copy elision. The compiler already
applies implicit move semantics for local variables in return statements when
NRVO doesn't apply, so explicit ``std::move`` is never needed and always harmful
for return values.

.. code-block:: cpp

    // BAD: Prevents RVO
    Foo BadRVO() {
      return std::move(Foo());  // Forces move, prevents RVO
    }

    // BAD: Prevents NRVO
    Foo BadNRVO() {
      Foo foo;
      return std::move(foo);  // Forces move, prevents NRVO
    }

    // GOOD: Let the compiler optimize
    Foo GoodRVO() {
      return Foo();  // RVO applies
    }

    Foo GoodNRVO() {
      Foo foo;
      return foo;  // NRVO may apply, implicit move otherwise
    }

.. warning::

    Never use ``std::move`` on return statements for local variables. It
    pessimizes performance by preventing copy elision. The compiler will
    automatically apply move semantics when appropriate.

Runtime Conditional Returns
---------------------------

:Source: `src/rvo/conditional <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rvo/conditional>`_

When returning different temporaries based on a runtime condition, RVO can
still apply because each branch returns a prvalue that can be constructed
directly in the return location. The compiler generates code that constructs
whichever temporary is selected directly in the caller's storage.

.. code-block:: cpp

    Foo ConditionalRVO(bool flag) {
      return flag ? Foo() : Foo();  // RVO applies to both branches
    }

However, mixing named variables with temporaries or using different named
variables in different branches prevents NRVO because the compiler cannot
determine a single variable to use as the return location:

.. code-block:: cpp

    Foo MixedReturn(bool flag) {
      Foo x;
      return flag ? x : Foo();  // NRVO cannot apply
    }

Guaranteed Copy Elision (C++17)
-------------------------------

C++17 mandates copy elision in specific cases, making RVO guaranteed rather
than optional. This is a significant language change because it means the
copy or move constructor doesn't even need to exist for these cases to work.
This enables returning non-copyable, non-movable types from functions and
makes factory functions work with any type.

The guaranteed cases are:

- Returning a prvalue of the same type as the function return type
- Initializing a variable from a prvalue of the same type

.. code-block:: cpp

    struct NonMovable {
      NonMovable() = default;
      NonMovable(const NonMovable&) = delete;
      NonMovable(NonMovable&&) = delete;
    };

    NonMovable Make() {
      return NonMovable();  // OK in C++17: guaranteed copy elision
    }

    int main() {
      NonMovable n = Make();  // OK: no copy/move needed
    }

Summary Table
-------------

The following table provides a quick reference for when RVO and NRVO apply.
Use this to identify patterns that enable copy elision and avoid patterns that
prevent it. Remember that RVO (returning temporaries) is guaranteed in C++17,
while NRVO (returning named variables) is a compiler optimization that typically
applies but is not guaranteed by the standard.

.. list-table::
   :header-rows: 1

   * - Scenario
     - RVO/NRVO
     - Notes
   * - ``return Foo();``
     - ✓ Guaranteed (C++17)
     - Prvalue return
   * - ``return local_var;``
     - ✓ Likely (NRVO)
     - Compiler optimization
   * - ``return std::move(x);``
     - ✗ Prevented
     - Never do this
   * - ``return global;``
     - ✗ No
     - Must copy
   * - ``return param;``
     - ✗ No
     - Must move
   * - ``return obj.member;``
     - ✗ No
     - Must move
   * - ``return flag ? x : y;``
     - ✗ No NRVO
     - Multiple candidates
