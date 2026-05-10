========
Closures
========

.. meta::
   :description: Rust closures compared to C++ lambdas. Covers Fn traits, move closures, and closure type inference.
   :keywords: Rust, closures, lambda, Fn, FnMut, FnOnce, C++ lambda, capture

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/closures <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/closures>`_

Rust closures are anonymous functions that can capture variables from their enclosing
scope. They're similar to C++ lambdas but with a key difference: Rust automatically
infers how to capture each variable (by reference, mutable reference, or by value)
based on how the closure uses it. This eliminates the need for explicit capture lists
like ``[&]`` or ``[=]`` in C++. Closures in Rust also implement one or more of the
``Fn``, ``FnMut``, or ``FnOnce`` traits, which determines how they can be called.

Basic Syntax
------------

Rust closures use vertical bars ``|args|`` instead of parentheses for parameters, and
the body can be a single expression or a block. Type annotations are optional - Rust
infers types from how the closure is used. Unlike C++ where you must explicitly specify
capture mode, Rust analyzes the closure body and captures each variable in the least
restrictive way possible:

**C++:**

.. code-block:: cpp

    #include <functional>

    int main() {
      // Lambda with explicit capture
      int x = 10;
      auto add = [x](int y) { return x + y; };
      auto result = add(5);  // 15

      // Mutable capture
      int count = 0;
      auto increment = [&count]() { count++; };
    }

**Rust:**

.. code-block:: rust

    fn main() {
        // Closure with automatic capture
        let x = 10;
        let add = |y| x + y;
        let result = add(5);  // 15

        // Mutable capture
        let mut count = 0;
        let mut increment = || count += 1;
        increment();
    }

Capture Modes
-------------

C++ requires explicit capture specifications (``[=]``, ``[&]``, ``[x]``, ``[&x]``),
which can be error-prone - capturing by reference when you meant by value can cause
dangling references. Rust infers the capture mode automatically: if the closure only
reads a variable, it captures by immutable reference; if it modifies the variable, it
captures by mutable reference; if it moves the variable (e.g., returns it or passes
it to a function taking ownership), it captures by value:

**C++ explicit captures:**

.. code-block:: cpp

    int a = 1, b = 2;

    auto by_value = [a, b]() { return a + b; };
    auto by_ref = [&a, &b]() { return a + b; };
    auto all_value = [=]() { return a + b; };
    auto all_ref = [&]() { return a + b; };
    auto mixed = [a, &b]() { return a + b; };

**Rust automatic inference:**

.. code-block:: rust

    let a = 1;
    let mut b = 2;

    // Rust infers capture mode based on usage
    let by_ref = || a + b;           // borrows a and b
    let by_mut_ref = || { b += 1; }; // mutably borrows b

    // Force move with `move` keyword
    let by_value = move || a + b;    // moves/copies a and b

Move Closures
~~~~~~~~~~~~~

The ``move`` keyword forces a closure to take ownership of all captured variables,
even if the closure body would only require borrowing. This is essential when the
closure needs to outlive the current scope, such as when spawning threads or returning
closures from functions. For ``Copy`` types like integers, ``move`` creates a copy;
for non-``Copy`` types like ``String``, it transfers ownership:

.. code-block:: rust

    fn main() {
        let s = String::from("hello");

        // Without move: borrows s
        let borrow = || println!("{}", s);

        // With move: takes ownership of s
        let own = move || println!("{}", s);
        // s is no longer valid here

        // Essential for threads
        let data = vec![1, 2, 3];
        std::thread::spawn(move || {
            println!("{:?}", data);
        });
    }

Fn Traits
---------

Rust closures implement one or more of three traits based on how they use captured
variables. ``FnOnce`` is the most general - all closures implement it. ``FnMut`` is
for closures that don't consume captured values but may mutate them. ``Fn`` is for
closures that only read captured values. When accepting closures as parameters, use
the most general trait that works (``FnOnce`` > ``FnMut`` > ``Fn``) to maximize flexibility:

+-------------+------------------+----------------------------------+
| Trait       | Receiver         | Can be called                    |
+=============+==================+==================================+
| ``FnOnce``  | ``self``         | Once (consumes captured values)  |
+-------------+------------------+----------------------------------+
| ``FnMut``   | ``&mut self``    | Multiple times (may mutate)      |
+-------------+------------------+----------------------------------+
| ``Fn``      | ``&self``        | Multiple times (no mutation)     |
+-------------+------------------+----------------------------------+

.. code-block:: rust

    // FnOnce - consumes captured value
    let s = String::from("hello");
    let consume = || drop(s);  // can only call once

    // FnMut - mutates captured value
    let mut count = 0;
    let mut increment = || count += 1;  // can call multiple times

    // Fn - only reads captured values
    let x = 10;
    let read = || x + 1;  // can call multiple times

Closures as Parameters
----------------------

Functions can accept closures as parameters using generics with trait bounds or trait
objects. The generic approach (monomorphization) generates specialized code for each
closure type, giving the best performance. The trait object approach (``dyn Fn``)
uses dynamic dispatch, which has a small runtime cost but allows storing different
closure types in the same collection:

**C++:**

.. code-block:: cpp

    #include <functional>

    void apply(std::function<int(int)> f, int x) {
      std::cout << f(x);
    }

    // Or with templates
    template<typename F>
    void apply_template(F f, int x) {
      std::cout << f(x);
    }

**Rust:**

.. code-block:: rust

    // Generic with trait bound (monomorphized, like C++ template)
    fn apply<F>(f: F, x: i32) -> i32
    where
        F: Fn(i32) -> i32,
    {
        f(x)
    }

    // impl Trait syntax (simpler)
    fn apply_impl(f: impl Fn(i32) -> i32, x: i32) -> i32 {
        f(x)
    }

    // Dynamic dispatch (like std::function)
    fn apply_dyn(f: &dyn Fn(i32) -> i32, x: i32) -> i32 {
        f(x)
    }

    fn main() {
        let double = |x| x * 2;
        println!("{}", apply(double, 5));  // 10
    }

Returning Closures
------------------

Returning closures from functions requires either ``impl Trait`` syntax (when returning
a single closure type) or ``Box<dyn Fn>`` (when the closure type might vary). The
``move`` keyword is usually necessary because the closure needs to own its captured
variables - otherwise they would be dropped when the function returns:

**C++:**

.. code-block:: cpp

    std::function<int(int)> make_adder(int x) {
      return [x](int y) { return x + y; };
    }

**Rust:**

.. code-block:: rust

    // Return with impl Trait
    fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
        move |y| x + y
    }

    // Return boxed closure (for dynamic dispatch)
    fn make_adder_boxed(x: i32) -> Box<dyn Fn(i32) -> i32> {
        Box::new(move |y| x + y)
    }

    fn main() {
        let add5 = make_adder(5);
        println!("{}", add5(10));  // 15
    }

Closure Type Inference
----------------------

Rust infers closure parameter and return types from how the closure is used. This
means you usually don't need type annotations, but it also means each closure has
a unique anonymous type - you can't have two closures with the same signature stored
in the same variable without using trait objects. Once a closure's types are inferred
from its first use, they're fixed:

.. code-block:: rust

    // Types inferred from usage
    let add = |a, b| a + b;
    let result: i32 = add(1, 2);  // infers i32

    // Explicit types when needed
    let parse = |s: &str| -> i32 { s.parse().unwrap() };

    // Once inferred, types are fixed
    let identity = |x| x;
    let s = identity("hello");  // infers &str
    // let n = identity(42);    // error: expected &str

Closures with Iterators
-----------------------

Closures are most commonly used with iterator adapters like ``map``, ``filter``, and
``fold``. These methods take closures that transform or filter elements. The closure
syntax is concise enough that complex transformations remain readable. Iterator
adapters are lazy - they don't execute until you consume the iterator with ``collect``,
``for_each``, or similar methods:

.. code-block:: rust

    let numbers = vec![1, 2, 3, 4, 5];

    // map
    let doubled: Vec<_> = numbers.iter().map(|x| x * 2).collect();

    // filter
    let evens: Vec<_> = numbers.iter().filter(|&&x| x % 2 == 0).collect();

    // fold
    let sum = numbers.iter().fold(0, |acc, x| acc + x);

    // for_each
    numbers.iter().for_each(|x| println!("{}", x));

    // find
    let first_even = numbers.iter().find(|&&x| x % 2 == 0);

See Also
--------

- :doc:`rust_iterator` - Closures with iterator adapters
- :doc:`rust_traits` - Fn traits in detail
