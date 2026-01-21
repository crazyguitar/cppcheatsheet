========
Closures
========

.. meta::
   :description: Rust closures compared to C++ lambdas. Covers Fn traits, move closures, and closure type inference.
   :keywords: Rust, closures, lambda, Fn, FnMut, FnOnce, C++ lambda, capture

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/closures <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/closures>`_

Rust closures are anonymous functions that can capture variables from their
environment. They're similar to C++ lambdas but with automatic capture mode
inference.

Basic Syntax
------------

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

Rust closures implement one or more of three traits:

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

Rust infers closure parameter and return types from usage:

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
