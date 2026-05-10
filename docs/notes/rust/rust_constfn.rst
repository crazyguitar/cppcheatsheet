===============
Const Functions
===============

.. meta::
   :description: Rust const fn compared to C++ constexpr. Covers compile-time evaluation, const generics, and limitations.
   :keywords: Rust, const fn, constexpr, compile-time, const generics, C++

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/const_fn <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/const_fn>`_

Rust's ``const fn`` is similar to C++ ``constexpr`` - functions that can be
evaluated at compile time. Rust's const evaluation is more restrictive but
guarantees deterministic behavior.

Basic const fn
--------------

**C++:**

.. code-block:: cpp

    constexpr int square(int x) {
      return x * x;
    }

    constexpr int factorial(int n) {
      return n <= 1 ? 1 : n * factorial(n - 1);
    }

    int main() {
      constexpr int a = square(5);      // compile-time
      constexpr int b = factorial(5);   // compile-time
      int runtime_val = 10;
      int c = square(runtime_val);      // runtime
    }

**Rust:**

.. code-block:: rust

    const fn square(x: i32) -> i32 {
        x * x
    }

    const fn factorial(n: u32) -> u32 {
        if n <= 1 { 1 } else { n * factorial(n - 1) }
    }

    fn main() {
        const A: i32 = square(5);       // compile-time
        const B: u32 = factorial(5);    // compile-time
        let runtime_val = 10;
        let c = square(runtime_val);    // runtime
    }

Const Context
-------------

``const fn`` can be called in const contexts:

.. code-block:: rust

    const fn compute() -> usize { 42 }

    // Array size
    const SIZE: usize = compute();
    let arr: [i32; SIZE] = [0; SIZE];

    // Static variables
    static VALUE: i32 = square(10);

    // Const generics
    struct Buffer<const N: usize> {
        data: [u8; N],
    }

    const fn buffer_size() -> usize { 1024 }
    let buf: Buffer<{ buffer_size() }> = Buffer { data: [0; 1024] };

Const Limitations
-----------------

Rust's const fn has more restrictions than C++ constexpr:

.. code-block:: rust

    // Allowed in const fn:
    const fn allowed() -> i32 {
        let x = 5;
        let y = if x > 0 { 10 } else { 20 };
        let mut sum = 0;
        let mut i = 0;
        while i < 5 {
            sum += i;
            i += 1;
        }
        sum
    }

    // NOT allowed (as of Rust 1.70):
    // - Heap allocation (Box, Vec, String)
    // - Trait objects
    // - Most iterator methods
    // - Floating point operations (stabilizing)
    // - Mutable references to non-local data

Const Generics
--------------

**C++ (non-type template parameters):**

.. code-block:: cpp

    template<size_t N>
    struct Buffer {
      char data[N];

      constexpr size_t size() const { return N; }
    };

    int main() {
      Buffer<1024> buf;
    }

**Rust:**

.. code-block:: rust

    struct Buffer<const N: usize> {
        data: [u8; N],
    }

    impl<const N: usize> Buffer<N> {
        const fn new() -> Self {
            Buffer { data: [0; N] }
        }

        const fn size(&self) -> usize {
            N
        }
    }

    fn main() {
        let buf = Buffer::<1024>::new();
        const SIZE: usize = buf.size();
    }

Const Expressions in Types
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

    // Array with computed size
    const fn array_len<T>() -> usize {
        std::mem::size_of::<T>() * 8
    }

    struct BitSet<T> {
        bits: [u8; std::mem::size_of::<T>()],
    }

    // Generic over const expression
    fn process<const N: usize>(arr: [i32; N]) -> i32 {
        let mut sum = 0;
        let mut i = 0;
        while i < N {
            sum += arr[i];
            i += 1;
        }
        sum
    }

Const Trait Methods
-------------------

.. code-block:: rust

    struct Point {
        x: i32,
        y: i32,
    }

    impl Point {
        // const method
        const fn new(x: i32, y: i32) -> Self {
            Point { x, y }
        }

        const fn origin() -> Self {
            Point { x: 0, y: 0 }
        }

        const fn manhattan_distance(&self) -> i32 {
            // abs() is const fn
            self.x.abs() + self.y.abs()
        }
    }

    const ORIGIN: Point = Point::origin();
    const P: Point = Point::new(3, 4);
    const DIST: i32 = P.manhattan_distance();

Compile-Time Assertions
-----------------------

**C++:**

.. code-block:: cpp

    static_assert(sizeof(int) == 4, "int must be 4 bytes");

    template<typename T>
    constexpr void check_size() {
      static_assert(sizeof(T) <= 16, "Type too large");
    }

**Rust:**

.. code-block:: rust

    // Using const evaluation
    const _: () = assert!(std::mem::size_of::<i32>() == 4);

    // In const fn
    const fn check_size<T>() {
        assert!(std::mem::size_of::<T>() <= 16, "Type too large");
    }

    // Compile-time check
    const _: () = check_size::<i32>();

See Also
--------

- :doc:`rust_traits` - Const trait bounds
