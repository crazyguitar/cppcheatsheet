====
RAII
====

.. meta::
   :description: Rust RAII and Drop trait compared to C++ destructors. Covers resource management, constructors, Drop implementation, and explicit drop.
   :keywords: Rust, RAII, Drop, destructor, constructor, resource management, C++ comparison

.. contents:: Table of Contents
    :backlinks: none

RAII (Resource Acquisition Is Initialization) is fundamental to both C++ and Rust.
Resources are acquired during initialization and released when the object goes out
of scope. Rust's ownership system makes RAII even more powerful by guaranteeing
exactly when destructors run.

Constructors
------------

:Source: `src/rust/structs <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/structs>`_

C++ has dedicated constructor syntax. Rust doesn't have constructors as a language
feature; instead, you define associated functions (commonly named ``new``) that
return ``Self``.

**C++:**

.. code-block:: cpp

    class Point {
      double x_, y_;
    public:
      // Default constructor
      Point() : x_(0), y_(0) {}

      // Parameterized constructor
      Point(double x, double y) : x_(x), y_(y) {}

      // Copy constructor (implicit or explicit)
      Point(const Point& other) = default;

      // Move constructor
      Point(Point&& other) noexcept = default;
    };

    int main() {
      Point p1;           // default constructor
      Point p2(3.0, 4.0); // parameterized constructor
      Point p3 = p2;      // copy constructor
    }

**Rust:**

.. code-block:: rust

    struct Point {
        x: f64,
        y: f64,
    }

    impl Point {
        // Associated function (not a method - no self)
        fn new(x: f64, y: f64) -> Self {
            Point { x, y }
        }

        // Default-like constructor
        fn origin() -> Self {
            Point { x: 0.0, y: 0.0 }
        }
    }

    fn main() {
        let p1 = Point::origin();      // "default" constructor
        let p2 = Point::new(3.0, 4.0); // parameterized constructor
        let p3 = p2;                   // move (Point doesn't impl Copy)
    }

Default Trait
~~~~~~~~~~~~~

Rust's ``Default`` trait is similar to C++ default constructors:

.. code-block:: rust

    #[derive(Default)]
    struct Config {
        debug: bool,      // defaults to false
        timeout: u32,     // defaults to 0
        name: String,     // defaults to ""
    }

    fn main() {
        let config = Config::default();
        // Or with struct update syntax
        let custom = Config {
            debug: true,
            ..Default::default()
        };
    }

Destructors and Drop
--------------------

:Source: `src/rust/raii <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/raii>`_

C++ uses destructors (``~ClassName``). Rust uses the ``Drop`` trait with a ``drop``
method. Both are called automatically when the object goes out of scope.

**C++:**

.. code-block:: cpp

    #include <iostream>

    class Resource {
      int* data;
    public:
      Resource(int value) : data(new int(value)) {
        std::cout << "Acquired: " << *data << "\n";
      }

      ~Resource() {
        std::cout << "Released: " << *data << "\n";
        delete data;
      }

      // Rule of five: need copy/move constructors and assignments
      Resource(const Resource&) = delete;
      Resource& operator=(const Resource&) = delete;
    };

    int main() {
      Resource r(42);
      // destructor called at end of scope
    }

**Rust:**

.. code-block:: rust

    struct Resource {
        data: i32,
    }

    impl Resource {
        fn new(value: i32) -> Self {
            println!("Acquired: {}", value);
            Resource { data: value }
        }
    }

    impl Drop for Resource {
        fn drop(&mut self) {
            println!("Released: {}", self.data);
        }
    }

    fn main() {
        let r = Resource::new(42);
        // Drop::drop called at end of scope
    }

Drop Order
~~~~~~~~~~

Variables are dropped in reverse order of declaration:

.. code-block:: rust

    struct Droppable(i32);

    impl Drop for Droppable {
        fn drop(&mut self) {
            println!("Dropping {}", self.0);
        }
    }

    fn main() {
        let a = Droppable(1);
        let b = Droppable(2);
        let c = Droppable(3);
    }
    // Output: Dropping 3, Dropping 2, Dropping 1

Explicit Drop
-------------

Use ``std::mem::drop()`` to drop a value before its scope ends:

**C++ (no direct equivalent, use scope or unique_ptr):**

.. code-block:: cpp

    {
      Resource r(42);
      // use r
    }  // r destroyed here
    // continue without r

**Rust:**

.. code-block:: rust

    fn main() {
        let r = Resource::new(42);
        // use r
        drop(r);  // explicitly drop
        // r is no longer valid
        println!("Resource already released");
    }

Preventing Drop
~~~~~~~~~~~~~~~

``std::mem::forget()`` prevents the destructor from running. Useful for FFI when
transferring ownership to C code:

.. code-block:: rust

    use std::mem;

    fn main() {
        let v = vec![1, 2, 3];
        mem::forget(v);  // destructor won't run, memory leaked
        // Use carefully - typically only for FFI
    }

Rule of Three/Five vs Rust
--------------------------

C++ requires implementing copy/move constructors and assignment operators (Rule of
Three/Five). Rust handles this through traits:

**C++ Rule of Five:**

.. code-block:: cpp

    class Buffer {
      char* data;
      size_t size;
    public:
      Buffer(size_t n) : data(new char[n]), size(n) {}
      ~Buffer() { delete[] data; }

      // Copy constructor
      Buffer(const Buffer& other) : data(new char[other.size]), size(other.size) {
        std::copy(other.data, other.data + size, data);
      }

      // Copy assignment
      Buffer& operator=(const Buffer& other) {
        if (this != &other) {
          delete[] data;
          size = other.size;
          data = new char[size];
          std::copy(other.data, other.data + size, data);
        }
        return *this;
      }

      // Move constructor
      Buffer(Buffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
      }

      // Move assignment
      Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
          delete[] data;
          data = other.data;
          size = other.size;
          other.data = nullptr;
          other.size = 0;
        }
        return *this;
      }
    };

**Rust (much simpler):**

.. code-block:: rust

    struct Buffer {
        data: Vec<u8>,
    }

    impl Buffer {
        fn new(size: usize) -> Self {
            Buffer { data: vec![0; size] }
        }
    }

    // Move is automatic, no implementation needed
    // Clone can be derived if needed
    impl Clone for Buffer {
        fn clone(&self) -> Self {
            Buffer { data: self.data.clone() }
        }
    }

    // Drop is automatic for Vec, no implementation needed

this vs self
------------

:Source: `src/rust/self_this <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/self_this>`_

C++ has implicit ``this`` pointer. Rust requires explicit ``self`` parameter with
three forms that integrate with ownership:

**C++:**

.. code-block:: cpp

    class Counter {
      int value = 0;
    public:
      void increment() { this->value++; }  // this is implicit
      int get() const { return value; }
    };

**Rust:**

.. code-block:: rust

    struct Counter {
        value: i32,
    }

    impl Counter {
        fn increment(&mut self) {    // mutable borrow
            self.value += 1;
        }

        fn get(&self) -> i32 {       // immutable borrow
            self.value
        }

        fn into_value(self) -> i32 { // takes ownership, consumes self
            self.value
        }
    }

See Also
--------

- :doc:`rust_basic` - Ownership and borrowing fundamentals
- :doc:`rust_smartptr` - Smart pointers for shared ownership
