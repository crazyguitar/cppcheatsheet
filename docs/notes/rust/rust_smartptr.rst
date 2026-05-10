==============
Smart Pointers
==============

.. meta::
   :description: Rust smart pointers compared to C++. Covers Box, Rc, Arc, RefCell, Mutex with C++ unique_ptr, shared_ptr equivalents.
   :keywords: Rust, Box, Rc, Arc, RefCell, Mutex, smart pointers, C++ unique_ptr, shared_ptr

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/smart_pointers <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/smart_pointers>`_

Rust smart pointers provide different ownership and borrowing patterns beyond what
basic references offer. Unlike C++ where smart pointers are library types that can
be misused (e.g., creating cycles with ``shared_ptr``), Rust's ownership system
prevents most pointer-related bugs at compile time. Smart pointers in Rust implement
the ``Deref`` trait, allowing them to be used like regular references, and the ``Drop``
trait for automatic cleanup.

Smart Pointer Comparison
------------------------

The following table maps C++ smart pointers to their Rust equivalents. Note that Rust
splits C++'s ``shared_ptr`` into two types: ``Rc`` for single-threaded code (faster,
no atomic operations) and ``Arc`` for multi-threaded code (thread-safe with atomic
reference counting):

+------------------------+------------------------+
| C++                    | Rust                   |
+========================+========================+
| ``std::unique_ptr<T>`` | ``Box<T>``             |
+------------------------+------------------------+
| ``std::shared_ptr<T>`` | ``Rc<T>`` / ``Arc<T>`` |
+------------------------+------------------------+
| ``std::weak_ptr<T>``   | ``Weak<T>``            |
+------------------------+------------------------+
| N/A                    | ``RefCell<T>``         |
+------------------------+------------------------+
| ``std::mutex<T>``      | ``Mutex<T>``           |
+------------------------+------------------------+

Box (Unique Ownership)
----------------------

``Box<T>`` provides heap allocation with single ownership, similar to ``std::unique_ptr<T>``.
When a ``Box`` goes out of scope, it automatically frees the heap memory. Unlike C++
where you might accidentally copy a ``unique_ptr`` (which is a compile error but easy
to attempt), Rust's move semantics make ownership transfer explicit and safe. ``Box``
is commonly used for recursive data structures, large data that shouldn't be copied,
and trait objects.

**C++:**

.. code-block:: cpp

    #include <memory>

    int main() {
      auto ptr = std::make_unique<int>(42);
      std::cout << *ptr;

      // Transfer ownership
      auto ptr2 = std::move(ptr);
      // ptr is now null
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let boxed = Box::new(42);
        println!("{}", *boxed);

        // Transfer ownership
        let boxed2 = boxed;
        // boxed is no longer valid - compile error if used
    }

Use Cases for Box
~~~~~~~~~~~~~~~~~

``Box`` is essential in several scenarios where stack allocation isn't possible or
desirable. Recursive types like linked lists and trees require ``Box`` because the
compiler needs to know the size of types at compile time - without indirection, a
recursive type would have infinite size. Large data benefits from ``Box`` to avoid
expensive stack copies. Trait objects require ``Box`` (or another pointer type)
because different implementations may have different sizes:

.. code-block:: rust

    // 1. Recursive types (unknown size at compile time)
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }

    // 2. Large data on heap
    let large = Box::new([0u8; 1_000_000]);

    // 3. Trait objects
    let drawable: Box<dyn Draw> = Box::new(Circle { radius: 5.0 });

Rc (Reference Counting)
-----------------------

``Rc<T>`` (Reference Counted) enables shared ownership where multiple parts of your
code need to read the same data. It's similar to ``std::shared_ptr<T>`` but is
explicitly single-threaded - attempting to send an ``Rc`` to another thread is a
compile error. This restriction allows ``Rc`` to use non-atomic reference counting,
making it faster than ``Arc`` when thread safety isn't needed. The reference count
is incremented when you clone an ``Rc`` and decremented when an ``Rc`` is dropped.

**C++:**

.. code-block:: cpp

    #include <memory>

    int main() {
      auto shared = std::make_shared<int>(42);
      auto shared2 = shared;  // ref count = 2

      std::cout << shared.use_count();  // 2
    }

**Rust:**

.. code-block:: rust

    use std::rc::Rc;

    fn main() {
        let shared = Rc::new(42);
        let shared2 = Rc::clone(&shared);  // ref count = 2

        println!("{}", Rc::strong_count(&shared));  // 2
    }

Rc with RefCell (Interior Mutability)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Rc<T>`` only provides shared (immutable) access to its contents - you cannot get
a mutable reference through an ``Rc``. When you need shared ownership with mutation,
combine ``Rc`` with ``RefCell``. ``RefCell`` moves Rust's borrow checking from compile
time to runtime: it tracks borrows dynamically and will panic if you violate the
borrowing rules (e.g., two simultaneous mutable borrows). This pattern is common for
graph structures, observer patterns, and caches:

.. code-block:: rust

    use std::cell::RefCell;
    use std::rc::Rc;

    fn main() {
        let shared = Rc::new(RefCell::new(42));

        // Multiple owners
        let shared2 = Rc::clone(&shared);

        // Mutate through RefCell
        *shared.borrow_mut() += 1;

        println!("{}", shared2.borrow());  // 43
    }

Arc (Atomic Reference Counting)
-------------------------------

``Arc<T>`` (Atomically Reference Counted) is the thread-safe version of ``Rc<T>``.
It uses atomic operations for reference counting, making it safe to share across
threads. This is equivalent to ``std::shared_ptr<T>`` in C++, which also uses atomic
reference counting. The atomic operations add some overhead compared to ``Rc``, so
prefer ``Rc`` when you don't need thread safety. ``Arc`` is commonly used with
``Mutex`` or ``RwLock`` to provide thread-safe shared mutable state.

**C++:**

.. code-block:: cpp

    #include <memory>
    #include <thread>

    int main() {
      auto shared = std::make_shared<int>(42);

      std::thread t([shared]() {
        std::cout << *shared;
      });
      t.join();
    }

**Rust:**

.. code-block:: rust

    use std::sync::Arc;
    use std::thread;

    fn main() {
        let shared = Arc::new(42);
        let shared2 = Arc::clone(&shared);

        let handle = thread::spawn(move || {
            println!("{}", *shared2);
        });

        handle.join().unwrap();
    }

Arc with Mutex
~~~~~~~~~~~~~~

For thread-safe mutable access, combine ``Arc`` with ``Mutex``. The ``Mutex`` ensures
only one thread can access the data at a time, while ``Arc`` allows multiple threads
to hold references to the mutex. This pattern is the Rust equivalent of sharing a
``std::shared_ptr<std::mutex<T>>`` in C++, but Rust's type system ensures you can't
forget to lock the mutex - the data is only accessible through the lock guard:

.. code-block:: rust

    use std::sync::{Arc, Mutex};
    use std::thread;

    fn main() {
        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                let mut num = counter.lock().unwrap();
                *num += 1;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        println!("{}", *counter.lock().unwrap());  // 10
    }

RefCell (Interior Mutability)
-----------------------------

``RefCell<T>`` provides interior mutability - the ability to mutate data even when
there are immutable references to it. This is checked at runtime rather than compile
time: ``RefCell`` tracks active borrows and panics if you violate the borrowing rules.
Use ``RefCell`` when you know your code follows borrowing rules but the compiler can't
verify it, such as in graph structures or mock objects for testing. The runtime checks
have a small performance cost, so prefer compile-time borrowing when possible:

.. code-block:: rust

    use std::cell::RefCell;

    fn main() {
        let cell = RefCell::new(42);

        // Immutable borrow
        let r1 = cell.borrow();
        println!("{}", *r1);
        drop(r1);  // must drop before mutable borrow

        // Mutable borrow
        *cell.borrow_mut() += 1;

        // Runtime panic if rules violated:
        // let r1 = cell.borrow();
        // let r2 = cell.borrow_mut();  // panic!
    }

Cell (Copy Types)
~~~~~~~~~~~~~~~~~

For types that implement ``Copy`` (like integers and floats), ``Cell<T>`` provides
a simpler alternative to ``RefCell``. Instead of borrowing, ``Cell`` copies values
in and out. This avoids the runtime borrow tracking overhead and can never panic,
but only works with ``Copy`` types since it needs to copy the entire value:

.. code-block:: rust

    use std::cell::Cell;

    fn main() {
        let cell = Cell::new(42);

        cell.set(43);
        let value = cell.get();  // copies value out
    }

Weak References
---------------

``Weak<T>`` prevents reference cycles that would cause memory leaks. A ``Weak``
reference doesn't contribute to the reference count, so it won't keep the data alive.
Before using a ``Weak``, you must upgrade it to an ``Rc`` or ``Arc``, which returns
``None`` if the data has been dropped. This is essential for parent-child relationships
where children need to reference their parent without creating a cycle:

**C++:**

.. code-block:: cpp

    #include <memory>

    struct Node {
      std::shared_ptr<Node> next;
      std::weak_ptr<Node> prev;  // weak to break cycle
    };

**Rust:**

.. code-block:: rust

    use std::rc::{Rc, Weak};
    use std::cell::RefCell;

    struct Node {
        next: Option<Rc<RefCell<Node>>>,
        prev: Option<Weak<RefCell<Node>>>,  // weak to break cycle
    }

    fn main() {
        let node = Rc::new(RefCell::new(Node {
            next: None,
            prev: None,
        }));

        // Create weak reference
        let weak: Weak<_> = Rc::downgrade(&node);

        // Upgrade to Rc (returns Option)
        if let Some(strong) = weak.upgrade() {
            println!("Node still exists");
        }
    }

Cow (Clone on Write)
--------------------

``Cow<T>`` (Clone on Write) is a smart pointer that can hold either borrowed or owned
data. It delays cloning until mutation is actually needed, which can significantly
improve performance when you often don't need to modify the data. This is useful for
functions that might need to modify their input but usually don't, or for caching
scenarios where you want to avoid unnecessary allocations:

.. code-block:: rust

    use std::borrow::Cow;

    fn process(input: &str) -> Cow<str> {
        if input.contains("bad") {
            // Only allocate if we need to modify
            Cow::Owned(input.replace("bad", "good"))
        } else {
            // No allocation, just borrow
            Cow::Borrowed(input)
        }
    }

    fn main() {
        let s1 = process("hello");        // Borrowed
        let s2 = process("bad word");     // Owned
    }

See Also
--------

- :doc:`rust_ownership` - Ownership fundamentals
- :doc:`rust_thread` - Arc and Mutex with threads
