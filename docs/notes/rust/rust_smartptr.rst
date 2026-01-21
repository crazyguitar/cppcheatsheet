==============
Smart Pointers
==============

.. meta::
   :description: Rust smart pointers compared to C++. Covers Box, Rc, Arc, RefCell, Mutex with C++ unique_ptr, shared_ptr equivalents.
   :keywords: Rust, Box, Rc, Arc, RefCell, Mutex, smart pointers, C++ unique_ptr, shared_ptr

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/smart_pointers <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/smart_pointers>`_

Rust smart pointers provide different ownership and borrowing patterns. Unlike C++,
Rust's ownership system prevents most pointer-related bugs at compile time.

Smart Pointer Comparison
------------------------

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

``Box<T>`` is like ``std::unique_ptr<T>`` - single owner, heap allocated.

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
        // boxed is no longer valid
    }

Use Cases for Box
~~~~~~~~~~~~~~~~~

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

``Rc<T>`` is like ``std::shared_ptr<T>`` but single-threaded only.

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

``Rc<T>`` only allows shared (immutable) access. Use ``RefCell<T>`` for
runtime-checked mutable access:

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

``Arc<T>`` is thread-safe ``Rc<T>``, like ``std::shared_ptr<T>`` with atomic ops.

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

For thread-safe mutable access:

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

``RefCell<T>`` allows mutable borrows checked at runtime instead of compile time:

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

For ``Copy`` types, ``Cell<T>`` is simpler:

.. code-block:: rust

    use std::cell::Cell;

    fn main() {
        let cell = Cell::new(42);

        cell.set(43);
        let value = cell.get();  // copies value out
    }

Weak References
---------------

Prevent reference cycles with ``Weak<T>``:

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

``Cow<T>`` delays cloning until mutation is needed:

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

- :doc:`rust_basic` - Ownership fundamentals
- :doc:`rust_thread` - Arc and Mutex with threads
