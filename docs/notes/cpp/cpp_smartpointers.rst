==============
Smart Pointers
==============

.. contents:: Table of Contents
    :backlinks: none

Smart pointers, introduced in C++11, provide automatic memory management through
RAII (Resource Acquisition Is Initialization). They ensure that dynamically
allocated memory is properly deallocated when no longer needed, preventing memory
leaks and dangling pointers. Unlike raw pointers, smart pointers automatically
manage the lifetime of the objects they point to, calling the appropriate
destructor or deleter when the pointer goes out of scope. The three main smart
pointer types are ``unique_ptr`` for exclusive ownership, ``shared_ptr`` for
shared ownership with reference counting, and ``weak_ptr`` for non-owning
references that don't prevent deallocation.

std::unique_ptr: Exclusive Ownership
------------------------------------

:Source: `src/smartptr/unique-ptr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/unique-ptr>`_

``std::unique_ptr`` represents exclusive ownership of a dynamically allocated
object. Only one ``unique_ptr`` can own a given object at a time, enforcing
a clear ownership model that prevents accidental sharing. When the ``unique_ptr``
is destroyed, goes out of scope, or is reset, the owned object is automatically
deleted. Because ownership is exclusive, ``unique_ptr`` cannot be copied—only
moved—which transfers ownership from one pointer to another. This makes
``unique_ptr`` ideal for factory functions, RAII wrappers, and any situation
where a single owner is responsible for an object's lifetime. It has zero
overhead compared to raw pointers when using the default deleter.

.. code-block:: cpp

    #include <memory>
    #include <iostream>

    int main() {
      auto ptr = std::make_unique<int>(42);
      std::cout << *ptr << "\n";  // Output: 42

      // Transfer ownership
      auto ptr2 = std::move(ptr);
      // ptr is now nullptr

      // Array support
      auto arr = std::make_unique<int[]>(5);
      arr[0] = 10;
    }

std::shared_ptr: Shared Ownership
---------------------------------

:Source: `src/smartptr/shared-ptr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/shared-ptr>`_

``std::shared_ptr`` allows multiple pointers to share ownership of an object
through reference counting. Each ``shared_ptr`` maintains a pointer to a control
block that tracks how many ``shared_ptr`` instances own the object. When a
``shared_ptr`` is copied, the reference count increments; when one is destroyed
or reset, the count decrements. The managed object is deleted only when the
last owning ``shared_ptr`` is destroyed (when the count reaches zero). This
makes ``shared_ptr`` suitable for scenarios where multiple parts of your code
need access to the same object and the lifetime cannot be determined statically.
However, the reference counting adds overhead, so prefer ``unique_ptr`` when
exclusive ownership is sufficient.

.. code-block:: cpp

    #include <memory>
    #include <iostream>

    int main() {
      auto ptr1 = std::make_shared<int>(42);
      std::cout << "Count: " << ptr1.use_count() << "\n";  // 1

      {
        auto ptr2 = ptr1;  // Share ownership
        std::cout << "Count: " << ptr1.use_count() << "\n";  // 2
      }  // ptr2 destroyed, count decremented

      std::cout << "Count: " << ptr1.use_count() << "\n";  // 1
    }

std::weak_ptr: Non-Owning Observer
----------------------------------

:Source: `src/smartptr/weak-ptr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/weak-ptr>`_

``std::weak_ptr`` holds a non-owning reference to an object managed by
``shared_ptr``. Unlike ``shared_ptr``, it doesn't contribute to the reference
count, so it doesn't prevent the object from being deleted. This makes
``weak_ptr`` essential for breaking circular references that would otherwise
cause memory leaks. It's also useful for implementing caches, observer patterns,
and any situation where you need to check if an object still exists without
extending its lifetime. To access the object, call ``lock()`` which returns
a ``shared_ptr``—if the object still exists, you get a valid ``shared_ptr``
that temporarily extends the object's lifetime; if it's been deleted, you
get an empty ``shared_ptr``. The ``expired()`` method provides a quick check
without creating a ``shared_ptr``.

.. code-block:: cpp

    #include <memory>
    #include <iostream>

    int main() {
      std::weak_ptr<int> weak;

      {
        auto shared = std::make_shared<int>(42);
        weak = shared;

        if (auto locked = weak.lock()) {
          std::cout << *locked << "\n";  // 42
        }
      }  // shared destroyed

      std::cout << "Expired: " << weak.expired() << "\n";  // 1 (true)
    }

std::make_unique and std::make_shared
-------------------------------------

:Source: `src/smartptr/make-functions <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/make-functions>`_

Always prefer ``std::make_unique`` (C++14) and ``std::make_shared`` over direct
``new`` expressions. These factory functions provide several important benefits.
First, they are exception-safe: if an exception is thrown during argument
evaluation, no memory is leaked. Second, ``make_shared`` is more efficient
because it performs a single memory allocation for both the object and the
control block, improving cache locality and reducing allocation overhead.
Third, they eliminate the redundancy of writing the type twice. Fourth, they
prevent potential memory leaks in complex expressions where the evaluation
order of function arguments is unspecified (before C++17).

.. code-block:: cpp

    #include <memory>

    struct Widget { int x, y; };

    int main() {
      // Preferred: exception-safe, efficient
      auto u = std::make_unique<Widget>();
      auto s = std::make_shared<Widget>();

      // Avoid: potential leak if exception thrown between allocations
      // std::unique_ptr<Widget> u2(new Widget());
    }

**Exception safety issue:**

.. code-block:: cpp

    void process(std::unique_ptr<A> a, std::unique_ptr<B> b);

    // DANGEROUS: evaluation order unspecified before C++17
    process(std::unique_ptr<A>(new A()), std::unique_ptr<B>(new B()));

    // SAFE: make_unique ensures no leak
    process(std::make_unique<A>(), std::make_unique<B>());

Custom Deleters
---------------

:Source: `src/smartptr/custom-deleter <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/custom-deleter>`_

Smart pointers can use custom deleters for resources that require special
cleanup beyond simple ``delete``. This is essential when working with C library
resources (file handles, sockets, database connections), memory from custom
allocators, or any resource that needs specific cleanup logic. For ``unique_ptr``,
the deleter type is part of the pointer type, which means different deleters
create different types. For ``shared_ptr``, the deleter is type-erased and
stored in the control block, so different deleters don't affect the pointer
type—this provides more flexibility but adds slight overhead.

.. code-block:: cpp

    #include <cstdio>
    #include <memory>

    int main() {
      // unique_ptr with custom deleter (deleter is part of type)
      auto file = std::unique_ptr<FILE, int(*)(FILE*)>(
          fopen("test.txt", "w"), fclose);

      if (file) {
        fprintf(file.get(), "Hello\n");
      }  // fclose called automatically

      // shared_ptr with custom deleter (type-erased)
      std::shared_ptr<FILE> file2(fopen("test2.txt", "w"), fclose);
    }

**Lambda as deleter:**

.. code-block:: cpp

    auto deleter = [](int* p) {
      std::cout << "Deleting " << *p << "\n";
      delete p;
    };

    std::unique_ptr<int, decltype(deleter)> ptr(new int(42), deleter);

enable_shared_from_this
-----------------------

:Source: `src/smartptr/enable-shared-from-this <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/enable-shared-from-this>`_

When an object managed by ``shared_ptr`` needs to obtain a ``shared_ptr`` to
itself (for example, to register itself as a callback or pass itself to another
function), inherit from ``std::enable_shared_from_this<T>``. This base class
provides the ``shared_from_this()`` method that returns a ``shared_ptr`` sharing
ownership with existing owners. This is necessary because creating a new
``shared_ptr`` directly from ``this`` would create a separate ownership group
with its own reference count, leading to double deletion when both groups
reach zero. The ``weak_from_this()`` method (C++17) returns a ``weak_ptr``
for cases where you don't want to extend the lifetime.

.. code-block:: cpp

    #include <memory>
    #include <iostream>

    class Widget : public std::enable_shared_from_this<Widget> {
     public:
      std::shared_ptr<Widget> get_shared() {
        return shared_from_this();
      }
    };

    int main() {
      auto w = std::make_shared<Widget>();
      auto w2 = w->get_shared();
      std::cout << "Count: " << w.use_count() << "\n";  // 2
    }

Aliasing Constructor (shared_ptr)
---------------------------------

:Source: `src/smartptr/aliasing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/smartptr/aliasing>`_

The aliasing constructor creates a ``shared_ptr`` that shares ownership with
another ``shared_ptr`` but points to a different object (typically a member
of the owned object). This is useful when you need to pass a pointer to a
member while ensuring the parent object stays alive. The aliasing ``shared_ptr``
increments the reference count of the original owner, so the parent object
won't be deleted while the aliasing pointer exists. This pattern is commonly
used when interfacing with APIs that expect a ``shared_ptr`` to a specific
type but you have a ``shared_ptr`` to a containing object.

.. code-block:: cpp

    #include <memory>
    #include <iostream>

    struct Inner { int value = 42; };
    struct Outer { Inner inner; };

    int main() {
      auto outer = std::make_shared<Outer>();

      // Aliasing: shares ownership with outer, points to inner
      std::shared_ptr<Inner> inner(outer, &outer->inner);

      std::cout << inner->value << "\n";  // 42
      std::cout << outer.use_count() << "\n";  // 2
    }

std::make_unique_for_overwrite (C++20)
--------------------------------------

C++20 added ``std::make_unique_for_overwrite`` which creates a ``unique_ptr``
with default-initialized (not value-initialized) memory. For scalar types and
arrays of scalars, this means the memory is left uninitialized rather than
being zeroed. This optimization is useful when you plan to immediately overwrite
all the memory anyway, such as when reading data from a file or network into
a buffer. Avoiding unnecessary zero-initialization can provide measurable
performance improvements for large allocations.

.. code-block:: cpp

    #include <memory>

    int main() {
      // Value-initialized (zeroed): slower
      auto arr1 = std::make_unique<int[]>(1000);

      // Default-initialized (uninitialized): faster
      auto arr2 = std::make_unique_for_overwrite<int[]>(1000);
      // Must initialize before reading!
    }

std::shared_ptr Array Support (C++17/C++20)
-------------------------------------------

C++17 added ``shared_ptr<T[]>`` support with proper array semantics, including
``operator[]`` for element access and automatic use of ``delete[]`` for cleanup.
C++20 completed the feature by adding ``std::make_shared`` overloads for arrays.
Before C++17, using ``shared_ptr`` with arrays required a custom deleter to
call ``delete[]`` instead of ``delete``, which was error-prone and verbose.

.. code-block:: cpp

    #include <memory>

    int main() {
      // C++20: make_shared for arrays
      auto arr = std::make_shared<int[]>(5);
      arr[0] = 10;

      // C++17: shared_ptr with array type
      std::shared_ptr<int[]> arr2(new int[5]);
    }

Atomic Operations on shared_ptr (C++20)
---------------------------------------

C++20 introduced ``std::atomic<std::shared_ptr<T>>`` for thread-safe atomic
operations on shared pointers. This provides a proper atomic type that supports
all standard atomic operations (load, store, exchange, compare_exchange) with
correct memory ordering semantics. Before C++20, thread-safe access to
``shared_ptr`` required using the free functions ``std::atomic_load`` and
``std::atomic_store``, which were less convenient and had subtle correctness
issues. The new atomic specialization is essential for lock-free data structures
and concurrent algorithms that need to safely share ownership across threads.

.. code-block:: cpp

    #include <atomic>
    #include <memory>

    std::atomic<std::shared_ptr<int>> atomic_ptr;

    void writer() {
      atomic_ptr.store(std::make_shared<int>(42));
    }

    void reader() {
      if (auto p = atomic_ptr.load()) {
        // Use *p safely
      }
    }

out_ptr and inout_ptr (C++23)
-----------------------------

C++23 added ``std::out_ptr`` and ``std::inout_ptr`` adaptors for interfacing
smart pointers with C APIs that return pointers through output parameters.
These adaptors eliminate the error-prone manual pattern of releasing the smart
pointer, calling the C function, and then resetting the smart pointer with
the result. ``out_ptr`` is for functions that only output a pointer (the smart
pointer should be empty beforehand), while ``inout_ptr`` is for functions that
may release an existing resource before outputting a new one.

.. code-block:: cpp

    #include <memory>

    // C API that allocates and returns through output parameter
    extern "C" int create_resource(void** out);
    extern "C" void destroy_resource(void* p);

    int main() {
      std::unique_ptr<void, decltype(&destroy_resource)> ptr(nullptr, destroy_resource);

      // C++23: out_ptr handles release and reset automatically
      create_resource(std::out_ptr(ptr));
    }

Common Pitfalls
---------------

Smart pointers eliminate many memory management bugs, but they introduce their
own set of pitfalls that can cause crashes, memory leaks, or undefined behavior.
Understanding these common mistakes helps you use smart pointers correctly.

**Creating shared_ptr from raw pointer multiple times:**

This is one of the most dangerous mistakes. When you create a ``shared_ptr``
from a raw pointer, it creates a new control block with a reference count of 1.
If you create another ``shared_ptr`` from the same raw pointer, it creates a
completely separate control block, also with count 1. When both ``shared_ptr``
instances are destroyed, each thinks it's the last owner and calls ``delete``,
resulting in a double-free crash. Always create ``shared_ptr`` using
``make_shared`` or by copying/moving existing ``shared_ptr`` instances.

.. code-block:: cpp

    int* raw = new int(42);
    std::shared_ptr<int> p1(raw);
    std::shared_ptr<int> p2(raw);  // BUG: double delete!

    // Correct approach:
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;  // Shares ownership correctly

**Circular references with shared_ptr:**

When two or more objects hold ``shared_ptr`` references to each other, they
create a cycle that prevents either from being deleted. Each object's reference
count never reaches zero because the other object holds a reference. This is
a memory leak that persists until the program ends. The solution is to use
``weak_ptr`` for at least one direction of the relationship, breaking the
cycle. Common scenarios include parent-child relationships (parent owns child
with ``shared_ptr``, child references parent with ``weak_ptr``), doubly-linked
lists, and graph structures.

.. code-block:: cpp

    struct Node {
      std::shared_ptr<Node> next;  // Creates cycle, memory leak!
    };

    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();
    a->next = b;
    b->next = a;  // Cycle: neither a nor b will ever be deleted

    // Fix: use weak_ptr for one direction
    struct FixedNode {
      std::shared_ptr<FixedNode> next;
      std::weak_ptr<FixedNode> prev;  // Breaks the cycle
    };

**Calling shared_from_this before shared_ptr exists:**

The ``shared_from_this()`` method only works after at least one ``shared_ptr``
to the object has been created. Internally, ``enable_shared_from_this`` stores
a ``weak_ptr`` that is initialized when the first ``shared_ptr`` is created.
If you call ``shared_from_this()`` in the constructor (before any ``shared_ptr``
exists) or on a stack-allocated object (which should never be managed by
``shared_ptr``), it throws ``std::bad_weak_ptr``. Always ensure the object
is managed by ``shared_ptr`` before calling ``shared_from_this()``.

.. code-block:: cpp

    class Widget : public std::enable_shared_from_this<Widget> {
     public:
      Widget() {
        // BUG: no shared_ptr owns this yet
        // auto self = shared_from_this();  // throws bad_weak_ptr
      }

      void init() {
        // OK: called after make_shared creates the shared_ptr
        auto self = shared_from_this();
      }
    };

    // Correct usage:
    auto w = std::make_shared<Widget>();
    w->init();  // Now shared_from_this() works

**Storing this in a shared_ptr:**

Never create a ``shared_ptr`` directly from ``this``. This creates a new
ownership group that doesn't know about any existing ``shared_ptr`` owners,
leading to double deletion. Use ``enable_shared_from_this`` instead.

.. code-block:: cpp

    class Bad {
     public:
      std::shared_ptr<Bad> get_ptr() {
        return std::shared_ptr<Bad>(this);  // BUG: double delete!
      }
    };

Smart Pointer Comparison
------------------------

The following table summarizes when to use each smart pointer type. Choose
the simplest pointer that meets your ownership requirements—prefer ``unique_ptr``
for its zero overhead and clear ownership semantics, use ``shared_ptr`` only
when ownership truly needs to be shared, and use ``weak_ptr`` to observe
without owning.

.. list-table::
   :header-rows: 1

   * - Type
     - Ownership
     - Use Case
   * - ``unique_ptr``
     - Exclusive
     - Single owner, factory functions, RAII wrappers
   * - ``shared_ptr``
     - Shared
     - Multiple owners, shared resources, caches
   * - ``weak_ptr``
     - None
     - Breaking cycles, observers, caches with expiration
