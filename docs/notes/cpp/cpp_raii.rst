===================
Resource Management
===================

.. meta::
   :description: C++ RAII (Resource Acquisition Is Initialization) guide covering resource management, constructor failure handling, exception safety guarantees, lock_guard / unique_lock / scoped_lock, scope guards, and two-phase initialization.
   :keywords: C++, RAII, resource acquisition is initialization, resource management, exception safety, constructor failure, failed resource acquisition, basic guarantee, strong guarantee, nothrow, std::lock_guard, std::unique_lock, std::scoped_lock, scope guard, scope_exit, two-phase initialization, RAII wrapper

.. contents:: Table of Contents
    :backlinks: none

Resource Acquisition Is Initialization (RAII) is a fundamental C++ programming idiom
that ties resource management to object lifetime. When an object is constructed, it
acquires resources; when it is destroyed, it releases them. This pattern eliminates
resource leaks and ensures exception safety by leveraging C++'s deterministic
destruction guarantees.

RAII underpins the standard library: ``std::unique_ptr`` owns a heap allocation,
``std::lock_guard`` owns a mutex lock, ``std::fstream`` owns a file handle. Each
releases its resource automatically when the wrapping object goes out of scope,
even if an exception propagates through the scope. For the copy/move machinery
that makes RAII types composable with containers and generic code, see
:doc:`cpp_move`.

Note that `Rust <https://www.rust-lang.org/>`_ enforces ownership at compile time, offering strong safety guarantees.
Interestingly, Rust's ownership model closely resembles C++'s RAII and move semantics
(available since C++11). For example, Rust's ownership transfer is analogous to
``std::move``, and its ``Drop`` trait mirrors C++ destructors.

Initializer Lists
-----------------

:Source: `src/raii/initializer-list <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/initializer-list>`_

``std::initializer_list`` provides a lightweight proxy for accessing an array of
objects. It enables uniform initialization syntax and is commonly used for
constructors that accept a variable number of homogeneous arguments.

The initializer list does not own its elements; it merely provides access to a
temporary array created by the compiler. This makes it efficient for passing
initialization data:

.. code-block:: cpp

    #include <initializer_list>
    #include <iostream>

    template <typename T>
    T sum(std::initializer_list<T> values) {
      T result = 0;
      for (const auto &v : values) {
        result += v;
      }
      return result;
    }

    int main(int argc, char *argv[]) {
      std::cout << sum({1, 2, 3, 4, 5}) << "\n";      // Output: 15
      std::cout << sum({1.5, 2.5, 3.0}) << "\n";      // Output: 7.0
    }

RAII Wrapper
------------

:Source: `src/raii/raii-wrapper <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/raii-wrapper>`_

The RAII wrapper pattern encapsulates resource management in a class, ensuring
resources are properly acquired in the constructor and released in the destructor.
This pattern is the foundation of smart pointers, lock guards, and file handles
in the C++ standard library.

This example demonstrates a simple RAII wrapper for a file handle:

.. code-block:: cpp

    #include <cstdio>
    #include <stdexcept>
    #include <utility>

    class File {
     public:
      explicit File(const char *path, const char *mode) : handle_(std::fopen(path, mode)) {
        if (!handle_) {
          throw std::runtime_error("Failed to open file");
        }
      }

      ~File() {
        if (handle_) {
          std::fclose(handle_);
        }
      }

      // Non-copyable
      File(const File &) = delete;
      File &operator=(const File &) = delete;

      // Movable
      File(File &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
      File &operator=(File &&other) noexcept {
        if (this != &other) {
          if (handle_) std::fclose(handle_);
          handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
      }

      std::FILE *get() const { return handle_; }

     private:
      std::FILE *handle_;
    };

    int main(int argc, char *argv[]) {
      try {
        File f("/tmp/test.txt", "w");
        std::fputs("RAII in action\n", f.get());
      } catch (const std::exception &e) {
        // File automatically closed even if exception thrown
      }
      // File automatically closed when 'f' goes out of scope
    }

.. note::

    Modern C++ provides ``std::unique_ptr`` with custom deleters and ``std::fstream``
    for most resource management needs. Use custom RAII wrappers only when standard
    library alternatives are insufficient.

Constructor Failure and Exception Safety
----------------------------------------

:Source: `src/raii/constructor-failure <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/constructor-failure>`_

When a constructor throws, the object being constructed **never existed** from
the language's perspective: its destructor does **not** run. However,
destructors for any fully-constructed subobjects—base classes and data
members—**do** run, in reverse order of construction. This asymmetry is the
single most important rule for handling failed resource acquisition in C++
constructors. It has two concrete consequences:

1. A raw-pointer member that has been ``new``'d but not yet handed off to an
   owning wrapper leaks when the constructor throws.
2. A smart-pointer or container member that owns its resource cleans up
   automatically, because the *member's* destructor runs even though the
   *enclosing object's* destructor does not.

**Leaky: raw pointer acquired, then throw:**

.. code-block:: cpp

    class LeakyResource {
     public:
      LeakyResource() {
        first_ = new Handle("first");
        throw std::runtime_error("second resource failed");
        // Never reached. ~LeakyResource does not run. first_ leaks.
      }
      ~LeakyResource() { delete first_; }
     private:
      Handle *first_ = nullptr;
    };

**Safe: unique_ptr member cleans up on throw:**

.. code-block:: cpp

    class SafeResource {
     public:
      SafeResource() : first_(std::make_unique<Handle>("first")) {
        throw std::runtime_error("second resource failed");
        // first_ is a fully-constructed member; its destructor runs.
      }
     private:
      std::unique_ptr<Handle> first_;
    };

The rule generalises: acquire every resource through an owning member
(``std::unique_ptr``, ``std::vector``, ``std::string``, a custom RAII
wrapper). Never leave a raw ``new`` exposed between acquisition and
ownership transfer.

**function-try-block:**

A *function-try-block* catches exceptions thrown from the
member-initializer-list. The catch handler cannot suppress the exception—if
the handler does not rethrow explicitly, the exception is rethrown
implicitly at the end of the block. By the time the handler runs, members
have already been destroyed:

.. code-block:: cpp

    class Reported {
     public:
      Reported() try : handle_(std::make_unique<Handle>("reported")) {
        throw std::runtime_error("post-init failure");
      } catch (const std::exception &) {
        // handle_ is already destroyed; exception rethrows implicitly.
      }
     private:
      std::unique_ptr<Handle> handle_;
    };

Exception Safety Guarantees
---------------------------

:Source: `src/raii/exception-safety <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/exception-safety>`_

When an operation throws, the calling code needs to know what state the
affected object is left in. C++ code is categorised into four escalating
guarantees (defined by Abrahams):

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Guarantee
     - Meaning
   * - No-throw
     - The operation never throws. Declared ``noexcept``. Required for move
       operations used by ``std::vector`` reallocation, for swap, and for
       destructors.
   * - Strong
     - The operation either succeeds or has no effect (commit-or-rollback).
       Achieved by the copy-and-swap idiom: do the work on a copy, then
       swap — the swap is nothrow, so the commit cannot fail.
   * - Basic
     - Invariants are preserved and no resources leak, but the object's
       observable state may be partially updated.
   * - None
     - Avoid. Broken invariants are possible.

**Copy-and-swap for the strong guarantee:**

.. code-block:: cpp

    class Counter {
     public:
      void swap(Counter &other) noexcept {
        std::swap(value_, other.value_);
        std::swap(log_, other.log_);
      }

      // Strong guarantee: copy constructs a temporary (may throw and die
      // unused, leaving *this unchanged), then nothrow-swaps it in.
      Counter &operator=(Counter other) noexcept {
        swap(other);
        return *this;
      }
     private:
      int value_ = 0;
      std::vector<int> log_;
    };

**Strong vs basic in practice:**

.. code-block:: cpp

    void Counter::add(int delta, bool fail) {
      log_.push_back(delta);                // happens first
      if (fail) throw std::runtime_error("");
      value_ += delta;
    }

After ``add(5, true)`` throws, ``value_`` is unchanged but ``log_`` has
grown — the object is still valid and usable (basic guarantee), but the
operation did not roll back (no strong guarantee). Reorder the work so the
throwing step runs first, before any visible mutation, to upgrade basic to
strong.

Lock-Based RAII
---------------

:Source: `src/raii/lock-guard <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/lock-guard>`_

The standard library ships three RAII wrappers for mutex locking. Each
acquires the lock in its constructor and releases it in its destructor,
guaranteeing that the mutex is released even if the protected scope exits
via exception.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Wrapper
     - Since
     - Use when
   * - ``std::lock_guard``
     - C++11
     - Simple scoped locking. No manual unlock or deferred acquisition.
   * - ``std::unique_lock``
     - C++11
     - Deferred/timed acquisition, manual unlock, transferable ownership,
       required by ``std::condition_variable::wait``.
   * - ``std::scoped_lock``
     - C++17
     - Locking two or more mutexes at once with built-in deadlock
       avoidance. Prefer over ``std::lock(m1, m2) + lock_guard``.

**Scoped mutual exclusion:**

.. code-block:: cpp

    std::mutex m;
    int counter = 0;
    auto worker = [&] {
      for (int i = 0; i < 10000; ++i) {
        std::lock_guard<std::mutex> guard(m);
        ++counter;
      }
    };

**Deferred acquisition with unique_lock:**

.. code-block:: cpp

    std::unique_lock<std::mutex> lock(m, std::defer_lock);
    // ... some work that does not need the lock ...
    lock.lock();
    // ... critical section ...
    lock.unlock();   // release early; destructor no longer unlocks

**Multiple mutexes without deadlock:**

.. code-block:: cpp

    std::mutex m1, m2;
    {
      std::scoped_lock lock(m1, m2);  // acquires both safely
      // critical section
    }

``std::scoped_lock`` uses a deadlock-avoidance algorithm (equivalent to
``std::lock``) when given multiple mutexes, so two threads locking
``(m1, m2)`` and ``(m2, m1)`` cannot deadlock.

Scope Guards
------------

:Source: `src/raii/scope-guard <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/scope-guard>`_

Writing a dedicated RAII wrapper class is overkill when the cleanup is a
single ad-hoc statement: restore a flag, decrement a counter, roll back a
partially-completed operation. A *scope guard* runs an arbitrary callable
when it leaves scope. It generalises RAII to any cleanup action, without
requiring a new class per resource.

C++26 standardises ``std::scope_exit`` / ``std::scope_fail`` /
``std::scope_success`` (formerly Library Fundamentals TS v3). Until those
are available, a small template is enough:

.. code-block:: cpp

    template <typename F>
    class ScopeGuard {
     public:
      explicit ScopeGuard(F f) : f_(std::move(f)) {}
      ~ScopeGuard() { if (active_) f_(); }
      void dismiss() noexcept { active_ = false; }
      ScopeGuard(const ScopeGuard &) = delete;
      ScopeGuard &operator=(const ScopeGuard &) = delete;
     private:
      F f_;
      bool active_ = true;
    };

    template <typename F>
    ScopeGuard<F> make_scope_guard(F f) { return ScopeGuard<F>(std::move(f)); }

**Rollback on early return:**

.. code-block:: cpp

    void transfer(Account &from, Account &to, int amount) {
      from.debit(amount);
      auto rollback = make_scope_guard([&] { from.credit(amount); });
      to.credit(amount);    // may throw — rollback runs
      rollback.dismiss();   // commit reached; skip the rollback
    }

**std::unique_ptr with custom deleter as a scope guard:**

A ``std::unique_ptr`` with a custom deleter is a ready-made scope guard —
the "pointer" can be any sentinel and the deleter is the cleanup:

.. code-block:: cpp

    auto deleter = [](void *) { /* cleanup */ };
    std::unique_ptr<void, decltype(deleter)> guard(
        reinterpret_cast<void *>(uintptr_t{1}), deleter);

This pattern is especially useful for wrapping C APIs with free-function
cleanup (e.g., ``std::unique_ptr<FILE, decltype(&std::fclose)>``). See
:doc:`cpp_smartpointers` for more on custom deleters.

Two-Phase Initialization: Why RAII Avoids It
--------------------------------------------

:Source: `src/raii/two-phase-init <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/two-phase-init>`_

Some older C++ code — and code translated from languages without
exceptions — uses *two-phase initialization*: a default constructor that
does almost nothing, followed by a separate ``init()`` call that does the
real acquisition and returns a success flag. RAII replaces this pattern.

The two-phase object has a zombie state between construction and
``init()`` where it exists but is not usable. Every method must check an
``initialized_`` flag, callers must remember to call ``init()``, and there
is no type-system guarantee that a constructed object is a usable object:

.. code-block:: cpp

    class TwoPhase {
     public:
      TwoPhase() = default;           // produces a zombie
      bool init(int v) {              // error signalled by bool — easy to ignore
        if (v < 0) return false;
        value_ = v;
        initialized_ = true;
        return true;
      }
      int value() const {
        if (!initialized_) throw std::logic_error("not initialized");
        return value_;
      }
     private:
      int value_ = 0;
      bool initialized_ = false;
    };

The RAII equivalent makes construction all-or-nothing. If acquisition
fails, the constructor throws and the object never existed; if it
succeeds, every method can assume valid state:

.. code-block:: cpp

    class Raii {
     public:
      explicit Raii(int v) : value_(v) {
        if (v < 0) throw std::invalid_argument("v must be non-negative");
      }
      int value() const noexcept { return value_; }   // no flag check
     private:
      int value_;
    };

**Why RAII wins:**

- No invalid intermediate state — the type system enforces "constructed ⇒
  usable".
- Errors cannot be silently ignored; an exception is hard to miss.
- Member functions do not carry an ``if (!initialized_)`` tax.
- Composes cleanly with containers, smart pointers, and generic code,
  which assume constructed objects are usable.

The historical argument against RAII was exception-handling overhead, but
modern compilers make the nothrow path essentially free. Use RAII.

See Also
--------

- :doc:`cpp_move` — special member functions, Rule of Zero/Three/Five, value
  categories, move-only types, perfect forwarding, and common move semantics
  pitfalls.
- :doc:`cpp_smartpointers` — ``std::unique_ptr`` and ``std::shared_ptr``, the
  standard-library RAII wrappers for heap allocations.
- :doc:`cpp_rvo` — return value optimization and copy elision, which interact
  with RAII-managed return values.
